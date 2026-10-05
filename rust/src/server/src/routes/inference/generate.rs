// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod compact;
mod convert;
mod render;
mod routed_experts;
mod types;
mod validate;

use std::collections::HashMap;
use std::convert::Infallible;
use std::result::Result;
use std::sync::Arc;

use asynk_strim_attr::{TryYielder, try_stream};
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::{Stream, StreamExt as _, pin_mut};
use thiserror_ext::AsReport as _;
use tracing::{error, info, trace};
use tracing_futures::Instrument as _;
use vllm_engine_core_client::protocol::logprobs::{Logprobs, PositionLogprobs};
use vllm_llm::{
    CollectedGenerateOutput, FinishReason, GenerateOutput, GenerateOutputStreamExt as _,
    LogprobsAccumulator, RoutedExperts, TokenUsage,
};

use self::compact::{CompactLogprobsAccumulator, LogprobsFormat, encode_compact};
use self::convert::{ResponseOptions, prepare_generate_request};
use self::render::{ChoiceLogprobs, GenerateEnvelope, RoutedExpertsField, generate_response};
use self::routed_experts::RoutedExpertsEncoder;
use self::types::{GenerateLogprob, GenerateResponseStreamChoice, GenerateStreamResponse};
pub(crate) use self::types::{GenerateRequest, GenerateSamplingParams};
pub(crate) use self::validate::validate_request_compat;
use crate::config::ApiServerOptions;
use crate::error::{ApiError, bail_server_error, server_error, text_submit_error};
use crate::routes::openai::utils::logprobs::clamp_logprob;
use crate::routes::openai::utils::types::{ChatLogProbs, ChatLogProbsContent, TopLogProb, Usage};
use crate::routes::openai::utils::validated_json::ValidatedJson;
use crate::state::AppState;
use crate::utils::resolve_request_context;

/// Validate one token-in/token-out request and proxy it into the shared
/// `vllm-text` stack.
pub async fn generate(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    ValidatedJson(mut body): ValidatedJson<GenerateRequest>,
) -> Response {
    let request_context = resolve_request_context(&headers, body.request_id.as_deref());
    let lora_resolution = state.resolve_model_with_loras(body.model.as_deref()).await;

    let mm_features = if let Some(parts) = body.content_parts.take() {
        match state.chat.prepare_media(parts, &mut body.token_ids).await {
            Ok(features) => features,
            Err(e) => {
                return ApiError::invalid_request(
                    format!("failed to resolve content_parts: {}", e.as_report()),
                    Some("content_parts"),
                )
                .into_response();
            }
        }
    } else {
        None
    };

    let prepared =
        match prepare_generate_request(body, &lora_resolution, request_context, mm_features) {
            Ok(prepared) => prepared,
            Err(error) => return error.into_response(),
        };
    let request_span = tracing::info_span!(
        "generate",
        request_id = %prepared.request_id,
        engine_request_id = tracing::field::Empty,
    );

    let api_server_options = state.api_server_options;
    let stream = prepared.stream;
    let raw_stream = match state
        .chat
        .text()
        .generate_raw(prepared.text_request)
        .instrument(request_span.clone())
        .await
    {
        Ok(stream) => stream,
        Err(error) => {
            return text_submit_error("failed to submit raw generate request", error)
                .into_response();
        }
    };

    if stream {
        let chunk_stream = generate_chunk_stream(
            raw_stream,
            prepared.request_id,
            api_server_options,
            prepared.options,
        );
        let sse_stream = generate_sse_stream(chunk_stream).instrument(request_span);

        return Sse::new(sse_stream).into_response();
    }

    let options = prepared.options;
    let collect_error = |error: vllm_llm::Error| {
        server_error!(
            "failed to collect raw generate response: {}",
            error.to_report_string()
        )
        .into_response()
    };

    let result =
        match collect_response(raw_stream, prepared.request_id, api_server_options, options)
            .instrument(request_span.clone())
            .await
        {
            Ok(result) => result,
            Err(error) => return collect_error(error),
        };

    match result {
        // In the request span so the body render task inherits it.
        Ok((envelope, logprobs)) => request_span.in_scope(|| generate_response(envelope, logprobs)),
        Err(error) => error.into_response(),
    }
}

/// Collect a non-streaming raw generate request and build the response
/// envelope and output logprobs (outer error: the engine stream failed).
async fn collect_response(
    raw_stream: impl futures::Stream<Item = vllm_llm::Result<GenerateOutput>> + Send,
    request_id: String,
    api_server_options: ApiServerOptions,
    options: ResponseOptions,
) -> vllm_llm::Result<Result<(GenerateEnvelope, ChoiceLogprobs), ApiError>> {
    Ok(match options.logprobs_format {
        LogprobsFormat::OpenAi => {
            let accumulator = WithRoutedExperts::new(None::<Logprobs>);
            let (collected, accumulator) = raw_stream.collect_output_into(accumulator).await?;
            let WithRoutedExperts { inner, routed } = accumulator;
            openai_choice_logprobs(inner, options.include_logprobs).and_then(|logprobs| {
                let envelope =
                    collect_generate(collected, routed, request_id, api_server_options, options)?;
                Ok((envelope, logprobs))
            })
        }
        LogprobsFormat::Compact => {
            let accumulator = WithRoutedExperts::new(
                CompactLogprobsAccumulator::new(options.logprobs_slots)
                    .with_switches(!options.compact_skip_sampled, !options.compact_skip_ranks),
            );
            let (collected, accumulator) = raw_stream.collect_output_into(accumulator).await?;
            let WithRoutedExperts { inner, routed } = accumulator;
            compact_choice_logprobs(
                inner,
                options.include_compact_logprobs,
                collected.token_ids.len(),
            )
            .and_then(|logprobs| {
                let envelope =
                    collect_generate(collected, routed, request_id, api_server_options, options)?;
                Ok((envelope, logprobs))
            })
        }
    })
}

/// A logprobs accumulator plus the request's routed-experts encoder.
struct WithRoutedExperts<L> {
    inner: L,
    routed: RoutedExpertsEncoder,
}

impl<L> WithRoutedExperts<L> {
    fn new(inner: L) -> Self {
        Self {
            inner,
            routed: RoutedExpertsEncoder::default(),
        }
    }
}

impl<L: LogprobsAccumulator> LogprobsAccumulator for WithRoutedExperts<L> {
    fn extend(&mut self, step: Logprobs) {
        self.inner.extend(step);
    }

    fn observe_output(&mut self, new_tokens: usize, logprob_positions: Option<usize>) {
        self.inner.observe_output(new_tokens, logprob_positions);
    }

    fn num_positions(&self) -> usize {
        self.inner.num_positions()
    }

    fn extend_routed_experts(&mut self, routed_experts: RoutedExperts) {
        self.routed.push(routed_experts);
    }
}

/// `choices[0].routed_experts`: present when the server returns routed
/// experts (`--enable-return-routed-experts`) or the engine sent any.
fn routed_experts_field(
    routed: RoutedExpertsEncoder,
    enabled: bool,
) -> Result<RoutedExpertsField, ApiError> {
    if !enabled && !routed.has_data() {
        return Ok(RoutedExpertsField::Omitted);
    }
    match routed.finish().map_err(ApiError::server_error)? {
        Some(npy) => Ok(RoutedExpertsField::Npy(npy)),
        None => Ok(RoutedExpertsField::Null),
    }
}

/// Validate collected output logprobs for the default OpenAI rendering.
fn openai_choice_logprobs(
    logprobs: Option<Logprobs>,
    include_logprobs: bool,
) -> Result<ChoiceLogprobs, ApiError> {
    if !include_logprobs {
        return Ok(ChoiceLogprobs::None);
    }
    let logprobs = logprobs.ok_or_else(|| {
        ApiError::server_error(
            "raw generate response requested logprobs but generation returned none".to_string(),
        )
    })?;
    // Rendering is streamed after the status line, so reject malformed rows
    // up front exactly like the eager conversion did.
    if logprobs.positions.iter().any(|position| position.entries.is_empty()) {
        return Err(empty_position_error());
    }
    Ok(ChoiceLogprobs::OpenAi(logprobs.positions))
}

fn compact_choice_logprobs(
    accumulator: CompactLogprobsAccumulator,
    include_logprobs: bool,
    output_tokens: usize,
) -> Result<ChoiceLogprobs, ApiError> {
    if !include_logprobs {
        return Ok(ChoiceLogprobs::Compact(None));
    }
    // Every generated token must have exactly one position (checked per
    // engine output by the accumulator and again here). A request that
    // produced no tokens (e.g. aborted while waiting: the abort output has
    // no tokens and no payload) gets an empty block with the requested width.
    let block = accumulator.finish().map_err(ApiError::server_error)?;
    if block.num_positions != output_tokens {
        return Err(ApiError::server_error(format!(
            "raw generate logprobs cover {} positions but generation returned {output_tokens} tokens",
            block.num_positions
        )));
    }
    Ok(ChoiceLogprobs::Compact(Some(block)))
}

fn empty_position_error() -> ApiError {
    ApiError::server_error(
        "raw generate logprobs position unexpectedly had no token candidates".to_string(),
    )
}

#[try_stream]
async fn generate_chunk_stream(
    stream: impl Stream<Item = vllm_llm::Result<GenerateOutput>>,
    request_id: String,
    ApiServerOptions {
        enable_log_requests,
        enable_prompt_tokens_details,
        enable_return_routed_experts,
        ..
    }: ApiServerOptions,
    ResponseOptions {
        include_usage,
        include_continuous_usage,
        include_logprobs,
        // Ignored: raw generate streaming has no prompt-logprobs wire shape.
        include_prompt_logprobs: _,
        logprobs_format,
        include_compact_logprobs,
        logprobs_slots,
        compact_skip_sampled,
        compact_skip_ranks,
    }: ResponseOptions,
    mut y: TryYielder<GenerateStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let mut prompt_tokens = None;
    let mut usage = TokenUsage::default();
    // Like Python, routed experts are returned once, on the finishing chunk,
    // as the concatenation of every chunk the engine sent.
    let mut routed = Some(RoutedExpertsEncoder::default());

    while let Some(next) = stream.next().await {
        match next {
            Ok(output) => {
                if prompt_tokens.is_none() {
                    prompt_tokens =
                        output.prompt_info.as_ref().map(|info| info.prompt_token_ids.len());
                }
                usage.prompt_token_count = prompt_tokens.unwrap_or_default();
                usage.cached_token_count = usage.cached_token_count.max(output.cached_token_count);

                let token_ids = output.token_ids;
                usage.output_token_count = usage.output_token_count.saturating_add(token_ids.len());
                let finish_reason = output.finish_reason;

                if matches!(finish_reason.as_ref(), Some(FinishReason::Error)) {
                    bail_server_error!("Internal server error");
                }

                // Compact: every output (including zero-token ones that are
                // skipped or terminal) carries exactly one position per token,
                // as in the non-streaming collector.
                if logprobs_format == LogprobsFormat::Compact && include_compact_logprobs {
                    let positions = output.logprobs.as_ref().map_or(0, Logprobs::len);
                    if positions != token_ids.len() {
                        bail_server_error!(
                            "raw generate output carried {positions} logprob positions for {} new tokens",
                            token_ids.len()
                        );
                    }
                }

                if let Some(finish_reason) = finish_reason.as_ref()
                    && enable_log_requests
                {
                    info!(
                        stream = true,
                        prompt_tokens = usage.prompt_token_count,
                        output_tokens = usage.output_token_count,
                        finish_reason = finish_reason.as_str(),
                        "generate finished"
                    );
                }

                if let (Some(chunk), Some(routed)) = (output.routed_experts, routed.as_mut()) {
                    routed.push(chunk);
                }

                if token_ids.is_empty() && finish_reason.is_none() {
                    continue;
                }

                let routed_experts = match (finish_reason.is_some(), routed.take()) {
                    (true, Some(routed)) => {
                        match routed_experts_field(routed, enable_return_routed_experts)? {
                            RoutedExpertsField::Omitted => None,
                            RoutedExpertsField::Null => Some(None),
                            RoutedExpertsField::Npy(npy) => Some(Some(npy.to_base64_string())),
                        }
                    }
                    (false, taken) => {
                        routed = taken;
                        None
                    }
                    (true, None) => None,
                };

                let wants_logprobs = match logprobs_format {
                    LogprobsFormat::OpenAi => include_logprobs,
                    LogprobsFormat::Compact => include_compact_logprobs,
                };
                let (logprobs, compact_logprobs) = if wants_logprobs && !token_ids.is_empty() {
                    let logprobs = output.logprobs.ok_or_else(|| {
                        server_error!(
                            "raw generate stream requested logprobs but generation returned none"
                        )
                    })?;
                    match logprobs_format {
                        LogprobsFormat::OpenAi => {
                            (Some(Some(raw_logprobs_to_openai_chat(&logprobs)?)), None)
                        }
                        // Per-chunk compact block covering this chunk's positions.
                        LogprobsFormat::Compact => (
                            None,
                            Some(
                                encode_compact(
                                    logprobs,
                                    token_ids.len(),
                                    logprobs_slots,
                                    !compact_skip_sampled,
                                    !compact_skip_ranks,
                                )
                                .map_err(ApiError::server_error)?,
                            ),
                        ),
                    }
                } else {
                    (None, None)
                };
                // Compact chunks carry an explicit `"logprobs": null`; the
                // default stream keeps omitting the key when absent.
                let logprobs = match logprobs_format {
                    LogprobsFormat::OpenAi => logprobs,
                    LogprobsFormat::Compact => Some(None),
                };

                y.yield_ok(GenerateStreamResponse {
                    request_id: request_id.clone(),
                    choices: vec![GenerateResponseStreamChoice {
                        index: 0,
                        logprobs,
                        finish_reason: finish_reason.map(|reason| reason.as_str().to_string()),
                        token_ids,
                        routed_experts,
                        compact_logprobs,
                    }],
                    usage: include_continuous_usage
                        .then(|| Usage::from_token_usage(usage, enable_prompt_tokens_details)),
                })
                .await;
            }
            Err(error) => {
                error!(
                    error = %error.as_report(),
                    "raw generate stream failed"
                );
                bail_server_error!("{}", error.to_report_string());
            }
        }
    }

    if include_usage {
        y.yield_ok(GenerateStreamResponse {
            request_id,
            choices: Vec::new(),
            usage: Some(Usage::from_token_usage(usage, enable_prompt_tokens_details)),
        })
        .await;
    }

    Ok(())
}

/// Build everything in the non-streaming response except the output logprobs
/// (which the caller takes out of `collected` beforehand).
fn collect_generate(
    collected: CollectedGenerateOutput,
    routed: RoutedExpertsEncoder,
    request_id: String,
    ApiServerOptions {
        enable_log_requests,
        enable_return_routed_experts,
        ..
    }: ApiServerOptions,
    ResponseOptions {
        include_prompt_logprobs,
        ..
    }: ResponseOptions,
) -> Result<GenerateEnvelope, ApiError> {
    let prompt_logprobs = if include_prompt_logprobs {
        match collected.prompt_logprobs.as_ref() {
            Some(prompt_logprobs) => Some(raw_prompt_logprobs_to_maps(prompt_logprobs)),
            // A single-token prompt has no scored positions; same mapping
            // as /v1/completions.
            None if collected.prompt_token_ids.len() == 1 => Some(vec![None]),
            None => {
                return Err(ApiError::server_error(
                    "raw generate response requested prompt_logprobs but generation returned none"
                        .to_string(),
                ));
            }
        }
    } else {
        None
    };
    let finish_reason = collected.finish_reason.as_str().to_string();
    let routed_experts = routed_experts_field(routed, enable_return_routed_experts)?;

    if enable_log_requests {
        info!(
            prompt_tokens = collected.prompt_token_ids.len(),
            output_tokens = collected.token_ids.len(),
            %finish_reason,
            "generate finished"
        );
    }

    Ok(GenerateEnvelope {
        request_id,
        finish_reason,
        token_ids: collected.token_ids,
        prompt_logprobs,
        kv_transfer_params: collected.kv_transfer_params,
        ec_transfer_params: collected.ec_transfer_params,
        routed_experts,
    })
}

fn raw_logprobs_to_openai_chat(logprobs: &Logprobs) -> Result<ChatLogProbs, ApiError> {
    let content = logprobs
        .positions
        .iter()
        .map(position_to_chat_logprobs_content)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(ChatLogProbs {
        content: Some(content),
    })
}

fn raw_prompt_logprobs_to_maps(
    prompt_logprobs: &Logprobs,
) -> Vec<Option<HashMap<u32, GenerateLogprob>>> {
    std::iter::once(None)
        .chain(
            prompt_logprobs
                .positions
                .iter()
                .map(|position| Some(position_to_logprob_map(position))),
        )
        .collect()
}

fn position_to_chat_logprobs_content(
    position: &PositionLogprobs,
) -> Result<ChatLogProbsContent, ApiError> {
    let chosen = position.entries.first().ok_or_else(empty_position_error)?;
    let token = format_token_id(chosen.token_id);

    Ok(ChatLogProbsContent {
        token: token.clone(),
        logprob: clamp_logprob(chosen.logprob),
        bytes: Some(token.as_bytes().to_vec()),
        top_logprobs: position
            .entries
            .iter()
            .map(|entry| {
                let token = format_token_id(entry.token_id);
                TopLogProb {
                    token: token.clone(),
                    logprob: clamp_logprob(entry.logprob),
                    bytes: Some(token.into_bytes()),
                }
            })
            .collect(),
    })
}

fn position_to_logprob_map(position: &PositionLogprobs) -> HashMap<u32, GenerateLogprob> {
    position
        .entries
        .iter()
        .map(|entry| {
            (
                entry.token_id,
                GenerateLogprob {
                    logprob: clamp_logprob(entry.logprob),
                    rank: Some(entry.rank),
                    decoded_token: Some(format_token_id(entry.token_id)),
                },
            )
        })
        .collect()
}

fn format_token_id(token_id: u32) -> String {
    format!("token_id:{token_id}")
}

/// Convert one raw-generate chunk stream into SSE events.
#[try_stream]
async fn generate_sse_stream(
    stream: impl Stream<Item = Result<GenerateStreamResponse, ApiError>>,
    mut y: TryYielder<Event, Infallible>,
) -> Result<(), Infallible> {
    pin_mut!(stream);

    while let Some(next) = stream.next().await {
        match next {
            Ok(chunk) => y.yield_ok(to_sse_event(&chunk)).await,
            Err(error) => {
                y.yield_ok(to_error_sse_event(&error)).await;
                break;
            }
        }
    }

    y.yield_ok(done_sse_event()).await;
    Ok(())
}

fn to_sse_event(chunk: &GenerateStreamResponse) -> Event {
    let payload = serde_json::to_string(chunk).expect("generate chunk must serialize to JSON");
    trace!(payload, "generate emitting chunk");
    Event::default().data(payload)
}

fn to_error_sse_event(error: &ApiError) -> Event {
    let payload = serde_json::to_string(&error.to_error_response())
        .expect("ErrorResponse must serialize to JSON");
    trace!(payload, "generate emitting error");
    Event::default().data(payload)
}

fn done_sse_event() -> Event {
    trace!("generate emitting done");
    Event::default().data("[DONE]")
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures::{TryStreamExt as _, stream};
    use vllm_llm::GeneratePromptInfo;

    use super::*;

    #[tokio::test]
    async fn generate_chunk_stream_captures_late_prompt_info() {
        let stream = stream::iter(vec![
            Ok(GenerateOutput {
                request_id: String::new(),
                prompt_info: None,
                token_ids: Vec::new(),
                logprobs: None,
                finish_reason: None,
                cached_token_count: 0,
                kv_transfer_params: None,
                ec_transfer_params: None,
                routed_experts: None,
            }),
            Ok(GenerateOutput {
                request_id: String::new(),
                prompt_info: Some(GeneratePromptInfo {
                    prompt_token_ids: Arc::from([11_u32, 22_u32]),
                    prompt_logprobs: None,
                }),
                token_ids: vec![33],
                logprobs: None,
                finish_reason: Some(FinishReason::stop_eos()),
                cached_token_count: 2,
                kv_transfer_params: None,
                ec_transfer_params: None,
                routed_experts: None,
            }),
        ]);

        let chunks: Vec<_> = generate_chunk_stream(
            stream,
            "raw-stream".to_string(),
            ApiServerOptions {
                enable_prompt_tokens_details: true,
                ..Default::default()
            },
            ResponseOptions {
                include_usage: true,
                include_continuous_usage: true,
                ..Default::default()
            },
        )
        .try_collect()
        .await
        .expect("collect chunks");

        assert_eq!(chunks.len(), 2);
        assert_eq!(
            chunks[0].usage.as_ref().expect("chunk usage").prompt_tokens,
            2
        );
        assert_eq!(
            chunks[0]
                .usage
                .as_ref()
                .expect("chunk usage")
                .prompt_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens),
            Some(2)
        );
        assert_eq!(
            chunks[1].usage.as_ref().expect("final usage").prompt_tokens,
            2
        );
        assert_eq!(
            chunks[1]
                .usage
                .as_ref()
                .expect("final usage")
                .prompt_tokens_details
                .as_ref()
                .map(|details| details.cached_tokens),
            Some(2)
        );
    }

    use axum::body::to_bytes;
    use serde_json::json;
    use vllm_engine_core_client::protocol::logprobs::TokenLogprob;

    use super::compact::tests::{decode_compact, position};
    use super::types::{GenerateResponse, GenerateResponseChoice};

    fn tricky_positions() -> Vec<PositionLogprobs> {
        vec![
            position(&[(0, -0.0, 1), (0, -0.0, 1), (9, -1e-7, 2)]),
            position(&[
                (151_935, f32::NEG_INFINITY, 77),
                (3, f32::NAN, 1),
                (i32::MAX as u32, -1e30, 2),
            ]),
            position(&[
                (42, f32::INFINITY, 2),
                (7, -12.345_678, 1),
                (42, f32::INFINITY, 2),
            ]),
            position(&[
                (1, f32::MIN_POSITIVE / 8.0, 1),
                (100, -9999.0, 1),
                (1000, -10000.5, 2),
            ]),
        ]
    }

    fn collected_output(
        logprobs: Option<Vec<PositionLogprobs>>,
        kv: Option<serde_json::Value>,
    ) -> CollectedGenerateOutput {
        let token_ids = logprobs
            .as_ref()
            .map(|positions| positions.iter().map(|p| p.entries[0].token_id).collect())
            .unwrap_or_else(|| vec![5, 6]);
        CollectedGenerateOutput {
            request_id: "raw-1".to_string(),
            prompt_logprobs: Some(Logprobs {
                positions: vec![position(&[(22, -0.5, 1)])],
            }),
            token_ids,
            logprobs: logprobs.map(|positions| Logprobs { positions }),
            finish_reason: FinishReason::Abort,
            usage: vllm_llm::TokenUsage::default(),
            kv_transfer_params: kv,
            ec_transfer_params: None,
            prompt_token_ids: vec![11, 22],
        }
    }

    /// Reference bytes: the pre-optimization eager conversion + serde_json.
    fn reference_bytes(collected: &CollectedGenerateOutput, request_id: &str) -> Vec<u8> {
        let response = GenerateResponse {
            request_id: request_id.to_string(),
            choices: vec![GenerateResponseChoice {
                index: 0,
                logprobs: collected
                    .logprobs
                    .as_ref()
                    .map(|logprobs| raw_logprobs_to_openai_chat(logprobs).expect("convert")),
                finish_reason: Some(collected.finish_reason.as_str().to_string()),
                token_ids: collected.token_ids.clone(),
            }],
            prompt_logprobs: collected.prompt_logprobs.as_ref().map(raw_prompt_logprobs_to_maps),
            kv_transfer_params: collected.kv_transfer_params.clone(),
            ec_transfer_params: collected.ec_transfer_params.clone(),
        };
        serde_json::to_vec(&response).expect("serialize reference")
    }

    async fn render_openai(mut collected: CollectedGenerateOutput, request_id: &str) -> Vec<u8> {
        let include_logprobs = collected.logprobs.is_some();
        let logprobs = openai_choice_logprobs(collected.logprobs.take(), include_logprobs)
            .unwrap_or_else(|_| panic!("valid logprobs"));
        let envelope = collect_generate(
            collected,
            RoutedExpertsEncoder::default(),
            request_id.to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_logprobs,
                include_prompt_logprobs: true,
                ..Default::default()
            },
        )
        .expect("envelope");
        let response = generate_response(envelope, logprobs);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "application/json"
        );
        to_bytes(response.into_body(), usize::MAX).await.expect("body").to_vec()
    }

    #[tokio::test]
    async fn direct_openai_render_is_byte_identical_to_serde_reference() {
        let many: Vec<PositionLogprobs> = (0..1000_u32)
            .map(|i| PositionLogprobs {
                entries: (0..9_u32)
                    .map(|j| TokenLogprob {
                        token_id: i * 131 + j * 7919,
                        logprob: -(i as f32) * 0.037 - j as f32 * 1.25e-3,
                        rank: j.max(1),
                    })
                    .collect(),
            })
            .collect();
        let mut with_max_id = tricky_positions();
        with_max_id.push(position(&[(u32::MAX, -2.5, 1), (u32::MAX, -2.5, 1)]));
        let cases = [
            (collected_output(Some(tricky_positions()), None), "raw-1"),
            (collected_output(Some(with_max_id), None), "max-id"),
            (
                collected_output(
                    Some(tricky_positions()),
                    Some(json!({"a": [1, "x"], "b": null})),
                ),
                "quote\"back\\slash\u{1}\u{e9}",
            ),
            (collected_output(Some(many), None), "many"),
            (collected_output(Some(Vec::new()), None), "empty-positions"),
            (collected_output(None, None), "no-logprobs"),
        ];
        for (collected, request_id) in cases {
            let expected = reference_bytes(&collected, request_id);
            let actual = render_openai(collected, request_id).await;
            assert_eq!(
                String::from_utf8(actual).unwrap(),
                String::from_utf8(expected).unwrap(),
                "request_id={request_id}"
            );
        }
    }

    #[test]
    fn openai_choice_logprobs_rejects_empty_position() {
        let logprobs = Logprobs {
            positions: vec![
                position(&[(1, -0.1, 1)]),
                PositionLogprobs { entries: vec![] },
            ],
        };
        assert!(openai_choice_logprobs(Some(logprobs), true).is_err());
        assert!(openai_choice_logprobs(None, true).is_err());
        assert!(matches!(
            openai_choice_logprobs(None, false),
            Ok(ChoiceLogprobs::None)
        ));
    }

    fn step(
        token_ids: Vec<u32>,
        positions: Option<Vec<PositionLogprobs>>,
        finish_reason: Option<FinishReason>,
    ) -> vllm_llm::Result<GenerateOutput> {
        Ok(GenerateOutput {
            request_id: "engine-1".to_string(),
            prompt_info: None,
            token_ids,
            logprobs: positions.map(|positions| Logprobs { positions }),
            finish_reason,
            cached_token_count: 0,
            kv_transfer_params: None,
            ec_transfer_params: None,
            routed_experts: None,
        })
    }

    async fn compact_response_json(
        steps: Vec<vllm_llm::Result<GenerateOutput>>,
        include_logprobs: bool,
    ) -> Result<serde_json::Value, ApiError> {
        let (collected, accumulator) = stream::iter(steps)
            .collect_output_into(CompactLogprobsAccumulator::new(3))
            .await
            .expect("collect");
        assert!(collected.logprobs.is_none());
        let logprobs =
            compact_choice_logprobs(accumulator, include_logprobs, collected.token_ids.len())?;
        let envelope = collect_generate(
            collected,
            RoutedExpertsEncoder::default(),
            "compact-1".to_string(),
            ApiServerOptions::default(),
            ResponseOptions::default(),
        )?;
        let response = generate_response(envelope, logprobs);
        let length = response
            .headers()
            .get("content-length")
            .map(|value| value.to_str().unwrap().parse::<usize>().unwrap());
        let body = to_bytes(response.into_body(), usize::MAX).await.expect("body");
        if let Some(length) = length {
            assert_eq!(length, body.len());
        }
        Ok(serde_json::from_slice(&body).expect("valid json"))
    }

    #[tokio::test]
    async fn compact_abort_with_partial_output_decodes_engine_rows() {
        let rows = tricky_positions()
            .into_iter()
            .map(|mut p| {
                p.entries.truncate(3);
                p
            })
            .collect::<Vec<_>>();
        let json = compact_response_json(
            vec![
                step(vec![0, 151_935], Some(rows[..2].to_vec()), None),
                step(vec![42], Some(rows[2..3].to_vec()), None),
                // Abort: terminal output with no new tokens or logprobs.
                step(vec![], None, Some(FinishReason::Abort)),
            ],
            true,
        )
        .await
        .expect("compact response");

        let choice = &json["choices"][0];
        assert_eq!(json["request_id"], "compact-1");
        assert_eq!(choice["finish_reason"], "abort");
        assert_eq!(choice["token_ids"], json!([0, 151_935, 42]));
        assert!(choice["logprobs"].is_null());
        assert!(json["prompt_logprobs"].is_null());
        let block = &choice["compact_logprobs"];
        assert_eq!(block["num_positions"], 3);
        assert_eq!(block["num_slots"], 3);
        let (token_ids, bits, ranks) = decode_compact(block);
        let expected_rows = &rows[..3];
        assert_eq!(
            token_ids,
            expected_rows
                .iter()
                .flat_map(|p| p.entries.iter().map(|e| e.token_id as i32))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            bits,
            expected_rows
                .iter()
                .flat_map(|p| p.entries.iter().map(|e| e.logprob.to_bits()))
                .collect::<Vec<_>>()
        );
        assert_eq!(ranks, vec![1, 77, 2]);
        // Sampled slot of each row equals the sampled token id.
        assert_eq!(token_ids[3], 151_935);
        // Raw, unclamped non-finite logprobs survive.
        assert_eq!(f32::from_bits(bits[3]), f32::NEG_INFINITY);
        assert!(f32::from_bits(bits[4]).is_nan());
        assert_eq!(f32::from_bits(bits[6]), f32::INFINITY);
    }

    #[tokio::test]
    async fn compact_without_requested_logprobs_omits_block() {
        let json = compact_response_json(
            vec![step(vec![1, 2], None, Some(FinishReason::Length))],
            false,
        )
        .await
        .expect("compact response");
        assert!(json["choices"][0]["logprobs"].is_null());
        // Same as the Python frontend: the key is omitted, not null.
        assert_eq!(
            json["choices"][0].as_object().unwrap().keys().collect::<Vec<_>>(),
            ["index", "logprobs", "finish_reason", "token_ids"]
        );
    }

    #[tokio::test]
    async fn compact_abort_before_any_position_uses_requested_width() {
        // The engine attached an empty logprobs payload to the abort output.
        let json = compact_response_json(
            vec![step(vec![], Some(Vec::new()), Some(FinishReason::Abort))],
            true,
        )
        .await
        .expect("compact response");
        let block = &json["choices"][0]["compact_logprobs"];
        assert_eq!(block["num_positions"], 0);
        assert_eq!(block["num_slots"], 3);
        assert_eq!(block["token_ids"], "");
    }

    #[tokio::test]
    async fn compact_zero_token_abort_without_payload_returns_empty_block() {
        // The real shape: the client-synthesized abort output (and the engine's
        // abort of a waiting request) carries no tokens and `logprobs: None`.
        let json = compact_response_json(vec![step(vec![], None, Some(FinishReason::Abort))], true)
            .await
            .expect("zero-token abort is a 200 with an empty block");
        let choice = &json["choices"][0];
        assert_eq!(choice["finish_reason"], "abort");
        assert_eq!(choice["token_ids"], json!([]));
        assert!(choice["logprobs"].is_null());
        let block = &choice["compact_logprobs"];
        assert_eq!(block["num_positions"], 0);
        assert_eq!(block["num_slots"], 3);
        assert_eq!(block["token_ids"], "");
        assert_eq!(block["logprobs"], "");
        assert_eq!(block["ranks"], "");
    }

    #[tokio::test]
    async fn compact_rejects_positions_misaligned_with_tokens() {
        let row = |id: u32| position(&[(id, -0.5, 1), (id, -0.5, 1), (id + 1, -0.6, 2)]);
        // Step 1 has a token but no payload; step 2 has one row.
        let missing_intermediate = compact_response_json(
            vec![
                step(vec![10], None, None),
                step(vec![20], Some(vec![row(20)]), Some(FinishReason::Abort)),
            ],
            true,
        )
        .await;
        assert!(missing_intermediate.is_err(), "{missing_intermediate:?}");
        // Token-bearing terminal output with an empty payload.
        let empty_payload = compact_response_json(
            vec![step(vec![30], Some(Vec::new()), Some(FinishReason::Length))],
            true,
        )
        .await;
        assert!(empty_payload.is_err(), "{empty_payload:?}");
        // Counts that cancel out across steps (0 rows then 2 rows).
        let shifted = compact_response_json(
            vec![
                step(vec![1], Some(Vec::new()), None),
                step(
                    vec![2],
                    Some(vec![row(1), row(2)]),
                    Some(FinishReason::Abort),
                ),
            ],
            true,
        )
        .await;
        assert!(shifted.is_err(), "{shifted:?}");
        // Aligned steps still succeed.
        compact_response_json(
            vec![
                step(vec![1], Some(vec![row(1)]), None),
                step(vec![2], Some(vec![row(2)]), Some(FinishReason::Abort)),
            ],
            true,
        )
        .await
        .expect("aligned steps");
    }

    #[tokio::test]
    async fn stream_compact_rejects_chunk_positions_misaligned_with_tokens() {
        let result: Result<Vec<_>, _> = generate_chunk_stream(
            stream::iter(vec![step(
                vec![1, 2],
                Some(tricky_positions()[..1].to_vec()),
                None,
            )]),
            "raw-stream".to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_compact_logprobs: true,
                logprobs_format: LogprobsFormat::Compact,
                logprobs_slots: 3,
                ..Default::default()
            },
        )
        .try_collect()
        .await;
        assert!(result.is_err());
    }

    async fn compact_stream(
        steps: Vec<vllm_llm::Result<GenerateOutput>>,
    ) -> Result<Vec<GenerateStreamResponse>, ApiError> {
        generate_chunk_stream(
            stream::iter(steps),
            "raw-stream".to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_compact_logprobs: true,
                logprobs_format: LogprobsFormat::Compact,
                logprobs_slots: 3,
                ..Default::default()
            },
        )
        .try_collect()
        .await
    }

    #[tokio::test]
    async fn stream_compact_rejects_rows_on_zero_token_outputs() {
        let row = || tricky_positions()[..1].to_vec();
        // Terminal zero-token output carrying a position.
        let terminal = compact_stream(vec![
            step(vec![0], Some(row()), None),
            step(vec![], Some(row()), Some(FinishReason::Abort)),
        ])
        .await;
        assert!(terminal.is_err(), "{terminal:?}");
        // Nonterminal zero-token output carrying rows.
        let nonterminal = compact_stream(vec![
            step(vec![], Some(row()), None),
            step(vec![0], Some(row()), Some(FinishReason::Length)),
        ])
        .await;
        assert!(nonterminal.is_err(), "{nonterminal:?}");
        // Zero tokens with no or empty payloads stay valid.
        let chunks = compact_stream(vec![
            step(vec![0], Some(row()), None),
            step(vec![], None, None),
            step(vec![], Some(Vec::new()), Some(FinishReason::Abort)),
        ])
        .await
        .expect("zero tokens without positions");
        assert_eq!(chunks.len(), 2);
    }

    #[tokio::test]
    async fn compact_missing_logprobs_payload_is_server_error() {
        let error =
            compact_response_json(vec![step(vec![1], None, Some(FinishReason::Length))], true)
                .await
                .expect_err("missing payload");
        assert!(format!("{error:?}").contains("positions"), "{error:?}");
    }

    #[tokio::test]
    async fn stream_compact_emits_per_chunk_blocks() {
        let rows = tricky_positions();
        let chunks: Vec<_> = generate_chunk_stream(
            stream::iter(vec![
                step(vec![0], Some(rows[..1].to_vec()), None),
                step(
                    vec![151_935],
                    Some(rows[1..2].to_vec()),
                    Some(FinishReason::Abort),
                ),
            ]),
            "raw-stream".to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_compact_logprobs: true,
                logprobs_format: LogprobsFormat::Compact,
                logprobs_slots: 3,
                ..Default::default()
            },
        )
        .try_collect()
        .await
        .expect("collect chunks");
        assert_eq!(chunks.len(), 2);
        for (chunk, row) in chunks.iter().zip(&rows) {
            let json = serde_json::to_value(chunk).unwrap();
            let choice = &json["choices"][0];
            // Compact chunks carry an explicit `"logprobs": null`.
            assert!(choice.get("logprobs").is_some_and(serde_json::Value::is_null));
            let keys: Vec<_> = choice.as_object().unwrap().keys().map(String::as_str).collect();
            assert!(
                keys == ["index", "logprobs", "token_ids", "compact_logprobs"]
                    || keys
                        == [
                            "index",
                            "logprobs",
                            "finish_reason",
                            "token_ids",
                            "compact_logprobs"
                        ],
                "{keys:?}"
            );
            let block = &choice["compact_logprobs"];
            assert_eq!(block["num_positions"], 1);
            let (token_ids, bits, _) = decode_compact(block);
            assert_eq!(
                token_ids,
                row.entries.iter().map(|e| e.token_id as i32).collect::<Vec<_>>()
            );
            assert_eq!(
                bits,
                row.entries.iter().map(|e| e.logprob.to_bits()).collect::<Vec<_>>()
            );
        }

        // The default stream format carries no compact block.
        let chunks: Vec<_> = generate_chunk_stream(
            stream::iter(vec![step(
                vec![0],
                Some(rows[..1].to_vec()),
                Some(FinishReason::Length),
            )]),
            "raw-stream".to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_logprobs: true,
                ..Default::default()
            },
        )
        .try_collect()
        .await
        .expect("collect chunks");
        let json = serde_json::to_value(&chunks[0]).unwrap();
        assert!(json["choices"][0].get("compact_logprobs").is_none());
        assert!(json["choices"][0]["logprobs"]["content"].is_array());
    }

    #[test]
    fn collect_generate_maps_prompt_logprobs_for_single_token_prompt() {
        let output_without_payload = |prompt_token_ids: Vec<u32>| CollectedGenerateOutput {
            request_id: "raw-1".to_string(),
            prompt_logprobs: None,
            token_ids: vec![3],
            logprobs: None,
            finish_reason: FinishReason::stop_eos(),
            usage: vllm_llm::TokenUsage {
                prompt_token_count: prompt_token_ids.len(),
                output_token_count: 1,
                cached_token_count: 0,
            },
            kv_transfer_params: None,
            ec_transfer_params: None,
            prompt_token_ids,
        };

        let response = collect_generate(
            output_without_payload(vec![9707]),
            RoutedExpertsEncoder::default(),
            "raw-1".to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_prompt_logprobs: true,
                ..Default::default()
            },
        )
        .expect("single-token prompt without payload maps to [None]");
        let prompt_logprobs = response.prompt_logprobs.expect("prompt logprobs present");
        assert_eq!(prompt_logprobs.len(), 1);
        assert!(prompt_logprobs[0].is_none());

        collect_generate(
            output_without_payload(vec![9707, 11]),
            RoutedExpertsEncoder::default(),
            "raw-2".to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_prompt_logprobs: true,
                ..Default::default()
            },
        )
        .expect_err("multi-token prompt without payload is an engine failure");
    }

    // ---- SPEC v3: compact field switches and routed experts (R3) ----

    use super::routed_experts::npy_header;
    use super::routed_experts::tests::reference_bytes as routed_reference_bytes;

    fn options_for(format: LogprobsFormat, slots: usize) -> ResponseOptions {
        ResponseOptions {
            include_logprobs: true,
            include_compact_logprobs: true,
            logprobs_format: format,
            logprobs_slots: slots,
            ..Default::default()
        }
    }

    /// Render a non-streaming response through the same path as the handler.
    async fn response_json(
        steps: Vec<vllm_llm::Result<GenerateOutput>>,
        options: ResponseOptions,
        enable_return_routed_experts: bool,
    ) -> Result<serde_json::Value, ApiError> {
        let (envelope, logprobs) = collect_response(
            stream::iter(steps),
            "r3-1".to_string(),
            ApiServerOptions {
                enable_return_routed_experts,
                ..Default::default()
            },
            options,
        )
        .await
        .expect("stream")?;
        let body = to_bytes(
            generate_response(envelope, logprobs).into_body(),
            usize::MAX,
        )
        .await
        .expect("body");
        Ok(serde_json::from_slice(&body).expect("valid json"))
    }

    fn routed(rows: usize, layers: usize, topk: usize, offset: usize) -> RoutedExperts {
        let all = routed_reference_bytes("|u1", (offset + rows) * layers * topk);
        RoutedExperts {
            dtype: "|u1".to_string(),
            shape: vec![rows, layers, topk],
            data: bytes::Bytes::copy_from_slice(&all[offset * layers * topk..]),
        }
    }

    fn with_routed(
        output: vllm_llm::Result<GenerateOutput>,
        chunk: RoutedExperts,
    ) -> vllm_llm::Result<GenerateOutput> {
        output.map(|mut output| {
            output.routed_experts = Some(chunk);
            output
        })
    }

    /// Steps of a request with a 5-row prompt block, then 1 routing row per
    /// token for the first two tokens (the last token has no row).
    fn routed_steps(layers: usize) -> Vec<vllm_llm::Result<GenerateOutput>> {
        let row = |id: u32| position(&[(id, -0.5, 1), (id, -0.5, 1), (id + 1, -0.75, 2)]);
        vec![
            with_routed(
                step(vec![10], Some(vec![row(10)]), None),
                routed(5, layers, 8, 0),
            ),
            with_routed(
                step(vec![11], Some(vec![row(11)]), None),
                routed(1, layers, 8, 5),
            ),
            with_routed(
                step(vec![12], Some(vec![row(12)]), Some(FinishReason::Abort)),
                routed(1, layers, 8, 6),
            ),
        ]
    }

    fn expected_npy_base64(rows: usize, layers: usize) -> String {
        use base64::Engine as _;
        let mut npy = npy_header("|u1", &[rows, layers, 8]).unwrap();
        npy.extend_from_slice(&routed_reference_bytes("|u1", rows * layers * 8));
        base64::engine::general_purpose::STANDARD.encode(npy)
    }

    #[tokio::test]
    async fn routed_experts_concatenate_all_chunks_for_both_formats() {
        for layers in [4, 61] {
            for format in [LogprobsFormat::OpenAi, LogprobsFormat::Compact] {
                let json = response_json(routed_steps(layers), options_for(format, 3), false)
                    .await
                    .expect("response");
                let choice = &json["choices"][0];
                assert_eq!(choice["token_ids"], json!([10, 11, 12]));
                assert_eq!(
                    choice["routed_experts"].as_str().unwrap(),
                    expected_npy_base64(7, layers),
                    "{format:?} layers {layers}"
                );
                let keys: Vec<_> = choice.as_object().unwrap().keys().cloned().collect();
                let token_ids = keys.iter().position(|k| k == "token_ids").unwrap();
                assert_eq!(keys[token_ids + 1], "routed_experts");
            }
        }
    }

    #[tokio::test]
    async fn routed_experts_null_when_enabled_without_forward_and_omitted_when_disabled() {
        let abort = || vec![step(vec![], None, Some(FinishReason::Abort))];
        for format in [LogprobsFormat::OpenAi, LogprobsFormat::Compact] {
            let mut options = options_for(format, 3);
            options.include_logprobs = false;
            options.include_compact_logprobs = false;
            // Enabled, aborted before any forward pass: null.
            let json = response_json(abort(), options, true).await.expect("response");
            assert!(json["choices"][0]["routed_experts"].is_null());
            assert!(json["choices"][0].get("routed_experts").is_some());
            // Disabled and no rows: key omitted (default bytes unchanged).
            let json = response_json(abort(), options, false).await.expect("response");
            assert!(json["choices"][0].get("routed_experts").is_none());
        }
    }

    #[tokio::test]
    async fn routed_experts_layout_change_is_server_error() {
        let mut steps = routed_steps(4);
        steps[1] = with_routed(step(vec![11], None, None), routed(1, 5, 8, 0));
        let mut options = options_for(LogprobsFormat::OpenAi, 3);
        options.include_logprobs = false;
        assert!(response_json(steps, options, true).await.is_err());
    }

    /// A chunk outside the R3 contract (huge rank, accepted by the wire
    /// decoder) is a server error for that request, not a panic.
    #[tokio::test]
    async fn routed_experts_huge_rank_is_server_error() {
        let huge = RoutedExperts {
            dtype: "|u1".to_string(),
            shape: vec![1; 22_000],
            data: bytes::Bytes::from_static(&[3]),
        };
        for (first, enabled) in [(true, true), (false, true), (true, false), (false, false)] {
            let mut steps = routed_steps(4);
            steps[usize::from(!first)] = with_routed(step(vec![10], None, None), huge.clone());
            let mut options = options_for(LogprobsFormat::OpenAi, 3);
            options.include_logprobs = false;
            let error = response_json(steps, options, enabled).await.unwrap_err();
            assert_eq!(
                error.into_response().status(),
                axum::http::StatusCode::INTERNAL_SERVER_ERROR
            );
        }
    }

    #[tokio::test]
    async fn stream_returns_routed_experts_on_the_finishing_chunk() {
        let chunks: Vec<_> = generate_chunk_stream(
            stream::iter(routed_steps(4)),
            "raw-stream".to_string(),
            ApiServerOptions::default(),
            options_for(LogprobsFormat::Compact, 3),
        )
        .try_collect()
        .await
        .expect("collect chunks");
        assert_eq!(chunks.len(), 3);
        for chunk in &chunks[..2] {
            let json = serde_json::to_value(chunk).unwrap();
            assert!(json["choices"][0].get("routed_experts").is_none());
        }
        let last = serde_json::to_value(&chunks[2]).unwrap();
        assert_eq!(
            last["choices"][0]["routed_experts"].as_str().unwrap(),
            expected_npy_base64(7, 4)
        );
    }

    /// Keys of the first `compact_logprobs` object in `body`, in order.
    fn compact_block_keys(body: &str) -> Vec<String> {
        let start = body.find("\"compact_logprobs\":{").expect("compact block")
            + "\"compact_logprobs\":{".len();
        let end = start + body[start..].find('}').expect("block end");
        body[start..end]
            .split(',')
            .map(|field| field.split(':').next().unwrap().trim_matches('"').to_string())
            .collect()
    }

    #[tokio::test]
    async fn compact_block_key_order_matches_python() {
        let rows = tricky_positions()
            .into_iter()
            .map(|mut p| {
                p.entries.truncate(3);
                p
            })
            .collect::<Vec<_>>();
        let steps = || {
            vec![step(
                vec![0, 151_935, 42],
                Some(rows[..3].to_vec()),
                Some(FinishReason::Abort),
            )]
        };
        let python_order = |sampled: bool, ranks: bool| {
            let mut keys = vec![
                "num_positions",
                "num_slots",
                "dtype_token_ids",
                "dtype_logprobs",
                "byteorder",
            ];
            if !sampled {
                keys.push("sampled_slot");
            }
            keys.extend(["token_ids", "logprobs"]);
            if ranks {
                keys.push("ranks");
            }
            keys
        };
        for (sampled, ranks) in [(true, true), (false, true), (true, false), (false, false)] {
            let mut options = options_for(LogprobsFormat::Compact, 3);
            options.compact_skip_sampled = !sampled;
            options.compact_skip_ranks = !ranks;
            let (envelope, logprobs) = collect_response(
                stream::iter(steps()),
                "order".to_string(),
                ApiServerOptions::default(),
                options,
            )
            .await
            .expect("stream")
            .expect("response");
            let body = to_bytes(
                generate_response(envelope, logprobs).into_body(),
                usize::MAX,
            )
            .await
            .expect("body");
            let body = std::str::from_utf8(&body).unwrap();
            assert_eq!(
                compact_block_keys(body),
                python_order(sampled, ranks),
                "{sampled} {ranks}"
            );
            if !sampled {
                assert!(
                    body.contains("\"byteorder\":\"little\",\"sampled_slot\":false,\"token_ids\":")
                );
            }

            let chunks: Vec<_> = generate_chunk_stream(
                stream::iter(steps()),
                "order".to_string(),
                ApiServerOptions::default(),
                options,
            )
            .try_collect()
            .await
            .expect("chunks");
            let chunk = serde_json::to_string(&chunks[0]).unwrap();
            assert_eq!(
                compact_block_keys(&chunk),
                python_order(sampled, ranks),
                "stream {sampled} {ranks}"
            );
        }
    }

    #[tokio::test]
    async fn compact_switches_drop_sampled_slot_and_ranks() {
        let rows = tricky_positions()
            .into_iter()
            .map(|mut p| {
                p.entries.truncate(3);
                p
            })
            .collect::<Vec<_>>();
        let steps = || {
            vec![
                step(vec![0, 151_935], Some(rows[..2].to_vec()), None),
                step(
                    vec![42],
                    Some(rows[2..3].to_vec()),
                    Some(FinishReason::Abort),
                ),
            ]
        };
        // Default switches: today's block.
        let json = response_json(steps(), options_for(LogprobsFormat::Compact, 3), false)
            .await
            .unwrap();
        let block = &json["choices"][0]["compact_logprobs"];
        assert!(block.get("sampled_slot").is_none());
        assert!(block.get("ranks").is_some());
        assert_eq!(block["num_slots"], 3);

        let mut options = options_for(LogprobsFormat::Compact, 3);
        options.compact_skip_sampled = true;
        options.compact_skip_ranks = true;
        let json = response_json(steps(), options, false).await.unwrap();
        let block = &json["choices"][0]["compact_logprobs"];
        assert_eq!(block["num_positions"], 3);
        assert_eq!(block["num_slots"], 2);
        assert_eq!(block["sampled_slot"], false);
        assert!(block.get("ranks").is_none());
        let decode = |key: &str| {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD
                .decode(block[key].as_str().unwrap())
                .unwrap()
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        // Engine slots 1..3 of each row, raw bits.
        let expected_ids: Vec<u32> = rows[..3]
            .iter()
            .flat_map(|p| p.entries[1..].iter().map(|e| e.token_id))
            .collect();
        let expected_bits: Vec<u32> = rows[..3]
            .iter()
            .flat_map(|p| p.entries[1..].iter().map(|e| e.logprob.to_bits()))
            .collect();
        assert_eq!(decode("token_ids"), expected_ids);
        assert_eq!(decode("logprobs"), expected_bits);

        // Ranks only off: sampled slot kept, ranks omitted.
        let mut options = options_for(LogprobsFormat::Compact, 3);
        options.compact_skip_ranks = true;
        let json = response_json(steps(), options, false).await.unwrap();
        let block = &json["choices"][0]["compact_logprobs"];
        assert!(block.get("sampled_slot").is_none());
        assert!(block.get("ranks").is_none());
        assert_eq!(block["num_slots"], 3);

        // Streaming chunks apply the same switches.
        let mut options = options_for(LogprobsFormat::Compact, 3);
        options.compact_skip_sampled = true;
        let chunks: Vec<_> = generate_chunk_stream(
            stream::iter(steps()),
            "s".to_string(),
            ApiServerOptions::default(),
            options,
        )
        .try_collect()
        .await
        .unwrap();
        let json = serde_json::to_value(&chunks[0]).unwrap();
        let block = &json["choices"][0]["compact_logprobs"];
        assert_eq!(block["num_slots"], 2);
        assert_eq!(block["sampled_slot"], false);
        assert!(block.get("ranks").is_some());
    }
}
