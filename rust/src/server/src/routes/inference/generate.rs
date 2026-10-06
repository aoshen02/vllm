// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

mod compact;
mod convert;
mod render;
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
use futures::{Stream, StreamExt as _, TryStreamExt as _, pin_mut};
use thiserror_ext::AsReport as _;
use tracing::{error, info, trace};
use tracing_futures::Instrument as _;
use vllm_engine_core_client::protocol::logprobs::{Logprobs, PositionLogprobs};
use vllm_llm::{
    CollectedGenerateOutput, FinishReason, GenerateOutput, GenerateOutputStreamExt as _, TokenUsage,
};

use self::compact::CompactLogprobsAccumulator;
use self::convert::{ResponseOptions, prepare_generate_request};
use self::render::{ChoiceLogprobs, GenerateEnvelope, generate_response};
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

    let result = match collect_response(
        raw_stream,
        prepared.request_id,
        api_server_options,
        prepared.options,
    )
    .instrument(request_span.clone())
    .await
    {
        Ok(result) => result,
        Err(error) => {
            return server_error!(
                "failed to collect raw generate response: {}",
                error.to_report_string()
            )
            .into_response();
        }
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
    raw_stream: impl Stream<Item = vllm_llm::Result<GenerateOutput>> + Send,
    request_id: String,
    api_server_options: ApiServerOptions,
    options: ResponseOptions,
) -> vllm_llm::Result<Result<(GenerateEnvelope, ChoiceLogprobs), ApiError>> {
    let mut compact = options.compact_row_width.map(CompactLogprobsAccumulator::new);
    let mut collected = match compact.as_mut() {
        None => raw_stream.collect_output().await?,
        // Compact: encode each step's logprobs as it arrives.
        Some(accumulator) => {
            raw_stream
                .map_ok(|mut output| {
                    accumulator.push(output.token_ids.len(), output.logprobs.take());
                    output
                })
                .collect_output()
                .await?
        }
    };
    let logprobs = match compact {
        None => openai_choice_logprobs(collected.logprobs.take(), options.include_logprobs),
        Some(accumulator) => accumulator
            .finish()
            .map(ChoiceLogprobs::Compact)
            .map_err(ApiError::server_error),
    };
    Ok(logprobs.and_then(|logprobs| {
        let envelope = collect_generate(collected, request_id, api_server_options, options)?;
        Ok((envelope, logprobs))
    }))
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
    // The body is rendered after the status line, so reject empty rows here.
    if logprobs.positions.iter().any(|position| position.entries.is_empty()) {
        return Err(empty_position_error());
    }
    Ok(ChoiceLogprobs::OpenAi(logprobs.positions))
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
        ..
    }: ApiServerOptions,
    ResponseOptions {
        include_usage,
        include_continuous_usage,
        include_logprobs,
        // Ignored: raw generate streaming has no prompt-logprobs wire shape.
        include_prompt_logprobs: _,
        // Compact is rejected for streaming at validation.
        compact_row_width: _,
    }: ResponseOptions,
    mut y: TryYielder<GenerateStreamResponse, ApiError>,
) -> Result<(), ApiError> {
    pin_mut!(stream);
    let mut prompt_tokens = None;
    let mut usage = TokenUsage::default();

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

                if token_ids.is_empty() && finish_reason.is_none() {
                    continue;
                }

                let logprobs = if include_logprobs && !token_ids.is_empty() {
                    let logprobs = output.logprobs.as_ref().ok_or_else(|| {
                        server_error!(
                            "raw generate stream requested logprobs but generation returned none"
                        )
                    })?;
                    Some(raw_logprobs_to_openai_chat(logprobs)?)
                } else {
                    None
                };

                y.yield_ok(GenerateStreamResponse {
                    request_id: request_id.clone(),
                    choices: vec![GenerateResponseStreamChoice {
                        index: 0,
                        logprobs,
                        finish_reason: finish_reason.map(|reason| reason.as_str().to_string()),
                        token_ids,
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
    request_id: String,
    ApiServerOptions {
        enable_log_requests,
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

    use axum::body::to_bytes;
    use futures::{TryStreamExt as _, stream};
    use serde_json::json;
    use vllm_engine_core_client::protocol::logprobs::TokenLogprob;
    use vllm_llm::GeneratePromptInfo;

    use super::compact::tests::{decode_compact, top_k};
    use super::types::{GenerateResponse, GenerateResponseChoice};
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
            "raw-2".to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_prompt_logprobs: true,
                ..Default::default()
            },
        )
        .expect_err("multi-token prompt without payload is an engine failure");
    }

    pub(super) fn position(entries: &[(u32, f32, u32)]) -> PositionLogprobs {
        PositionLogprobs {
            entries: entries
                .iter()
                .map(|&(token_id, logprob, rank)| TokenLogprob {
                    token_id,
                    logprob,
                    rank,
                })
                .collect(),
        }
    }

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
            position(&[(8, -0.5, 1)]),
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

    /// Reference bytes: the eager `ChatLogProbs` conversion + serde_json.
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
        })
    }

    async fn openai_body_bytes(
        steps: Vec<vllm_llm::Result<GenerateOutput>>,
    ) -> Result<Vec<u8>, ApiError> {
        let (envelope, logprobs) = collect_response(
            stream::iter(steps),
            "probe".to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_logprobs: true,
                ..Default::default()
            },
        )
        .await
        .expect("stream")?;
        Ok(to_bytes(
            generate_response(envelope, logprobs).into_body(),
            usize::MAX,
        )
        .await
        .expect("body")
        .to_vec())
    }

    /// An empty row fails the request with a 500 (non-streaming and streaming).
    #[tokio::test]
    async fn openai_empty_row_fails_the_request() {
        let empty_row = || {
            vec![
                step(vec![1], Some(vec![position(&[(1, -0.1, 1)])]), None),
                step(
                    vec![2],
                    Some(vec![PositionLogprobs { entries: vec![] }]),
                    Some(FinishReason::Length),
                ),
            ]
        };
        let error = openai_body_bytes(empty_row()).await.expect_err("empty row");
        assert_eq!(
            error.into_response().status(),
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        );
        let streamed: Result<Vec<_>, _> = generate_chunk_stream(
            stream::iter(empty_row()),
            "probe".to_string(),
            ApiServerOptions::default(),
            ResponseOptions {
                include_logprobs: true,
                ..Default::default()
            },
        )
        .try_collect()
        .await;
        assert!(streamed.is_err());
    }

    // ---- logprobs_format: "compact" ----

    fn compact_options(width: usize) -> ResponseOptions {
        ResponseOptions {
            include_logprobs: true,
            compact_row_width: Some(width),
            ..Default::default()
        }
    }

    /// Render a non-streaming response through the handler's path.
    async fn response_body(
        steps: Vec<vllm_llm::Result<GenerateOutput>>,
        options: ResponseOptions,
    ) -> Result<String, ApiError> {
        let (envelope, logprobs) = collect_response(
            stream::iter(steps),
            "compact-1".to_string(),
            ApiServerOptions::default(),
            options,
        )
        .await
        .expect("stream")?;
        let response = generate_response(envelope, logprobs);
        // Exact size, so the response carries a Content-Length.
        let exact = http_body::Body::size_hint(response.body()).exact();
        let body = to_bytes(response.into_body(), usize::MAX).await.expect("body");
        assert_eq!(exact, Some(body.len() as u64));
        Ok(String::from_utf8(body.to_vec()).unwrap())
    }

    async fn response_json(
        steps: Vec<vllm_llm::Result<GenerateOutput>>,
        options: ResponseOptions,
    ) -> Result<serde_json::Value, ApiError> {
        let body = response_body(steps, options).await?;
        Ok(serde_json::from_str(&body).expect("valid json"))
    }

    #[tokio::test]
    async fn compact_body_is_exact() {
        let body = response_body(
            vec![step(
                vec![7],
                Some(vec![position(&[(7, -0.5, 1), (3, -1.0, 2), (9, -2.0, 3)])]),
                Some(FinishReason::Length),
            )],
            compact_options(2),
        )
        .await
        .expect("compact response");
        assert_eq!(
            body,
            concat!(
                r#"{"request_id":"compact-1","choices":[{"index":0,"logprobs":null,"#,
                r#""finish_reason":"length","token_ids":[7],"compact_logprobs":{"#,
                r#""num_positions":1,"num_slots":1,"dtype_token_ids":"int32","#,
                r#""dtype_logprobs":"float32","byteorder":"little","#,
                r#""token_ids":"AwAAAA==","logprobs":"AACAvw=="}}],"#,
                r#""prompt_logprobs":null,"kv_transfer_params":null,"ec_transfer_params":null}"#
            )
        );
    }

    #[tokio::test]
    async fn compact_abort_with_partial_output_decodes_engine_rows() {
        let rows = tricky_positions()[..3].to_vec();
        let json = response_json(
            vec![
                step(vec![0, 151_935], Some(rows[..2].to_vec()), None),
                step(vec![42], Some(rows[2..].to_vec()), None),
                // Abort: terminal output with no new tokens or logprobs.
                step(vec![], None, Some(FinishReason::Abort)),
            ],
            compact_options(3),
        )
        .await
        .expect("compact response");

        let choice = &json["choices"][0];
        assert_eq!(choice["finish_reason"], "abort");
        assert_eq!(choice["token_ids"], json!([0, 151_935, 42]));
        assert!(choice["logprobs"].is_null());
        let block = &choice["compact_logprobs"];
        assert_eq!(block["num_positions"], 3);
        assert_eq!(block["num_slots"], 2);
        let (token_ids, bits) = decode_compact(block);
        assert_eq!((token_ids, bits.clone()), top_k(&rows, 3));
        // Raw, unclamped non-finite logprobs survive.
        assert!(f32::from_bits(bits[2]).is_nan());
        assert_eq!(f32::from_bits(bits[5]), f32::INFINITY);
    }

    #[tokio::test]
    async fn compact_zero_token_abort_returns_empty_block() {
        // An abort before any token carries no tokens and `logprobs: None`;
        // the engine may also attach an empty payload.
        for payload in [None, Some(Vec::new())] {
            let json = response_json(
                vec![step(vec![], payload, Some(FinishReason::Abort))],
                compact_options(3),
            )
            .await
            .expect("zero-token abort is a 200 with an empty block");
            let block = &json["choices"][0]["compact_logprobs"];
            assert_eq!(block["num_positions"], 0);
            assert_eq!(block["num_slots"], 2);
            assert_eq!(block["token_ids"], "");
            assert_eq!(block["logprobs"], "");
        }
    }

    /// Outputs the layout cannot represent fail the request with a 500; rows
    /// wider than k + 1 (engine padding) are truncated.
    #[tokio::test]
    async fn compact_malformed_outputs_fail_the_request() {
        let row = |id: u32, width: u32| {
            position(
                &(0..width)
                    .map(|slot| (id + slot, -0.25 * slot as f32, slot.max(1)))
                    .collect::<Vec<_>>(),
            )
        };
        let cases = [
            (
                "token without payload",
                vec![
                    step(vec![1], None, None),
                    step(vec![2], Some(vec![row(2, 3)]), Some(FinishReason::Length)),
                ],
            ),
            (
                "counts shifted across steps",
                vec![
                    step(vec![1], Some(Vec::new()), None),
                    step(
                        vec![2],
                        Some(vec![row(1, 3), row(2, 3)]),
                        Some(FinishReason::Abort),
                    ),
                ],
            ),
            (
                "rows on a zero-token output",
                vec![
                    step(vec![1], Some(vec![row(1, 3)]), None),
                    step(vec![], Some(vec![row(9, 3)]), Some(FinishReason::Abort)),
                ],
            ),
            (
                "row narrower than k + 1",
                vec![step(
                    vec![1],
                    Some(vec![row(1, 2)]),
                    Some(FinishReason::Length),
                )],
            ),
            (
                "empty row",
                vec![step(
                    vec![1],
                    Some(vec![PositionLogprobs { entries: vec![] }]),
                    Some(FinishReason::Length),
                )],
            ),
        ];
        for (name, steps) in cases {
            let error = response_body(steps, compact_options(3)).await.expect_err(name);
            assert_eq!(
                error.into_response().status(),
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "{name}"
            );
        }
        let json = response_json(
            vec![
                step(vec![1], Some(vec![row(1, 3)]), None),
                step(vec![2], Some(vec![row(2, 5)]), Some(FinishReason::Length)),
            ],
            compact_options(3),
        )
        .await
        .expect("valid");
        let block = &json["choices"][0]["compact_logprobs"];
        assert_eq!(block["num_positions"], 2);
        assert_eq!(decode_compact(block).0, vec![2, 3, 3, 4]);
    }
}
