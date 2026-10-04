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
use futures::{Stream, StreamExt as _, pin_mut};
use thiserror_ext::AsReport as _;
use tracing::{error, info, trace};
use tracing_futures::Instrument as _;
use vllm_engine_core_client::protocol::logprobs::{Logprobs, PositionLogprobs};
use vllm_llm::{
    CollectedGenerateOutput, FinishReason, GenerateOutput, GenerateOutputStreamExt as _, TokenUsage,
};

use self::compact::{CompactLogprobsAccumulator, LogprobsFormat, encode_compact};
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

    let options = prepared.options;
    let collect_error = |error: vllm_llm::Error| {
        server_error!(
            "failed to collect raw generate response: {}",
            error.to_report_string()
        )
        .into_response()
    };

    let result = match options.logprobs_format {
        LogprobsFormat::OpenAi => {
            let mut collected =
                match raw_stream.collect_output().instrument(request_span.clone()).await {
                    Ok(collected) => collected,
                    Err(error) => return collect_error(error),
                };
            let logprobs = collected.logprobs.take();
            openai_choice_logprobs(logprobs, options.include_logprobs).and_then(|logprobs| {
                let envelope =
                    collect_generate(collected, prepared.request_id, api_server_options, options)?;
                Ok((envelope, logprobs))
            })
        }
        LogprobsFormat::Compact => {
            let accumulator = CompactLogprobsAccumulator::new(options.logprobs_slots);
            let (collected, accumulator) = match raw_stream
                .collect_output_into(accumulator)
                .instrument(request_span.clone())
                .await
            {
                Ok(collected) => collected,
                Err(error) => return collect_error(error),
            };
            compact_choice_logprobs(
                accumulator,
                options.include_compact_logprobs,
                collected.token_ids.len(),
            )
            .and_then(|logprobs| {
                let envelope =
                    collect_generate(collected, prepared.request_id, api_server_options, options)?;
                Ok((envelope, logprobs))
            })
        }
    };

    match result {
        // In the request span so the body render task inherits it.
        Ok((envelope, logprobs)) => request_span.in_scope(|| generate_response(envelope, logprobs)),
        Err(error) => error.into_response(),
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
    // A request that produced no tokens (e.g. aborted while waiting: the
    // engine / client abort output has no tokens and no logprobs payload)
    // gets an empty block with the requested width. Missing logprobs for
    // produced tokens are still an engine failure.
    if !accumulator.saw_payload() && output_tokens > 0 {
        return Err(ApiError::server_error(
            "raw generate response requested logprobs but generation returned none".to_string(),
        ));
    }
    accumulator
        .finish()
        .map(|block| ChoiceLogprobs::Compact(Some(block)))
        .map_err(ApiError::server_error)
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
        logprobs_format,
        include_compact_logprobs,
        logprobs_slots,
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
                            (Some(raw_logprobs_to_openai_chat(&logprobs)?), None)
                        }
                        // Per-chunk compact block covering this chunk's positions.
                        LogprobsFormat::Compact => (
                            None,
                            Some(
                                encode_compact(logprobs, logprobs_slots)
                                    .map_err(ApiError::server_error)?,
                            ),
                        ),
                    }
                } else {
                    (None, None)
                };

                y.yield_ok(GenerateStreamResponse {
                    request_id: request_id.clone(),
                    choices: vec![GenerateResponseStreamChoice {
                        index: 0,
                        logprobs,
                        finish_reason: finish_reason.map(|reason| reason.as_str().to_string()),
                        token_ids,
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
    async fn compact_missing_logprobs_payload_is_server_error() {
        let error =
            compact_response_json(vec![step(vec![1], None, Some(FinishReason::Length))], true)
                .await
                .expect_err("missing payload");
        assert!(format!("{error:?}").contains("returned none"), "{error:?}");
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
            assert!(choice.get("logprobs").is_none());
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
}
