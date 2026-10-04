// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use vllm_engine_core_client::protocol::multimodal::MmFeatures;
use vllm_text::{Prompt, TextDecodeOptions, TextRequest};

use super::compact::LogprobsFormat;
use super::types::GenerateRequest;
use super::validate;
use crate::error::ApiError;
use crate::lora::LoraModelResolution;
use crate::utils::{ResolvedRequestContext, merge_ec_transfer_params, merge_kv_transfer_params};

/// Lowered generate request plus the response request ID.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct PreparedRequest {
    pub request_id: String,
    pub text_request: TextRequest,
    pub stream: bool,
    /// Public response rendering options for route-layer helpers.
    pub options: ResponseOptions,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(super) struct ResponseOptions {
    /// Whether the caller asked for the final streamed usage chunk.
    pub include_usage: bool,
    /// Whether the caller asked for usage on every streamed chunk.
    pub include_continuous_usage: bool,
    /// Whether the caller requested output logprobs on generate choices.
    pub include_logprobs: bool,
    /// Whether the caller requested top-level prompt logprobs.
    pub include_prompt_logprobs: bool,
    /// Wire format for output logprobs.
    pub logprobs_format: LogprobsFormat,
    /// Whether compact output carries a block: `logprobs` or a non-empty
    /// `logprob_token_ids` was requested (the engine returns rows for both).
    pub include_compact_logprobs: bool,
    /// Requested engine row width (`logprobs + 1`), used as `num_slots` for a
    /// compact block with no scored positions; `0` when unknown (`-1`).
    pub logprobs_slots: usize,
}

/// Validate and lower one raw generate request into the internal
/// text-generation format.
pub(super) fn prepare_generate_request(
    request: GenerateRequest,
    lora_resolution: &LoraModelResolution,
    ctx: ResolvedRequestContext,
    mm_features: Option<MmFeatures>,
) -> Result<PreparedRequest, ApiError> {
    validate::validate_request_compat(&request, &lora_resolution.model_names)?;

    let stream = request.stream;
    let include_usage = request
        .stream_options
        .as_ref()
        .and_then(|options| options.include_usage)
        .unwrap_or(false);
    let include_continuous_usage = include_usage
        && request
            .stream_options
            .as_ref()
            .and_then(|options| options.continuous_usage_stats)
            .unwrap_or(false);
    let include_logprobs = request.sampling_params.inner.logprobs.is_some();
    let include_prompt_logprobs = request.sampling_params.inner.prompt_logprobs.is_some();
    let logprobs_format = LogprobsFormat::parse(request.logprobs_format.as_ref())
        .expect("logprobs_format validated by validate_request_compat");
    // Compact follows the engine / Python `num_logprobs`: `logprobs` when
    // set, else `len(logprob_token_ids)` (lowering rejects both set with
    // different counts). The default format keeps gating on `logprobs` only.
    let logprob_token_ids = request
        .sampling_params
        .inner
        .logprob_token_ids
        .as_deref()
        .filter(|ids| !ids.is_empty());
    let include_compact_logprobs = include_logprobs || logprob_token_ids.is_some();
    let logprobs_slots = match request.sampling_params.inner.logprobs {
        Some(k) => usize::try_from(k).map_or(0, |k| k + 1),
        None => logprob_token_ids.map_or(0, |ids| ids.len() + 1),
    };
    let mut sampling_params = request.sampling_params.inner;
    sampling_params.vllm_xargs = merge_kv_transfer_params(
        sampling_params.vllm_xargs,
        request.kv_transfer_params.as_ref(),
    );
    sampling_params.vllm_xargs = merge_ec_transfer_params(
        sampling_params.vllm_xargs,
        request.ec_transfer_params.as_ref(),
    );

    let text_request = TextRequest {
        request_id: ctx.request_id.clone(),
        prompt: Prompt::TokenIds(request.token_ids),
        mm_features,
        sampling_params,
        decode_options: TextDecodeOptions::default(),
        intermediate: false,
        priority: request.priority,
        cache_salt: request.cache_salt,
        add_special_tokens: false,
        data_parallel_rank: ctx.data_parallel_rank,
        session_id: ctx.session_id,
        reasoning_parser_kwargs: None,
        lora_request: lora_resolution.lora_request.clone(),
        arrival_time: None,
    };

    Ok(PreparedRequest {
        request_id: ctx.request_id,
        text_request,
        stream,
        options: ResponseOptions {
            include_usage,
            include_continuous_usage,
            include_logprobs,
            include_prompt_logprobs,
            logprobs_format,
            include_compact_logprobs,
            logprobs_slots,
        },
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use vllm_text::Prompt;

    use super::prepare_generate_request;
    use crate::lora::LoraModelResolution;
    use crate::routes::inference::generate::types::GenerateRequest;
    use crate::utils::ResolvedRequestContext;

    fn served(names: &[&str]) -> LoraModelResolution {
        LoraModelResolution {
            model_names: names.iter().map(|s| s.to_string()).collect(),
            lora_request: None,
        }
    }

    #[test]
    fn prepare_generate_request_maps_token_prompt_and_sampling_params() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22, 33],
            "priority": -3,
            "cache_salt": "salt",
            "sampling_params": {
                "max_tokens": 7,
                "logprobs": 2,
                "prompt_logprobs": 1,
                "ignore_eos": true
            },
            "kv_transfer_params": {
                "connector": "x"
            }
        }))
        .expect("parse request");

        let prepared = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            None,
        )
        .expect("prepare");

        assert_eq!(
            prepared.text_request.prompt,
            Prompt::TokenIds(vec![11, 22, 33])
        );
        assert_eq!(prepared.text_request.sampling_params.max_tokens, Some(7));
        assert_eq!(prepared.text_request.sampling_params.logprobs, Some(2));
        assert_eq!(
            prepared.text_request.sampling_params.prompt_logprobs,
            Some(1)
        );
        assert!(prepared.text_request.sampling_params.ignore_eos);
        assert_eq!(prepared.text_request.priority, -3);
        assert_eq!(prepared.text_request.cache_salt.as_deref(), Some("salt"));
        assert_eq!(
            prepared
                .text_request
                .sampling_params
                .vllm_xargs
                .and_then(|mut xargs| xargs.remove("kv_transfer_params")),
            Some(json!({"connector": "x"}))
        );
    }

    #[test]
    fn prepare_generate_request_forwards_thinking_token_budget() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22, 33],
            "sampling_params": {
                "thinking_token_budget": 64
            }
        }))
        .expect("parse request");

        let prepared = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            None,
        )
        .expect("prepare");

        // The raw inference route shares `vllm_text::SamplingParams`, so the
        // field is carried through to lowering exactly like the OpenAI routes
        // (normalization/validation then happens in `lower_sampling_params`).
        assert_eq!(
            prepared.text_request.sampling_params.thinking_token_budget,
            Some(64)
        );
    }

    #[test]
    fn prepare_generate_request_compact_counts_logprob_token_ids() {
        // `logprob_token_ids` without `logprobs`: the engine returns rows of
        // [sampled, ids...] (padded to the batch max with -inf), and Python's
        // `num_logprobs` is len(ids), so compact renders S = len(ids) + 1.
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "logprobs_format": "compact",
            "sampling_params": {"logprob_token_ids": [5, 6, 7]}
        }))
        .expect("parse request");
        let prepared = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            None,
        )
        .expect("prepare");
        assert_eq!(prepared.options.logprobs_slots, 4);
        assert!(prepared.options.include_compact_logprobs);
        // The default (openai) gate is unchanged: `logprobs` only.
        assert!(!prepared.options.include_logprobs);
    }

    #[test]
    fn prepare_generate_request_gates_continuous_usage_on_include_usage() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": true,
            "stream_options": {
                "continuous_usage_stats": true
            },
            "sampling_params": {}
        }))
        .expect("parse request");

        let prepared = prepare_generate_request(
            request,
            &served(&["Qwen/Qwen1.5-0.5B-Chat"]),
            ResolvedRequestContext::default(),
            None,
        )
        .expect("prepare");

        assert!(!prepared.options.include_usage);
        assert!(!prepared.options.include_continuous_usage);
    }
}
