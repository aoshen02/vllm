// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::types::{GenerateRequest, LogprobsFormat};
use crate::error::{ApiError, bail_invalid_request};

/// Enforce the minimal compatibility contract for the Rust token generate
/// route.
pub(crate) fn validate_request_compat(
    request: &GenerateRequest,
    served_model_names: &[String],
) -> Result<(), ApiError> {
    if let Some(model) = request.model.as_ref()
        && !served_model_names.iter().any(|n| n == model)
    {
        return Err(ApiError::model_not_found(model.clone()));
    }

    if request.stream_options.is_some() && !request.stream {
        bail_invalid_request!(
            param = "stream_options",
            "stream_options are only supported when stream=true."
        );
    }

    if request.logprobs_format == Some(LogprobsFormat::Compact) {
        if request.stream {
            bail_invalid_request!(
                param = "logprobs_format",
                "logprobs_format \"compact\" is not available when `stream=true`."
            );
        }
        // The full-vocabulary payload (`logprobs: -1`) has no compact form.
        if request.sampling_params.inner.logprobs.is_some_and(|k| k < 0) {
            bail_invalid_request!(
                param = "logprobs",
                "logprobs=-1 is not supported with logprobs_format \"compact\"."
            );
        }
    }

    if request.sampling_params.n.unwrap_or(1) != 1 {
        bail_invalid_request!(param = "n", "Only n=1 is supported.");
    }

    if request.token_ids.is_empty() {
        bail_invalid_request!(
            param = "token_ids",
            "token_ids must contain at least one token ID."
        );
    }

    if request.sampling_params.inner.max_tokens == Some(0) {
        bail_invalid_request!(
            param = "sampling_params",
            "max_tokens must be greater than 0."
        );
    }

    if let Some(prompt_logprobs) = request.sampling_params.inner.prompt_logprobs {
        if prompt_logprobs < 0 && prompt_logprobs != -1 {
            bail_invalid_request!(
                param = "sampling_params",
                "`prompt_logprobs` must be a non-negative value or -1."
            );
        }

        if request.stream {
            bail_invalid_request!(
                param = "sampling_params",
                "`prompt_logprobs` are not available when `stream=true`."
            );
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::validate_request_compat;
    use crate::routes::inference::generate::types::GenerateRequest;

    fn base_request() -> GenerateRequest {
        serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {}
        }))
        .expect("parse request")
    }

    fn served(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn validate_request_compat_accepts_streaming() {
        let request = GenerateRequest {
            stream: true,
            ..base_request()
        };
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok());
    }

    #[test]
    fn validate_request_compat_rejects_stream_options_without_streaming() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": false,
            "stream_options": {"include_usage": true},
            "sampling_params": {}
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());
    }

    #[test]
    fn validate_request_compat_rejects_parallel_sampling() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {"n": 4}
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());
    }

    #[test]
    fn validate_request_compat_accepts_explicit_n_one() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "sampling_params": {"n": 1}
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok());
    }

    #[test]
    fn validate_request_compat_compact_rejects_stream_and_full_vocab_logprobs() {
        let served = served(&["Qwen/Qwen1.5-0.5B-Chat"]);
        let request = |extra: serde_json::Value| -> GenerateRequest {
            let mut body = json!({
                "token_ids": [11, 22],
                "logprobs_format": "compact",
                "sampling_params": {"logprobs": 5}
            });
            body.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
            serde_json::from_value(body).expect("parse request")
        };
        assert!(validate_request_compat(&request(json!({})), &served).is_ok());
        for extra in [
            json!({"stream": true}),
            json!({"sampling_params": {"logprobs": -1}}),
        ] {
            assert!(validate_request_compat(&request(extra), &served).is_err());
        }
    }

    #[test]
    fn validate_request_compat_rejects_empty_token_ids() {
        let request = GenerateRequest {
            token_ids: Vec::new(),
            ..base_request()
        };
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());
    }

    #[test]
    fn validate_request_compat_rejects_streaming_prompt_logprobs() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": true,
            "sampling_params": {
                "prompt_logprobs": 0
            }
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());

        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": true,
            "sampling_params": {
                "prompt_logprobs": 1
            }
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());

        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": true,
            "sampling_params": {
                "prompt_logprobs": -1
            }
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_err());
    }

    #[test]
    fn validate_request_compat_accepts_non_stream_prompt_logprobs() {
        let request: GenerateRequest = serde_json::from_value(json!({
            "model": "Qwen/Qwen1.5-0.5B-Chat",
            "token_ids": [11, 22],
            "stream": false,
            "sampling_params": {
                "prompt_logprobs": 1
            }
        }))
        .expect("parse request");
        assert!(validate_request_compat(&request, &served(&["Qwen/Qwen1.5-0.5B-Chat"])).is_ok());
    }
}
