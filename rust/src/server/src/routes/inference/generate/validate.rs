// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use super::compact::LogprobsFormat;
use super::types::GenerateRequest;
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

    let Some(logprobs_format) = LogprobsFormat::parse(request.logprobs_format.as_ref()) else {
        bail_invalid_request!(
            param = "logprobs_format",
            "logprobs_format must be \"openai\" or \"compact\"."
        );
    };

    // SPEC v3 field switches: booleans, and non-default values only with
    // the compact format.
    for (param, value) in [
        ("compact_include_sampled", &request.compact_include_sampled),
        ("compact_include_ranks", &request.compact_include_ranks),
    ] {
        match value {
            None | Some(serde_json::Value::Bool(true)) => {}
            Some(serde_json::Value::Bool(false)) if logprobs_format == LogprobsFormat::Compact => {}
            Some(serde_json::Value::Bool(false)) => {
                return Err(ApiError::invalid_request(
                    format!("{param}=false requires logprobs_format \"compact\"."),
                    Some(param),
                ));
            }
            Some(_) => {
                return Err(ApiError::invalid_request(
                    format!("{param} must be a boolean."),
                    Some(param),
                ));
            }
        }
    }

    // The full-vocabulary payload (`logprobs: -1`) has no compact encoding.
    if logprobs_format == LogprobsFormat::Compact
        && request.sampling_params.inner.logprobs.is_some_and(|k| k < 0)
    {
        bail_invalid_request!(
            param = "logprobs",
            "logprobs=-1 is not supported with logprobs_format \"compact\"."
        );
    }

    if request.sampling_params.n.unwrap_or(1) != 1 {
        bail_invalid_request!(param = "n", "Only n=1 is supported.");
    }

    if let Some(start) = &request.sampling_params.routed_experts_prompt_start
        && start.as_u64() != Some(0)
    {
        bail_invalid_request!(
            param = "routed_experts_prompt_start",
            "Only routed_experts_prompt_start=0 is supported."
        );
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
    fn validate_request_compat_rejects_compact_full_vocab_logprobs() {
        let served = served(&["Qwen/Qwen1.5-0.5B-Chat"]);
        let request = |format: &str, logprobs: i32| -> GenerateRequest {
            serde_json::from_value(json!({
                "token_ids": [11, 22],
                "logprobs_format": format,
                "sampling_params": {"logprobs": logprobs}
            }))
            .expect("parse request")
        };
        let error = validate_request_compat(&request("compact", -1), &served)
            .expect_err("compact with logprobs=-1 is not supported");
        assert_eq!(
            error.to_error_response().error.param.as_deref(),
            Some("logprobs")
        );
        assert!(validate_request_compat(&request("compact", 0), &served).is_ok());
        assert!(validate_request_compat(&request("compact", 128), &served).is_ok());
        // The default format keeps accepting -1 (unchanged from base).
        assert!(validate_request_compat(&request("openai", -1), &served).is_ok());
    }

    #[test]
    fn validate_request_compat_checks_logprobs_format() {
        let served = served(&["Qwen/Qwen1.5-0.5B-Chat"]);
        for (format, ok) in [
            (json!("openai"), true),
            (json!("compact"), true),
            (json!("Compact"), false),
            (json!("numpy"), false),
            (json!(""), false),
            // Same as the Python frontend's Literal field: an explicit null or
            // a non-string value is rejected (only an absent field defaults).
            (json!(null), false),
            (json!(1), false),
            (json!(["compact"]), false),
        ] {
            let request: GenerateRequest = serde_json::from_value(json!({
                "token_ids": [11, 22],
                "logprobs_format": format,
                "sampling_params": {}
            }))
            .expect("parse request");
            assert_eq!(
                validate_request_compat(&request, &served).is_ok(),
                ok,
                "format={format}"
            );
        }
        // An absent field means the default format.
        let request: GenerateRequest = serde_json::from_value(json!({
            "token_ids": [11, 22],
            "sampling_params": {}
        }))
        .expect("parse request");
        assert!(request.logprobs_format.is_none());
        assert!(validate_request_compat(&request, &served).is_ok());
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

    #[test]
    fn validate_request_compat_rejects_nonzero_routed_experts_prompt_start() {
        let served = served(&["Qwen/Qwen1.5-0.5B-Chat"]);
        let check = |value: serde_json::Value| {
            let request: GenerateRequest = serde_json::from_value(json!({
                "token_ids": [11, 22],
                "sampling_params": {"routed_experts_prompt_start": value}
            }))
            .expect("parse request");
            validate_request_compat(&request, &served)
                .map_err(|error| error.to_error_response().error.param)
        };
        assert!(check(json!(0)).is_ok());
        for bad in [json!(1), json!(-1), json!(2.5), json!("0"), json!(null)] {
            assert_eq!(
                check(bad),
                Err(Some("routed_experts_prompt_start".to_string()))
            );
        }
    }

    #[test]
    fn validate_request_compat_checks_compact_switches() {
        let served = served(&["Qwen/Qwen1.5-0.5B-Chat"]);
        let check = |format: Option<&str>, field: &str, value: serde_json::Value| {
            let mut body = json!({
                "token_ids": [11, 22],
                "sampling_params": {"logprobs": 2}
            });
            if let Some(format) = format {
                body["logprobs_format"] = json!(format);
            }
            body[field] = value;
            let request: GenerateRequest = serde_json::from_value(body).expect("parse request");
            validate_request_compat(&request, &served)
                .map_err(|error| error.to_error_response().error.param)
        };
        for field in ["compact_include_sampled", "compact_include_ranks"] {
            // Compact accepts both values.
            assert!(check(Some("compact"), field, json!(false)).is_ok());
            assert!(check(Some("compact"), field, json!(true)).is_ok());
            // The default value is accepted with any format.
            assert!(check(None, field, json!(true)).is_ok());
            assert!(check(Some("openai"), field, json!(true)).is_ok());
            // A non-default value needs compact.
            assert_eq!(
                check(None, field, json!(false)),
                Err(Some(field.to_string()))
            );
            assert_eq!(
                check(Some("openai"), field, json!(false)),
                Err(Some(field.to_string()))
            );
            // Non-booleans (including null) are rejected.
            for bad in [json!(null), json!(0), json!("false")] {
                assert_eq!(
                    check(Some("compact"), field, bad),
                    Err(Some(field.to_string()))
                );
            }
        }
    }
}
