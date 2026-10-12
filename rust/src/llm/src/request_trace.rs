use std::collections::HashMap;
use std::time::{Duration, UNIX_EPOCH};

use opentelemetry::trace::{SpanKind, TraceContextExt, Tracer};
use opentelemetry::{Context, KeyValue, global};
use vllm_engine_core_client::protocol::request::EngineCoreRequest;

pub(crate) struct RequestTrace {
    context: Context,
    finished: bool,
}

impl RequestTrace {
    pub(crate) fn start(request: &mut EngineCoreRequest) -> Option<Self> {
        let carrier: HashMap<String, String> = request
            .trace_headers
            .iter()
            .flat_map(|headers| headers.iter())
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect();
        let parent = global::get_text_map_propagator(|propagator| propagator.extract(&carrier));
        let tracer = global::tracer("vllm.llm_engine");
        let start_time =
            UNIX_EPOCH.checked_add(Duration::try_from_secs_f64(request.arrival_time).ok()?)?;
        let span = tracer
            .span_builder("llm_request")
            .with_kind(SpanKind::Server)
            .with_start_time(start_time)
            .with_attributes([KeyValue::new(
                "gen_ai.request.id",
                request.external_req_id.as_ref().unwrap_or(&request.request_id).clone(),
            )])
            .start_with_context(&tracer, &parent);
        let context = parent.with_span(span);
        if !context.span().span_context().is_valid() {
            return None;
        }
        if let Some(parameters) = request.sampling_params.as_ref() {
            if parameters.temperature != 0.0 {
                context.span().set_attribute(KeyValue::new(
                    "gen_ai.request.temperature",
                    f64::from(parameters.temperature),
                ));
            }
            if parameters.top_p != 0.0 {
                context.span().set_attribute(KeyValue::new(
                    "gen_ai.request.top_p",
                    f64::from(parameters.top_p),
                ));
            }
            if parameters.max_tokens != 0 {
                context.span().set_attribute(KeyValue::new(
                    "gen_ai.request.max_tokens",
                    i64::from(parameters.max_tokens),
                ));
            }
        }
        let mut propagated = HashMap::new();
        global::get_text_map_propagator(|propagator| {
            propagator.inject_context(&context, &mut propagated);
        });
        request.trace_headers = Some(propagated.into_iter().collect());
        Some(Self {
            context,
            finished: false,
        })
    }

    pub(crate) fn finish(&mut self, aborted: bool) {
        if !self.finished {
            if aborted {
                self.context.span().set_attribute(KeyValue::new("aborted", true));
            }
            self.context.span().end();
            self.finished = true;
        }
    }

    pub(crate) fn set_attributes(&self, attributes: impl IntoIterator<Item = KeyValue>) {
        self.context.span().set_attributes(attributes);
    }
}

impl Drop for RequestTrace {
    fn drop(&mut self) {
        self.finish(true);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use opentelemetry::Value;
    use opentelemetry_sdk::error::OTelSdkResult;
    use opentelemetry_sdk::propagation::TraceContextPropagator;
    use opentelemetry_sdk::trace::{SdkTracerProvider, SpanData, SpanExporter};
    use vllm_engine_core_client::protocol::output::{
        EngineCoreEvent, EngineCoreEventType, EngineCoreOutput,
    };
    use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

    use super::*;
    use crate::FinishReason;
    use crate::request_metrics::RequestMetricsTracker;

    #[derive(Clone, Debug, Default)]
    struct CaptureExporter(Arc<Mutex<Vec<SpanData>>>);

    impl SpanExporter for CaptureExporter {
        fn export(
            &self,
            batch: Vec<SpanData>,
        ) -> impl std::future::Future<Output = OTelSdkResult> + Send {
            self.0.lock().unwrap().extend(batch);
            async { Ok(()) }
        }
    }

    #[test]
    fn root_preserves_parent_reports_metrics_and_ends_once_on_finish_or_drop() {
        let exporter = CaptureExporter::default();
        let provider = SdkTracerProvider::builder().with_simple_exporter(exporter.clone()).build();
        global::set_tracer_provider(provider.clone());
        global::set_text_map_propagator(TraceContextPropagator::new());
        let mut request = EngineCoreRequest {
            request_id: "internal".into(),
            external_req_id: Some("external".into()),
            arrival_time: 1000.0,
            trace_headers: Some(
                [(
                    "traceparent".into(),
                    "00-33333333333333333333333333333333-1111111111111111-01".into(),
                )]
                .into(),
            ),
            sampling_params: Some(EngineCoreSamplingParams {
                max_tokens: 16,
                temperature: 0.0,
                top_p: 0.95,
                ..EngineCoreSamplingParams::for_test()
            }),
            ..EngineCoreRequest::default()
        };
        let mut trace = RequestTrace::start(&mut request).unwrap();
        assert!(
            request.trace_headers.as_ref().unwrap()["traceparent"]
                .starts_with("00-33333333333333333333333333333333-")
        );
        assert_ne!(
            request.trace_headers.as_ref().unwrap()["traceparent"],
            "00-33333333333333333333333333333333-1111111111111111-01"
        );
        let mut metrics =
            RequestMetricsTracker::new("model".into(), 0, 1000.0, 3, Some(16), 1, true);
        metrics.observe_output(
            1002.0,
            1002.0,
            &EngineCoreOutput {
                new_token_ids: vec![1, 2],
                events: Some(vec![
                    EngineCoreEvent {
                        r#type: EngineCoreEventType::Queued,
                        timestamp: 1000.0,
                    },
                    EngineCoreEvent {
                        r#type: EngineCoreEventType::Scheduled,
                        timestamp: 1001.0,
                    },
                ]),
                ..EngineCoreOutput::default()
            },
        );
        metrics.record_finished(1003.0, FinishReason::Length, Some(&mut trace));
        drop(trace);
        drop(RequestTrace::start(&mut request).unwrap());
        provider.force_flush().unwrap();
        let captured = exporter.0.lock().unwrap();
        assert_eq!(captured.len(), 2);
        assert_eq!(captured[0].name, "llm_request");
        assert_eq!(
            captured[0].span_context.trace_id().to_string(),
            "33333333333333333333333333333333"
        );
        assert_eq!(captured[0].parent_span_id.to_string(), "1111111111111111");
        let attributes: HashMap<_, _> = captured[0]
            .attributes
            .iter()
            .map(|entry| (entry.key.as_str(), &entry.value))
            .collect();
        assert_eq!(
            attributes["gen_ai.request.id"],
            &Value::String("external".into())
        );
        assert_eq!(attributes["gen_ai.usage.prompt_tokens"], &Value::I64(3));
        assert_eq!(attributes["gen_ai.usage.completion_tokens"], &Value::I64(2));
        assert_eq!(attributes["gen_ai.request.max_tokens"], &Value::I64(16));
        assert_eq!(attributes["gen_ai.latency.e2e"], &Value::F64(3.0));
        assert_eq!(attributes["gen_ai.latency.time_in_queue"], &Value::F64(1.0));
        assert_eq!(
            attributes["gen_ai.latency.time_in_model_prefill"],
            &Value::F64(1.0)
        );
        assert!(!attributes.contains_key("gen_ai.request.temperature"));
        assert!(!attributes.contains_key("aborted"));
        assert!(
            captured[1]
                .attributes
                .iter()
                .any(|entry| entry.key.as_str() == "aborted" && entry.value == Value::Bool(true))
        );
        drop(captured);
        provider.shutdown().unwrap();
    }
}
