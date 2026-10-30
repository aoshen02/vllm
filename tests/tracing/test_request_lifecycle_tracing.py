# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from opentelemetry.sdk.environment_variables import (
    OTEL_EXPORTER_OTLP_TRACES_INSECURE,
)

import vllm.v1.request as request_module
from tests.tracing.conftest import FAKE_TRACE_SERVER_ADDRESS, FakeTraceService
from tests.v1.core.utils import create_requests, create_scheduler, mock_kv
from vllm.sampling_params import SamplingParams
from vllm.tracing import (
    SpanAttributes,
    SpanKind,
    init_tracer,
    is_otel_available,
    otel,
    start_request_span,
)
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.engine.utils import EngineZmqAddresses, launch_core_engines
from vllm.v1.metrics.stats import IterationStats
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

# Skip everything if OTel is missing
pytestmark = pytest.mark.skipif(not is_otel_available(), reason="OTel required")


class TestRequestLifecycleTracing:
    """Focuses on request lifecycle spans and root llm_request hierarchy."""

    @pytest.fixture(autouse=True)
    def setup_tracing(self, monkeypatch):
        monkeypatch.setenv(OTEL_EXPORTER_OTLP_TRACES_INSECURE, "true")
        init_tracer("test.request_lifecycle", FAKE_TRACE_SERVER_ADDRESS)

    @pytest.mark.parametrize(
        "status",
        [
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_ABORTED,
            RequestStatus.FINISHED_ERROR,
        ],
    )
    def test_remote_wait_accumulates_and_cleanup_ends_once(
        self, trace_service, monkeypatch, status
    ):
        root, headers = start_request_span("llm_request", time.time_ns())
        request = Request.from_engine_core_request(make_engine_request(headers), None)
        timestamps = iter([10.0, 10.75, 20.0, 21.25])
        anchors = iter([100_000_000_000, 200_000_000_000])
        monkeypatch.setattr(
            request_module,
            "time",
            SimpleNamespace(
                monotonic=lambda: next(timestamps), time_ns=lambda: next(anchors)
            ),
        )
        request.start_remote_kv_wait()
        request.stop_remote_kv_wait()
        request.start_remote_kv_wait()
        request.status = status
        request.trace_cleanup()
        request.trace_cleanup()
        request.stop_remote_kv_wait()
        root.end()
        assert request.remote_kv_wait_time == 2.0
        assert request.remote_kv_wait_started_at is None
        assert request.async_kv_load_start_time_ns is None
        assert otel._GLOBAL_TRACER_PROVIDER.force_flush()
        spans = [
            span
            for span in trace_service.get_all_spans()
            if span["name"] == "vllm.request.wait_remote_kv"
        ]
        assert len(spans) == 2
        assert (
            sum(
                span["end_time_unix_nano"] - span["start_time_unix_nano"]
                for span in spans
            )
            == 2_000_000_000
        )
        assert sum(bool(span["attributes"].get("aborted")) for span in spans) == int(
            status in (RequestStatus.FINISHED_ABORTED, RequestStatus.FINISHED_ERROR)
        )

    def test_lifecycle_spans_nested_under_root_request(
        self, trace_service: FakeTraceService
    ):
        """Verify root span establishes valid hierarchy for lifecycle spans."""
        arrival_time_ns = time.time_ns()

        # 1. Start root llm_request span (as done in AsyncLLM/OutputProcessor)
        root_span, updated_headers = start_request_span(
            span_name="llm_request",
            start_time=arrival_time_ns,
            kind=SpanKind.SERVER,
        )
        assert root_span is not None
        assert updated_headers is not None
        assert "traceparent" in updated_headers

        # 2. Instantiate engine Request with updated_headers
        sampling_params = SamplingParams(max_tokens=10)
        req = Request(
            request_id="req-123",
            prompt_token_ids=[1, 2, 3],
            sampling_params=sampling_params,
            pooling_params=None,
            arrival_time=arrival_time_ns / 1e9,
            trace_headers=updated_headers,
        )

        # 3. Simulate Queue -> Prefill -> Decode lifecycle
        req.trace_end_queuing()
        req.prefill_start_time_ns = time.time_ns()
        time.sleep(0.01)

        req.trace_end_prefill()
        req.decode_start_time_ns = time.time_ns()
        time.sleep(0.01)

        req.trace_end_decode()

        # 4. End root llm_request span
        root_span.set_attributes(
            {
                SpanAttributes.GEN_AI_REQUEST_ID: "req-123",
                SpanAttributes.GEN_AI_USAGE_PROMPT_TOKENS: 3,
                SpanAttributes.GEN_AI_USAGE_COMPLETION_TOKENS: 5,
            }
        )
        root_span.end()

        # 5. Verify all spans and hierarchy
        assert trace_service.wait_for_spans(count=4)
        spans = trace_service.get_all_spans()

        root = next(s for s in spans if s["name"] == "llm_request")
        queue_span = next(s for s in spans if s["name"] == "vllm.request.queue")
        prefill_span = next(s for s in spans if s["name"] == "vllm.request.prefill")
        decode_span = next(s for s in spans if s["name"] == "vllm.request.decode")

        # Check all child spans have root_span as parent
        root_span_id = root["span_id"]
        assert queue_span["parent_span_id"] == root_span_id
        assert prefill_span["parent_span_id"] == root_span_id
        assert decode_span["parent_span_id"] == root_span_id

        # Check all spans share the same trace_id
        trace_id = root["trace_id"]
        assert queue_span["trace_id"] == trace_id
        assert prefill_span["trace_id"] == trace_id
        assert decode_span["trace_id"] == trace_id

    def test_preemption_and_cleanup_spans(self, trace_service: FakeTraceService):
        """Verify preemption and cleanup gracefully close active spans."""
        arrival_time_ns = time.time_ns()
        root_span, updated_headers = start_request_span(
            span_name="llm_request",
            start_time=arrival_time_ns,
            kind=SpanKind.SERVER,
        )

        sampling_params = SamplingParams(max_tokens=10)
        req = Request(
            request_id="req-456",
            prompt_token_ids=[1, 2, 3],
            sampling_params=sampling_params,
            pooling_params=None,
            arrival_time=arrival_time_ns / 1e9,
            trace_headers=updated_headers,
        )

        # Start prefill then preempt
        req.prefill_start_time_ns = time.time_ns()
        req.trace_preempt()

        # Resume decode then finish with abort
        req.decode_start_time_ns = time.time_ns()
        req.status = RequestStatus.FINISHED_ABORTED
        req.trace_cleanup()

        root_span.end()

        assert trace_service.wait_for_spans(count=3)
        spans = trace_service.get_all_spans()

        prefill_span = next(s for s in spans if s["name"] == "vllm.request.prefill")
        decode_span = next(s for s in spans if s["name"] == "vllm.request.decode")

        assert prefill_span["attributes"].get("preempted") is True
        assert decode_span["attributes"].get("aborted") is True

    def test_disabled_processor_preserves_headers_without_root(self, trace_service):
        headers = {"traceparent": "00-" + "a" * 32 + "-" + "b" * 16 + "-01"}
        request = make_engine_request(headers)
        processor = OutputProcessor(None, log_stats=False, tracing_enabled=False)
        processor.add_request(request, prompt=None)
        assert processor.request_states[request.request_id].trace_span is None
        assert request.trace_headers == headers
        processor.abort_requests([request.request_id], internal=True)
        assert otel._GLOBAL_TRACER_PROVIDER.force_flush()
        assert trace_service.get_all_spans() == []

    def test_headerless_request_emits_no_orphan_span(self, trace_service):
        request = Request.from_engine_core_request(make_engine_request(None), None)
        request.trace_end_queuing()
        request.trace_cleanup()
        assert otel._GLOBAL_TRACER_PROVIDER.force_flush()
        assert trace_service.get_all_spans() == []

    @pytest.mark.parametrize("abort", [False, True])
    @pytest.mark.parametrize("log_stats", [False, True])
    def test_processor_closes_root_and_propagates_parent(
        self, trace_service, abort, log_stats
    ):
        trace_id, parent_id = "a" * 32, "b" * 16
        request = make_engine_request(
            {"traceparent": f"00-{trace_id}-{parent_id}-01", "tracestate": "test=value"}
        )
        processor = OutputProcessor(None, log_stats=log_stats, tracing_enabled=True)
        processor.add_request(request, prompt=None)
        assert request.trace_headers["traceparent"].split("-")[1] == trace_id
        assert request.trace_headers["traceparent"].split("-")[2] != parent_id
        assert request.trace_headers["tracestate"] == "test=value"
        engine_request = Request.from_engine_core_request(request, None)
        engine_request.trace_end_queuing()
        if abort:
            assert processor.abort_requests([request.request_id], internal=True)
            assert not processor.abort_requests([request.request_id], internal=True)
        else:
            result = processor.process_outputs(
                [
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[7],
                        finish_reason=FinishReason.LENGTH,
                        trace_headers=request.trace_headers,
                    )
                ],
                engine_core_timestamp=time.monotonic(),
                iteration_stats=IterationStats() if log_stats else None,
            )
            assert result.request_outputs[0].outputs[0].token_ids == [7]
        assert processor.request_states == {}
        assert otel._GLOBAL_TRACER_PROVIDER.force_flush()
        spans = trace_service.get_all_spans()
        roots = [span for span in spans if span["name"] == "llm_request"]
        assert len(roots) == 1
        assert roots[0]["parent_span_id"] == parent_id
        assert roots[0]["trace_id"] == trace_id
        queue_span = next(
            span for span in spans if span["name"] == "vllm.request.queue"
        )
        assert queue_span["parent_span_id"] == roots[0]["span_id"]
        assert queue_span["trace_id"] == trace_id
        assert all(
            span["end_time_unix_nano"] >= span["start_time_unix_nano"] for span in spans
        )

    @pytest.mark.parametrize(
        "matched_tokens,expected_phase,expected_tokens",
        [
            (0, "prefill", 64),
            (32, "prefill", 32),
            (64, "decode", 1),
        ],
    )
    def test_scheduler_remote_hit_phase(
        self, matched_tokens, expected_phase, expected_tokens
    ):
        scheduler = create_scheduler(
            enable_prefix_caching=True,
            use_kv_connector=mock_kv(
                matched_tokens=matched_tokens, is_async=matched_tokens > 0
            ),
            block_size=16,
        )
        request = create_requests(num_requests=1, num_tokens=64, block_size=16)[0]
        scheduler.add_request(request)
        output = scheduler.schedule()
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            scheduler.update_from_output(
                output,
                ModelRunnerOutput(
                    req_ids=[],
                    req_id_to_index={},
                    sampled_token_ids=[],
                    logprobs=None,
                    prompt_logprobs_dict={},
                    pooler_output=[],
                    kv_connector_output=KVConnectorOutput(
                        finished_recving={request.request_id}
                    ),
                ),
            )
            output = scheduler.schedule()
        assert output.num_scheduled_tokens[request.request_id] == expected_tokens
        assert (request.prefill_start_time_ns is not None) == (
            expected_phase == "prefill"
        )
        assert (request.decode_start_time_ns is not None) == (
            expected_phase == "decode"
        )

    def test_launch_endpoint_reaches_real_child_exporter(
        self, trace_service, monkeypatch
    ):
        config = create_scheduler().vllm_config
        config.parallel_config.data_parallel_size_local = 0
        config.parallel_config.data_parallel_rank_local = None
        config.observability_config.otlp_traces_endpoint = (
            "grpc://" + FAKE_TRACE_SERVER_ADDRESS
        )
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
        monkeypatch.setenv("TRACEPARENT", "00-" + "c" * 32 + "-" + "d" * 16 + "-01")
        addresses = EngineZmqAddresses(inputs=[], outputs=[])
        launch = launch_core_engines(config, object, False, addresses)
        try:
            launch.__enter__()
            assert (
                os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"]
                == config.observability_config.otlp_traces_endpoint
            )
            child = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "import time; from vllm.tracing import otel; "
                    "otel.init_otel_worker_tracer("
                    "'tracing.cpu.child', 'test', 'child'); "
                    "span, headers = otel.start_request_span_otel("
                    "'child', time.time_ns()); "
                    "assert headers; span.end(); "
                    "assert otel._GLOBAL_TRACER_PROVIDER.force_flush(); "
                    "otel._GLOBAL_TRACER_PROVIDER.shutdown()",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            assert child.returncode == 0
        finally:
            launch.gen.close()
        assert trace_service.wait_for_spans(1)
        child_span = next(
            span for span in trace_service.get_all_spans() if span["name"] == "child"
        )
        assert child_span["trace_id"] == "c" * 32
        assert child_span["parent_span_id"] == "d" * 16


@pytest.fixture(autouse=True)
def close_test_provider(trace_service):
    yield
    assert otel._GLOBAL_TRACER_PROVIDER.force_flush()
    otel._GLOBAL_TRACER_PROVIDER.shutdown()


def make_engine_request(headers):
    return EngineCoreRequest(
        request_id="tracing-internal",
        external_req_id="tracing-external",
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=SamplingParams(max_tokens=2, detokenize=False),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        trace_headers=headers,
    )
