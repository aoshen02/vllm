# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The frontend-only sample-logprobs container hook (``vllm.logprobs``):
registration, selection, the core-owned ``SampleLogprobsHandle`` (structural
containment of third-party containers), EngineCore wire, detokenizer skip."""

import asyncio
import copy
import io
import logging
import threading
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm import logprobs as logprobs_mod
from vllm.entrypoints.scale_out.token_in_token_out.protocol import GenerateRequest
from vllm.logging_utils import NewLineFormatter
from vllm.logprobs import (
    SampleLogprobsHandle,
    create_sample_logprobs,
    register_sample_logprobs_container,
    sample_logprobs_skip_text,
    set_sample_logprobs_container,
)
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor, RequestOutputCollector
from vllm.v1.engine.parallel_sampling import ParentRequest
from vllm.v1.outputs import LogprobsLists
from vllm.v1.serial_utils import MsgpackEncoder


class RowsContainer:
    """Minimal container recording the rows it receives."""

    def __init__(self, params=None, rows=None):
        self.rows: list = [] if rows is None else rows

    def append_rows(self, token_ids, logprobs, ranks):
        self.rows.append((token_ids, logprobs, ranks))

    def extend(self, other):
        self.rows.extend(other.rows)

    def __len__(self):
        return sum(len(r) for _, _, r in self.rows)

    def __getitem__(self, index):
        return RowsContainer(rows=list(self.rows))


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(logprobs_mod, "_SAMPLE_LOGPROBS_CONTAINERS", {})
    return logprobs_mod._SAMPLE_LOGPROBS_CONTAINERS


def _register(name, factory, skip=True):
    register_sample_logprobs_container(name, factory, skip_sampled_text=skip)


def _request(rid, params, kind=RequestOutputKind.FINAL_ONLY):
    params.output_kind = kind
    return EngineCoreRequest(
        request_id=rid,
        external_req_id=rid,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        arrival_time=0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        sampling_params=params,
        pooling_params=None,
    )


def _rows(n, width, start=0):
    rng = np.random.default_rng(start)
    ids = rng.integers(0, 1000, (n, width)).astype(np.int64)
    lps = (-rng.random((n, width))).astype(np.float32)
    return ids, lps, rng.integers(1, 9, n).astype(np.int64)


def _step(rid, ids, lps, ranks):
    return EngineCoreOutput(
        request_id=rid,
        new_token_ids=ids[:, 0].tolist(),
        new_logprobs=LogprobsLists(ids, lps, ranks),
    )


# ---------------------------------------------------------------- basics
def test_default_requests_are_unchanged(registry):
    params = SamplingParams(logprobs=2)
    assert create_sample_logprobs(False, params) == []
    assert type(create_sample_logprobs(True, params)).__name__ == "FlatLogprobs"
    assert not sample_logprobs_skip_text(params)


@pytest.mark.parametrize(
    "key", ["sample_logprobs_container", "_sample_logprobs_container"]
)
@pytest.mark.parametrize("value", ["rl", 123, None])
def test_clients_cannot_select_a_container(key, value):
    """The selector is private: client JSON keys are ignored (any type, like
    any unknown sampling field on base) and not in schemas."""
    request = GenerateRequest.model_validate(
        {"token_ids": [1], "sampling_params": {"logprobs": 2, key: value}}
    )
    assert request.sampling_params._sample_logprobs_container is None
    assert "sample_logprobs_container" not in str(GenerateRequest.model_json_schema())


def test_rows_reach_the_container_and_stay_off_the_engine_wire(registry):
    _register("rows", RowsContainer)
    params = SamplingParams(max_tokens=10, logprobs=2)
    set_sample_logprobs_container(params, "rows")
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    request = _request("a", params)
    processor.add_request(
        request, None, queue=RequestOutputCollector(params.output_kind, "a")
    )
    assert request.sampling_params._sample_logprobs_container is None
    assert params._sample_logprobs_container == "rows"
    assert b"rows" not in b"".join(bytes(b) for b in MsgpackEncoder().encode(request))
    ids, lps, ranks = _rows(3, 5)  # engine rows padded to 5 > k + 1 = 3 slots
    processor.process_outputs([_step("a", ids, lps, ranks)])
    state = processor.request_states["a"]
    handle = state.logprobs_processor.logprobs
    assert type(handle) is SampleLogprobsHandle and len(handle) == 3
    ((got_ids, got_lps, got_ranks),) = handle.unwrap().rows
    np.testing.assert_array_equal(got_ids, ids[:, :3])
    np.testing.assert_array_equal(got_lps, lps[:, :3])
    np.testing.assert_array_equal(got_ranks, ranks)
    expected = 0.0
    for value in lps[:, 0].tolist():
        expected += value
    assert state.logprobs_processor.cumulative_logprob == expected


# ----------------------------------------------------------- registration
def test_registration_never_calls_the_factory(registry):
    """Codex r17 #2/#5: no probe (no third-party code at registration, so no
    deadlock); the skip flag is an explicit argument."""
    calls = []

    def factory(params):
        calls.append(params)
        return RowsContainer()

    _register("rows", factory)
    _register("rows-text", RowsContainer, skip=False)
    assert calls == []
    assert registry["rows"][1] is True and registry["rows-text"][1] is False


def test_registration_is_idempotent_and_rejects_takeovers(registry):
    _register("rows", RowsContainer)
    _register("rows", RowsContainer)  # same factory and flag: no-op
    with pytest.raises(ValueError, match="already registered"):
        _register("rows", lambda params: RowsContainer())
    with pytest.raises(ValueError, match="already registered"):
        _register("rows", RowsContainer, skip=False)
    with pytest.raises(ValueError, match="Unknown"):
        set_sample_logprobs_container(SamplingParams(), "nope")


@pytest.mark.parametrize("name", ["", 123, None, b"rows"])
def test_registration_rejects_invalid_names(registry, name):
    with pytest.raises(ValueError, match="Invalid"):
        _register(name, RowsContainer)
    assert not registry


def test_concurrent_registration_has_exactly_one_winner(registry):
    barrier = threading.Barrier(8)
    results = []

    def worker(i):
        factory = lambda params, i=i: RowsContainer()  # noqa: E731
        barrier.wait()
        try:
            _register("rows", factory)
            results.append(("ok", factory))
        except ValueError:
            results.append(("rejected", factory))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    winners = [f for r, f in results if r == "ok"]
    assert len(winners) == 1 and registry["rows"][0] is winners[0]


def test_factory_registering_another_container_does_not_deadlock(registry):
    """Codex r17 #2: the registry lock is never held around third-party code."""

    def factory(params):
        _register("inner", RowsContainer)  # re-enters the registry
        return RowsContainer()

    _register("outer", factory)
    params = SamplingParams(logprobs=1)
    set_sample_logprobs_container(params, "outer")
    done = []
    thread = threading.Thread(
        target=lambda: done.append(create_sample_logprobs(False, params))
    )
    thread.start()
    thread.join(timeout=10)
    assert done and not done[0].broken and "inner" in registry


def test_skip_text_is_false_without_a_selected_container(registry):
    _register("rows", RowsContainer)
    registry[""] = (RowsContainer, True)  # even a smuggled empty name
    assert not sample_logprobs_skip_text(SamplingParams(logprobs=1))


@pytest.mark.parametrize("logprobs", [None, 2])
@pytest.mark.parametrize("stop", [None, ["x"]])
@pytest.mark.parametrize("skip", [True, False])
def test_sampled_text_skip_follows_the_registration(
    registry, monkeypatch, logprobs, stop, skip
):
    _register("rows", RowsContainer, skip=skip)
    monkeypatch.setattr(
        "vllm.v1.engine.output_processor.IncrementalDetokenizer.from_new_request",
        lambda tokenizer, request: ("detok", tokenizer),
    )
    tokenizer = MagicMock()
    for selected in (False, True):
        params = SamplingParams(max_tokens=4, logprobs=logprobs, stop=stop)
        if selected:
            set_sample_logprobs_container(params, "rows")
        processor = OutputProcessor(tokenizer=tokenizer, log_stats=False)
        processor.add_request(_request("r", params), None)
        skipped = selected and skip and not stop
        assert processor.request_states["r"].detokenizer == (
            "detok",
            None if skipped else tokenizer,
        )


# ------------------------------------------------- structural containment
class Hostile(RowsContainer):
    """Raises from ``op`` once it holds rows (i.e. during output processing)."""

    op = ""

    def _maybe(self, op):
        if op == self.op and self.rows:
            raise RuntimeError(f"hostile {op}")

    def append_rows(self, token_ids, logprobs, ranks):
        self._maybe("append_rows")
        super().append_rows(token_ids, logprobs, ranks)

    def __len__(self):
        self._maybe("__len__")
        return super().__len__()

    def __bool__(self):
        self._maybe("__bool__")
        return bool(self.rows)

    def __getitem__(self, index):
        self._maybe("__getitem__")
        return self

    def extend(self, other):
        self._maybe("extend")
        super().extend(other)


class HostileList(list):
    """Codex r17 #1: a list subclass container, raising from list behaviour."""

    def __init__(self, params=None):
        super().__init__()

    def append_rows(self, token_ids, logprobs, ranks):
        self.append(len(ranks))

    def __bool__(self):
        raise RuntimeError("hostile list __bool__")

    def __len__(self):
        raise RuntimeError("hostile list __len__")

    def __getitem__(self, index):
        raise RuntimeError("hostile list __getitem__")

    def extend(self, other):
        raise RuntimeError("hostile list extend")


class HostileClass(RowsContainer):
    """Codex r17 #1: a raising ``__class__`` breaks isinstance on it."""

    @property  # type: ignore[misc]
    def __class__(self):
        raise RuntimeError("hostile __class__")


def _hostile(op):
    return type(f"Hostile_{op.strip('_')}", (Hostile,), {"op": op})


class CustomBaseException(BaseException):
    pass


def _raising(error_type):
    """A container whose append_rows raises ``error_type`` once it holds rows."""

    def _maybe(self, op):
        if op == self.op and self.rows:
            raise error_type()

    name = f"Raising{error_type.__name__}"
    return type(name, (Hostile,), {"op": "append_rows", "_maybe": _maybe})


CASES = {
    # factory -> whether core calls the failing operation (handle broken)
    "append_rows": (_hostile("append_rows"), True),
    "__getitem__": (_hostile("__getitem__"), True),
    "extend": (_hostile("extend"), True),
    "__len__": (_hostile("__len__"), False),  # core counts positions itself
    "__bool__": (_hostile("__bool__"), False),
    "list_subclass": (HostileList, True),
    "raising___class__": (HostileClass, False),
    "factory_raises": (lambda params: 1 / 0, True),
    "factory_returns_none": (lambda params: None, True),
    "cancelled_error": (_raising(asyncio.CancelledError), True),
    "generator_exit": (_raising(GeneratorExit), True),
    "custom_base_exception": (_raising(CustomBaseException), True),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("n", [1, 2])
def test_hostile_containers_are_request_local(registry, case, n):
    """Every way a registered container can fail during shared processing
    (append, DELTA slices, collector aggregation, list subclasses, a raising
    __class__, failing factories) stays local to its request: no exception
    leaves process_outputs, the request's handle is broken (or intact if core
    never calls the failing operation), positions are core-counted, and a
    co-batched default request is unaffected."""
    factory, breaks = CASES[case]
    _register("hostile", factory)
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    queues = {
        "a": RequestOutputCollector(RequestOutputKind.DELTA, "a"),
        "b": RequestOutputCollector(RequestOutputKind.DELTA, "b"),
    }
    hostile = SamplingParams(max_tokens=10, logprobs=2, n=n)
    set_sample_logprobs_container(hostile, "hostile")
    request = _request("a", hostile, RequestOutputKind.DELTA)
    if n == 1:
        a_ids = ["a"]
        processor.add_request(request, None, queue=queues["a"])
    else:
        parent, a_ids = ParentRequest(request), []
        for idx in range(n):
            child_id, child_params = parent.get_child_info(idx)
            child = _request(child_id, child_params, RequestOutputKind.DELTA)
            processor.add_request(child, None, parent, idx, queues["a"])
            a_ids.append(child_id)
    healthy = SamplingParams(max_tokens=10, logprobs=2)
    processor.add_request(
        _request("b", healthy, RequestOutputKind.DELTA), None, queue=queues["b"]
    )
    for step in range(3):  # queues not drained: the collector aggregates (extend)
        ids, lps, ranks = _rows(2, 3, start=step)
        processor.process_outputs(
            [_step(rid, ids, lps, ranks) for rid in a_ids + ["b"]]
        )
    for completion in queues["a"].output.outputs:
        handle = completion.logprobs
        assert type(handle) is SampleLogprobsHandle
        assert handle.broken is breaks, case
        assert len(handle) == 6  # positions the engine emitted, core-counted
    out_b = queues["b"].output.outputs[0].logprobs
    assert type(out_b) is list and len(out_b) == 6


def test_delta_output_never_returns_the_live_handle(registry):
    """Claude r17: a DELTA output must not alias the request's live handle,
    even before any position arrived (empty handle, no new tokens)."""
    from vllm.v1.engine.output_processor import RequestState

    _register("rows", RowsContainer)
    params = SamplingParams(logprobs=1)
    set_sample_logprobs_container(params, "rows")
    live = create_sample_logprobs(False, params)
    state = RequestState.__new__(RequestState)
    state.detokenizer = MagicMock()
    state.detokenizer.get_next_output_text.return_value = ""
    state.logprobs_processor = MagicMock()
    state.logprobs_processor.logprobs = live
    state.output_kind = RequestOutputKind.DELTA
    state.request_index = 0
    state.sampling_mask_chunks = []
    state.routed_experts_chunks = []
    state.spec_decode_metrics = None
    for rows in (0, 2):  # empty handle, then a handle holding positions
        if rows:
            live.append_rows(*_rows(rows, 2))
        output = state._new_completion_output([], None, None)
        assert type(output.logprobs) is SampleLogprobsHandle
        assert output.logprobs is not live and len(output.logprobs) == 0


def test_malformed_engine_rows_break_only_the_handle(registry):
    """Rows whose ranks have no length (0-d) cannot be counted: the handle
    turns broken instead of raising out of output processing."""
    _register("rows", RowsContainer)
    params = SamplingParams(logprobs=1)
    set_sample_logprobs_container(params, "rows")
    handle = create_sample_logprobs(False, params)
    ids, lps, _ = _rows(1, 2)
    handle.append_rows(ids, lps, np.int64(1))
    assert handle.broken and len(handle) == 0


class HookedError(Exception):
    """Codex r18 #1: an exception whose hooks raise while it is formatted."""

    def __getattribute__(self, name):
        if name in ("__cause__", "__context__", "__traceback__"):
            raise asyncio.CancelledError("hostile exception hook")
        return super().__getattribute__(name)


@pytest.fixture
def vllm_log_stream():
    """vLLM's formatter on a plain StreamHandler (as configured by vLLM)."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(NewLineFormatter("%(levelname)s %(message)s"))
    log = logging.getLogger(logprobs_mod.__name__)
    log.addHandler(handler)
    try:
        yield stream
    finally:
        log.removeHandler(handler)


def _rows_container_raising(error):
    class Raising(RowsContainer):
        def append_rows(self, token_ids, logprobs, ranks):
            raise error

    return Raising


def test_failure_logging_runs_no_container_hooks_unguarded(registry, vllm_log_stream):
    """Codex r18 #1: formatting a container exception's traceback may run its
    hooks; the handle is broken first and the logging itself is contained."""
    _register("hooked", _rows_container_raising(HookedError("rows")))
    params = SamplingParams(logprobs=1)
    set_sample_logprobs_container(params, "hooked")
    handle = create_sample_logprobs(False, params)
    handle.append_rows(*_rows(2, 2))  # no exception escapes
    assert handle.broken and len(handle) == 2
    assert "append_rows failed" in vllm_log_stream.getvalue()

    def factory(params):
        raise HookedError("factory")

    _register("hooked_factory", factory)
    set_sample_logprobs_container(params, "hooked_factory")
    assert create_sample_logprobs(False, params).broken
    assert "factory 'hooked_factory' failed" in vllm_log_stream.getvalue()


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit])
def test_process_level_exceptions_propagate(registry, error):
    """KeyboardInterrupt / SystemExit from a container are not contained."""
    _register("raising", _rows_container_raising(error()))
    params = SamplingParams(logprobs=1)
    set_sample_logprobs_container(params, "raising")
    handle = create_sample_logprobs(False, params)
    with pytest.raises(error):
        handle.append_rows(*_rows(1, 2))

    def factory(params):
        raise error()

    _register("raising_factory", factory)
    set_sample_logprobs_container(params, "raising_factory")
    with pytest.raises(error):
        create_sample_logprobs(False, params)


def _output(logprobs, token_ids, finished=False):
    return RequestOutput(
        request_id="r",
        prompt=None,
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=[CompletionOutput(0, "", token_ids, None, logprobs)],
        finished=finished,
    )


def test_broken_zero_position_source_is_merged(registry):
    """Codex r18 #2: a terminal DELTA with no new positions whose slice of the
    live container failed is a broken zero-position handle; aggregation must
    carry the failure (base truth-testing would drop it)."""
    _register("rows", RowsContainer)
    params = SamplingParams(logprobs=1)
    set_sample_logprobs_container(params, "rows")
    pending = create_sample_logprobs(False, params)
    pending.append_rows(*_rows(2, 2))
    first = _output(pending, [1, 2])
    first.add(_output(SampleLogprobsHandle(None, 0), [], finished=True), True)
    merged = first.outputs[0].logprobs
    assert first.finished and merged.broken and len(merged) == 2
    # Lists keep the base truth test (an empty source is not merged).
    plain = _output([{1: None}], [1])
    plain.add(_output([], [], finished=True), True)
    assert plain.outputs[0].logprobs == [{1: None}]


class HookedName(str):
    hashed = 0

    def __hash__(self):
        HookedName.hashed += 1
        return super().__hash__()


def test_names_and_selectors_must_be_exact_strings(registry):
    """Codex r18 #3: a str subclass would run its hooks under the registry lock
    (deadlock if they register); rejected before the lock, like selectors."""
    with pytest.raises(ValueError):
        _register(HookedName("sub"), RowsContainer)
    assert HookedName.hashed == 0 and "sub" not in registry
    _register("rows", RowsContainer)
    with pytest.raises(ValueError):
        set_sample_logprobs_container(SamplingParams(), HookedName("rows"))


def test_handle_cannot_be_copied_or_pickled(registry):
    """Claude r18 NIT: copy / deepcopy / pickle would run container hooks."""

    class NoCopy(RowsContainer):
        def __deepcopy__(self, memo):
            raise AssertionError("container hook ran")

        def __reduce_ex__(self, protocol):
            raise AssertionError("container hook ran")

    _register("nocopy", NoCopy)
    params = SamplingParams(logprobs=1)
    set_sample_logprobs_container(params, "nocopy")
    handle = create_sample_logprobs(False, params)
    # (pickle uses __reduce_ex__, like copy)
    for operation in (copy.copy, copy.deepcopy, lambda h: h.__reduce_ex__(4)):
        with pytest.raises(TypeError):
            operation(handle)


def test_foreign_destination_in_request_output_add(registry):
    """Codex r17 #1: a handle destination merged with a non-handle (cannot
    happen for one request) degrades instead of raising."""
    _register("rows", RowsContainer)
    params = SamplingParams(logprobs=1)
    set_sample_logprobs_container(params, "rows")
    handle = create_sample_logprobs(False, params)
    handle.append_rows(*_rows(2, 2))

    def output(logprobs):
        return RequestOutput(
            request_id="r",
            prompt=None,
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[CompletionOutput(0, "", [1, 2], None, logprobs)],
            finished=False,
        )

    first = output(handle)
    first.add(output([{1: None}, {2: None}]), aggregate=True)
    merged = first.outputs[0].logprobs
    assert type(merged) is SampleLogprobsHandle and merged.broken and len(merged) == 4


def test_handle_surface():
    handle = SampleLogprobsHandle(RowsContainer())
    handle.append_rows(*_rows(2, 2))
    assert len(handle[-1:]) == 1 and len(handle[:0]) == 0 and bool(handle)
    with pytest.raises(TypeError):
        handle[0]
    with pytest.raises(TypeError):
        list(handle)
    broken = SampleLogprobsHandle(None, 3)
    with pytest.raises(ValueError, match="unavailable"):
        broken.unwrap()
    assert "broken=True" in repr(broken) and len(broken[-2:]) == 2


class _DummyEndpointPlugin:
    name = "wanted"
    required_tasks = None


@pytest.mark.parametrize("endpoint_plugin_loaded", [True, False])
def test_vllm_plugins_exclusion_warning(monkeypatch, endpoint_plugin_loaded):
    """Claude r17: warn only when an endpoint plugin is actually loaded (so a
    plugin-absent server logs as on base), naming excluded other plugins."""
    import importlib.metadata

    from vllm import plugins

    general = [
        importlib.metadata.EntryPoint(
            name="other", value="json:dumps", group="vllm.general_plugins"
        )
    ]
    monkeypatch.setattr(
        importlib.metadata,
        "entry_points",
        lambda group: general if group == "vllm.general_plugins" else [],
    )
    monkeypatch.setattr(
        plugins,
        "load_plugins_by_group",
        lambda group: {"wanted": _DummyEndpointPlugin}
        if endpoint_plugin_loaded
        else {},
    )
    monkeypatch.setenv("VLLM_PLUGINS", "wanted" if endpoint_plugin_loaded else "")
    warnings = []
    monkeypatch.setattr(
        plugins.logger, "warning", lambda msg, *args: warnings.append(msg % args)
    )
    loaded = plugins.load_endpoint_plugins(("generate",))
    assert len(loaded) == int(endpoint_plugin_loaded)
    assert any("other" in w for w in warnings) is endpoint_plugin_loaded
