# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
import threading
import time
import types
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import regex as re
from safetensors.numpy import save

from tests.distributed.aux_output_connector.utils import (
    DictObjectStore,
    FakeMooncake,
    fake_store,
    wait_for_writer,
)
from vllm.distributed.aux_output_connector import object_store
from vllm.distributed.aux_output_connector.object_store import (
    AuxObjectCorrupted,
    AuxObjectWriter,
    AuxRowsMisaligned,
    MooncakeAuxObjectStore,
    aux_namespace_prefix,
    aux_prefix,
    create_aux_object_store,
    read_aux_rows,
    read_aux_spans,
    verify_aux_rows,
)
from vllm.distributed.aux_output_connector.store import BlockObjectStoreError

pytestmark = pytest.mark.cpu_test


def _rows(start, n, dtype=np.uint8):
    return (np.arange(start, start + n)[:, None, None] + np.zeros((1, 2, 3))).astype(
        dtype
    )


def _stored(blocks):
    _, store = fake_store()
    keys = [f"job/g/{i}" for i in range(len(blocks))]
    store.put_batch(keys, [save({"routed_experts": block}) for block in blocks])
    return store, keys


@pytest.mark.parametrize(
    "num_objects,dtype",
    [(3, np.uint8), (3, np.uint16), (0, np.uint8)],
    ids=["uint8", "uint16", "no-keys"],
)
def test_read_aux_rows_concatenates_objects_in_key_order(num_objects, dtype):
    store, keys = _stored([_rows(4 * i, 4, dtype) for i in range(num_objects)])
    rows = read_aux_rows(store, keys[::-1], "routed_experts")
    assert len(rows) == 4 * num_objects
    if num_objects:
        assert rows.dtype == dtype
        np.testing.assert_array_equal(
            rows, np.concatenate([_rows(8, 4), _rows(4, 4), _rows(0, 4)])
        )


def test_read_aux_rows_waits_for_an_object_through_a_write_stall(monkeypatch):
    store, keys = _stored([_rows(0, 4)])
    late = save({"routed_experts": _rows(4, 4)})
    now = [0.0]

    def sleep(seconds):
        now[0] += seconds
        if now[0] >= 599.0 and store.get("job/g/late") is None:
            store.put_batch(["job/g/late"], [late])

    monkeypatch.setattr(
        object_store, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep)
    )
    rows = read_aux_rows(store, [*keys, "job/g/late"], "routed_experts")
    np.testing.assert_array_equal(rows, _rows(0, 8))
    assert now[0] >= 599.0


def test_read_aux_rows_raises_for_a_missing_object():
    store, keys = _stored([_rows(0, 4)])
    with pytest.raises(BlockObjectStoreError, match="'job/g/gone' is missing"):
        read_aux_rows(store, [*keys, "job/g/gone"], "routed_experts", timeout_s=0.05)


@pytest.fixture
def mooncake(tmp_path, monkeypatch):
    paths = {}
    for name, mode, segment_size in (
        ("kv", "standalone-store", 0),
        ("objects", "standalone-store", 0),
        ("embedded", "embedded", 1 << 20),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "master_server_address": f"{name}-master:50051",
                    "mode": mode,
                    "global_segment_size": segment_size,
                }
            )
        )
        paths[name] = str(path)
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", paths["kv"])
    clients = []

    def make_client():
        clients.append(FakeMooncake())
        return clients[-1]

    store_module = types.ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = make_client  # type: ignore[attr-defined]
    store_module.ReplicateConfig = SimpleNamespace  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    connected = []
    monkeypatch.setattr(
        "vllm.distributed.mooncake_store.setup_mooncake_store",
        lambda store, config, hostname: connected.append(config.master_server_address),
    )
    return paths, connected, clients


@pytest.mark.parametrize("name", [None, "objects"], ids=["environment", "argument"])
def test_store_connects_to_the_master_of_its_config(mooncake, name):
    paths, connected, _ = mooncake
    MooncakeAuxObjectStore.create(None if name is None else paths[name])
    assert connected == [f"{name or 'kv'}-master:50051"]


@pytest.mark.parametrize("hard_pin", [None, True], ids=["default", "hard-pin"])
def test_store_writes_objects_pinned_as_configured(mooncake, hard_pin):
    _, _, clients = mooncake
    store = MooncakeAuxObjectStore.create(
        **({} if hard_pin is None else {"hard_pin": hard_pin})
    )
    store.put_batch(["job/g/0"], [b"rows"])
    assert [config.with_hard_pin for config in clients[0].configs] == [bool(hard_pin)]
    assert store.get("job/g/0") == b"rows"


def test_store_refuses_a_mode_other_than_standalone_store(mooncake):
    paths, connected, clients = mooncake
    with pytest.raises(ValueError, match="standalone-store"):
        MooncakeAuxObjectStore.create(paths["embedded"])
    assert connected == [] and clients == []


def test_store_is_created_by_name_with_its_extra_config(mooncake):
    paths, connected, _ = mooncake
    store = create_aux_object_store("mooncake", {"config_path": paths["objects"]})
    assert isinstance(store, MooncakeAuxObjectStore)
    assert connected == ["objects-master:50051"]
    other = create_aux_object_store(
        f"{DictObjectStore.__module__}.DictObjectStore", {"bucket": "b"}
    )
    assert isinstance(other, DictObjectStore) and other.extra_config == {"bucket": "b"}
    for name in ("Mooncake", ".Store", "pkg."):
        with pytest.raises(ValueError, match=f"object_store {name!r}"):
            create_aux_object_store(name, {})


def test_objects_round_trip_through_a_store_other_than_mooncake():
    store = DictObjectStore()
    writer = AuxObjectWriter(store, timeout_s=5)
    rows = _rows(0, 4)
    writer.put(
        [("job/g/0", {"routed_experts": rows}, {}), ("job/h/0", {"x": rows}, {})]
    )
    wait_for_writer(writer)
    np.testing.assert_array_equal(
        read_aux_rows(store, ["job/g/0"], "routed_experts"), rows
    )
    store.remove_prefix(aux_namespace_prefix("job", "g"))
    assert list(store.objects) == ["job/h/0"]
    writer.close()


_ECMASCRIPT_SYNTAX_CHARACTERS = set("^$\\.*+?()[]{}|")


def _released(prefix, keys, kind="mooncake"):
    if kind == "dict":
        store = client = DictObjectStore()
    else:
        client, store = fake_store()
    store.put_batch(keys, [b"rows"] * len(keys))
    store.remove_prefix(prefix)
    for pattern in getattr(client, "patterns", []):
        assert set(re.findall(r"\\(.)", pattern)) <= _ECMASCRIPT_SYNTAX_CHARACTERS
    return [key for key in keys if key not in client.objects]


@pytest.mark.parametrize(
    "prefix,namespace",
    [
        ("job", "g"),
        ("run-1", "g_1/s-0"),
        ("p+q*r?", "g"),
        ("(p)[q]{r}", "g"),
        ("p|q^r$", "g"),
        ("p#1 &~", "g"),
        ("p\\q", "g"),
    ],
    ids=[
        "plain",
        "nested",
        "quantifiers",
        "groups",
        "anchors",
        "punctuation",
        "backslash",
    ],
)
@pytest.mark.parametrize("kind", ["mooncake", "dict"])
def test_release_prefixes_remove_exactly_their_keys(prefix, namespace, kind):
    prefixes = [aux_prefix(prefix), aux_namespace_prefix(prefix, namespace)]
    keys = [
        f"{prefix}/{namespace}/ab",
        f"{prefix}/{namespace}/tail-cd",
        f"{prefix}/{namespace}x/ab",
        f"{prefix}x/{namespace}/ab",
        f"x{prefix}/{namespace}/ab",
    ]
    assert [_released(p, keys, kind) for p in prefixes] == [keys[:3], keys[:2]]


def test_release_prefixes_match_dots_literally():
    keys = ["a.b/g.0/ab", "axb/g.0/ab", "a.b/gx0/ab"]
    assert _released(aux_prefix("a.b"), keys) == [keys[0], keys[2]]
    assert _released(aux_namespace_prefix("a.b", "g.0"), keys) == keys[:1]


@pytest.mark.parametrize(
    "namespace,released",
    [("g", [0, 1, 2, 3, 4]), ("g/0", [1, 2, 3])],
    ids=["parent", "nested"],
)
def test_namespace_prefix_covers_the_namespaces_nested_in_it(namespace, released):
    keys = ["p/g/ab", "p/g/0/ab", "p/g/0/tail-cd", "p/g/0/1/ab", "p/g/01/ab", "p/g0/ab"]
    assert _released(aux_namespace_prefix("p", namespace), keys) == [
        keys[i] for i in released
    ]


def _objects(*keys):
    return [(key, {"routed_experts": np.zeros((1, 1), np.uint8)}, {}) for key in keys]


def _writer(put, timeout_s=10.0):
    client, store = fake_store()
    put_batch = client.put_batch
    client.put_batch = lambda keys, values, config: (
        put(keys) or put_batch(keys, values, config)
    )
    return client, store, AuxObjectWriter(store, timeout_s=timeout_s)


def test_writer_writes_a_batch_a_store_refused_once():
    refusals: list[int] = []

    def put(keys):
        if refusals:
            return None
        refusals.append(1)
        return -200

    client, _, writer = _writer(put)
    start = time.monotonic()
    writer.put(_objects("g/0"))
    writer.put(_objects("h/0"))
    # Queuing never blocks.
    assert time.monotonic() - start < 0.1
    wait_for_writer(writer)
    assert list(client.objects) == ["g/0", "h/0"] and refusals == [1]
    writer.close()
    writer._thread.join(10)
    assert client.closed


@pytest.mark.parametrize(
    "failure,reason",
    [("refused", BlockObjectStoreError), ("raises", RuntimeError)],
    ids=["refused-to-deadline", "put-raises"],
)
def test_writer_drops_only_the_batch_it_cannot_write(monkeypatch, failure, reason):
    logger = Mock()
    monkeypatch.setattr(object_store, "logger", logger)
    refusals = []
    queued = threading.Event()

    def put(keys):
        if keys != ["g/0"]:
            return None
        if failure == "raises":
            queued.wait(5)
            raise RuntimeError("put failed")
        refusals.append(time.monotonic())
        return -200

    client, store, writer = _writer(put, 0.5)
    writer.put(_objects("g/0"))
    time.sleep(0.3)
    # Queued behind the failing batch, with time left to be written.
    writer.put(_objects("g/1", "h/0"))
    queued.set()
    wait_for_writer(writer)
    # Its keys are missing; later objects of its namespace are written.
    assert list(client.objects) == ["g/1", "h/0"]
    assert client.writes[-1] == ["g/1", "h/0"]
    assert len(read_aux_rows(store, ["g/1"], "routed_experts", timeout_s=0)) == 1
    ((_, _, key, logged), _) = logger.error.call_args
    assert key == "g/0" and isinstance(logged, reason)
    assert logger.error.call_count == 1
    if refusals:
        # Refusals are retried until the deadline, the last one less than a
        # retry interval before it.
        assert refusals[-1] - refusals[0] >= 0.35
    # A batch queued after the failure is written as well.
    writer.put(_objects("g/2", "h/1"))
    wait_for_writer(writer)
    assert list(client.objects) == ["g/1", "h/0", "g/2", "h/1"]
    assert logger.error.call_count == 1
    writer.close()
    writer._thread.join(10)


def test_writer_deadline_counts_from_queueing():
    # A batch queued behind a slow one keeps the deadline it was queued with.
    refusals: list[float] = []

    def put(keys):
        if keys == ["g/0"]:
            time.sleep(0.4)
            return None
        refusals.append(time.monotonic())
        return -200

    client, _, writer = _writer(put, 0.5)
    writer.put(_objects("g/0"))
    time.sleep(0.1)
    queued = time.monotonic()
    writer.put(_objects("h/0"))
    wait_for_writer(writer)
    assert list(client.objects) == ["g/0"]
    assert 0.25 <= refusals[-1] - queued < 0.5
    writer.close()


def test_writer_close_writes_the_queued_batches():
    client, _, writer = _writer(lambda keys: time.sleep(0.1))
    writer.put(_objects("g/0", "h/0"))
    writer.put(_objects("k/0"))
    writer.close()
    assert list(client.objects) == ["g/0", "h/0", "k/0"] and client.closed
    # One store call per batch.
    assert client.writes == [["g/0", "h/0"], ["k/0"]]


def test_writer_drops_the_batches_queued_behind_a_put_that_does_not_return(
    monkeypatch,
):
    logger = Mock()
    monkeypatch.setattr(object_store, "logger", logger)
    release = threading.Event()
    calls = []

    def put(keys):
        calls.append(keys)
        return release.wait(10) and None

    client, _, writer = _writer(put, 0.5)
    start = time.monotonic()
    writer.put(_objects("g/0"))
    writer.put(_objects("h/0"))
    writer.put(_objects("h/1"))
    time.sleep(0.55)
    # Past their deadline the batches are dropped, also those queued behind
    # the put that does not return, and the writer no longer holds their
    # objects. While the put has not returned, queuing drops without waiting.
    writer.put(_objects("k/0"))
    assert time.monotonic() - start < 0.65
    assert [batch.objects for batch in writer._pending] == [None] * 3
    assert logger.error.call_count == 4
    # Every deadline has passed: closing does not wait for the put.
    writer.close()
    assert time.monotonic() - start < 0.8
    # The writer thread closes the store once the put returns, and writes
    # none of the dropped batches.
    assert not client.closed
    release.set()
    writer._thread.join(10)
    assert client.closed
    assert calls == [["g/0"]]


@pytest.mark.parametrize(
    "stuck,waited,written",
    [(False, (0.05, 0.4), ["g/0", "h/0", "k/0"]), (True, (0.1, 0.35), ["g/0"])],
    ids=["write-ends", "store-stuck"],
)
def test_writer_blocks_a_full_queue_until_a_write_ends_or_misses_its_deadline(
    monkeypatch, stuck, waited, written
):
    monkeypatch.setattr(object_store, "_MAX_QUEUED_BATCHES", 1)
    release = threading.Event()
    client, _, writer = _writer(
        lambda keys: (release.wait(10) and None) if keys == ["g/0"] else None, 0.5
    )
    writer.put(_objects("g/0"))
    if not stuck:
        threading.Timer(0.1, release.set).start()
    else:
        # The wait ends at the deadline of the head, not one timeout later.
        time.sleep(0.3)
    start = time.monotonic()
    writer.put(_objects("h/0"))
    assert waited[0] <= time.monotonic() - start < waited[1]
    # Behind a stuck store the next put does not wait another deadline.
    start = time.monotonic()
    writer.put(_objects("k/0"))
    assert time.monotonic() - start < (0.05 if stuck else 0.6)
    # Behind a stuck store only the put that did not return lands, late.
    release.set()
    wait_for_writer(writer)
    writer.close()
    writer._thread.join(10)
    assert list(client.objects) == written


_SPAN = 4
# Two samples of a group: a shared 6-token prompt, then their own outputs.
_TRAJECTORIES = [
    [7, 3, 9, 1, 4, 4, *range(10, 20)],
    [7, 3, 9, 1, 4, 4, *range(20, 30)],
]


def _written(trajectories=_TRAJECTORIES):
    """Objects of `_SPAN` positions over each trajectory, as an engine writes
    them, and the listing of each."""
    store = DictObjectStore()
    writer = AuxObjectWriter(store, timeout_s=5)
    listings = []
    for sample, tokens in enumerate(trajectories):
        objects, listing = [], []
        for start in range(0, len(tokens) - 1, _SPAN):
            end = min(start + _SPAN, len(tokens) - 1)
            key = f"job/g/{sample}-{start}"
            rows = _rows(100 * sample + start, end - start)
            token_ids = np.array(tokens[start:end], np.int32)
            fields = {"routed_experts": rows, "token_ids": token_ids}
            objects.append((key, fields, {"start": str(start)}))
            listing.append((key, start, end))
        writer.put(objects)
        listings.append(listing)
    wait_for_writer(writer)
    writer.close()
    return store, listings


def _packed(store, listings, trajectories=_TRAJECTORIES):
    """A trainer's packed batch: input ids, positions, the binding of each
    row (one placeholder for each last token, which has no row), the replay
    mask, and sequence starts."""
    inputs, positions, row_tokens, row_positions, mask = [], [], [], [], []
    for tokens, listing in zip(trajectories, listings, strict=True):
        _, row_token, row_position = read_aux_spans(store, listing)
        inputs.append(np.array(tokens))
        positions.append(np.arange(len(tokens)))
        row_tokens.append(np.append(row_token, 0))
        row_positions.append(np.append(row_position, 0))
        mask.append(np.arange(len(tokens)) < len(tokens) - 1)
    lengths = [len(tokens) for tokens in trajectories]
    return (
        np.concatenate(inputs),
        np.concatenate(positions),
        np.concatenate(row_tokens),
        np.concatenate(row_positions),
        np.concatenate(mask),
        np.cumsum([0, *lengths]),
    )


def _zigzag(array, cu_seqlens, rank, cp):
    """Rank `rank`'s part of each sequence under zigzag context parallelism:
    chunks `rank` and `2 * cp - 1 - rank` of `2 * cp`."""
    parts = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        chunks = np.array_split(array[start:end], 2 * cp)
        parts += [chunks[rank], chunks[2 * cp - 1 - rank]]
    return np.concatenate(parts)


def _contiguous(array, cu_seqlens, rank, cp):
    """The bug: a contiguous split where the inputs use zigzag."""
    parts = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        chunks = np.array_split(array[start:end], 2 * cp)
        parts += [chunks[2 * rank], chunks[2 * rank + 1]]
    return np.concatenate(parts)


def _shift_one(binding, cu_seqlens):
    """The bug: each sample's rows placed one position early."""
    shifted = binding.copy()
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        shifted[start : end - 1] = binding[start + 1 : end]
    return shifted


def _swap_samples(binding, cu_seqlens):
    """The bug: the rows of two samples of equal length swapped."""
    (a, b, c) = cu_seqlens
    return np.concatenate([binding[b:c], binding[a:b]])


def _swap_repeated_tokens(binding, cu_seqlens):
    """The bug: the rows of positions 4 and 5, both token 4, swapped; only
    positions tell them apart."""
    swapped = binding.copy()
    for start in cu_seqlens[:-1]:
        swapped[[start + 4, start + 5]] = binding[[start + 5, start + 4]]
    return swapped


class TestVerifyAuxRows:
    def test_aligned_rows_pass_packing_and_context_parallelism(self):
        inputs, positions, row_tokens, row_positions, mask, cu = _packed(*_written())
        verify_aux_rows(inputs, positions, row_tokens, row_positions, mask=mask)
        for rank in range(2):
            local = [_zigzag(a, cu, rank, 2) for a in (inputs, positions)]
            binding = [_zigzag(a, cu, rank, 2) for a in (row_tokens, row_positions)]
            verify_aux_rows(*local, *binding, mask=_zigzag(mask, cu, rank, 2))

    @pytest.mark.parametrize(
        "misplace,index,position,offset",
        [
            (_shift_one, 0, 0, 1),
            (_swap_samples, 6, 6, 0),
            (_swap_repeated_tokens, 4, 4, 1),
        ],
        ids=["shifted-one-position", "swapped-samples", "swapped-repeated-tokens"],
    )
    def test_misplaced_rows_are_reported(self, misplace, index, position, offset):
        inputs, positions, row_tokens, row_positions, mask, cu = _packed(*_written())
        with pytest.raises(AuxRowsMisaligned) as error:
            verify_aux_rows(
                inputs,
                positions,
                misplace(row_tokens, cu),
                misplace(row_positions, cu),
                mask=mask,
            )
        found = (error.value.index, error.value.position, error.value.offset)
        assert found == (index, position, offset)

    def test_rows_sharded_unlike_the_inputs_are_reported_on_their_rank(self):
        # Inputs are sharded zigzag, the rows contiguously: rank 0's second
        # chunk holds positions 4-7 where its tokens are positions 12-15.
        inputs, positions, row_tokens, row_positions, mask, cu = _packed(*_written())
        local = [_zigzag(a, cu, 0, 2) for a in (inputs, positions, mask)]
        binding = [_contiguous(a, cu, 0, 2) for a in (row_tokens, row_positions)]
        with pytest.raises(AuxRowsMisaligned) as error:
            verify_aux_rows(local[0], local[1], *binding, mask=local[2])
        assert (error.value.position, error.value.offset) == (12, -8)

    def test_rows_missing_from_a_replayed_position_are_reported(self):
        # The trainer fills a span it did not read with zeros, as it pads.
        inputs, positions, row_tokens, row_positions, mask, cu = _packed(*_written())
        row_tokens[4:8] = row_positions[4:8] = 0
        with pytest.raises(AuxRowsMisaligned) as error:
            verify_aux_rows(inputs, positions, row_tokens, row_positions, mask=mask)
        assert (error.value.position, error.value.offset) == (4, -4)

    def test_rows_of_other_tokens_at_their_position_are_reported(self):
        # A trajectory retokenized after the engine returned it.
        store, listings = _written()
        _, row_tokens, row_positions = read_aux_spans(store, listings[0])
        tokens = np.array(_TRAJECTORIES[0][:-1])
        tokens[9] += 1
        match = r"computed for token 13 at position 9 \(offset 0\), not token 14"
        with pytest.raises(AuxRowsMisaligned, match=match):
            verify_aux_rows(tokens, row_positions, row_tokens, row_positions)

    def test_a_mask_not_one_per_row_is_refused(self):
        # Broadcast, one False would skip every row.
        with pytest.raises(ValueError, match="a mask of 1"):
            verify_aux_rows([1, 2], [0, 1], [9, 9], [8, 8], mask=[False])


def test_strided_fields_read_back_as_written():
    rows = np.arange(48, dtype=np.uint8).reshape(4, 2, 6)[:, :, ::2]
    store = DictObjectStore()
    writer = AuxObjectWriter(store, timeout_s=5)
    token_ids = np.arange(4, dtype=np.int32)
    fields = {"routed_experts": rows, "token_ids": token_ids}
    writer.put([("job/g/0", fields, {"start": "0"})])
    wait_for_writer(writer)
    writer.close()
    np.testing.assert_array_equal(
        read_aux_rows(store, ["job/g/0"], "routed_experts"), rows
    )


def test_read_aux_spans_reads_a_spliced_listing_from_object_starts():
    # Turn 2 holds positions [0, 6) from turn 1 and takes [6, 8) from the
    # object that starts at 4.
    store, (listing, _) = _written()
    spliced = [listing[0], ("job/g/0-4", 4, 6), ("job/g/0-4", 6, 8), listing[2]]
    rows, row_tokens, row_positions = read_aux_spans(store, spliced)
    np.testing.assert_array_equal(rows, _rows(0, 12))
    assert row_tokens.tolist() == _TRAJECTORIES[0][:12]
    assert row_positions.tolist() == list(range(12))


@pytest.mark.parametrize(
    "spans,match",
    [
        ([("job/g/0-4", 3, 6)], r"\[3, 6\) are not in .*holds \[4, 8\)"),
        ([("job/g/0-4", 6, 9)], r"\[6, 9\) are not in .*holds \[4, 8\)"),
        ([("job/g/0-0", 0, 4), ("job/g/0-8", 8, 12)], "starts at 8, not at .* 4"),
    ],
    ids=["before-object", "past-object", "gap"],
)
def test_read_aux_spans_refuses_positions_its_objects_do_not_hold(spans, match):
    store, _ = _written()
    with pytest.raises(ValueError, match=match):
        read_aux_spans(store, spans)


def _corrupt(data: bytes, how: str) -> bytes:
    if how == "payload":
        return data[:-1] + bytes([data[-1] ^ 1])
    # Same length, so the header still parses: rows reshaped, or every
    # position shifted.
    old, new = {"shape": (b"[4,2,3]", b"[2,4,3]"), "start": (b'"4"', b'"5"')}[how]
    assert data.count(old) == 1
    return data.replace(old, new)


@pytest.mark.parametrize("how", ["payload", "shape", "start"])
def test_corrupted_objects_are_refused_on_read(how):
    store, (listing, _) = _written()
    store.objects["job/g/0-4"] = _corrupt(store.objects["job/g/0-4"], how)
    with pytest.raises(AuxObjectCorrupted, match="'job/g/0-4' does not match"):
        read_aux_rows(store, [key for key, _, _ in listing], "routed_experts")
    with pytest.raises(AuxObjectCorrupted, match="'job/g/0-4' does not match"):
        read_aux_spans(store, listing)
    store.objects["job/g/0-4"] = b"\xff" * 16
    with pytest.raises(AuxObjectCorrupted, match="'job/g/0-4' does not parse"):
        read_aux_rows(store, ["job/g/0-4"], "routed_experts")


def test_objects_without_a_checksum_read_as_rows_only():
    store, keys = _stored([_rows(0, 4)])
    np.testing.assert_array_equal(
        read_aux_rows(store, keys, "routed_experts"), _rows(0, 4)
    )
    with pytest.raises(AuxObjectCorrupted, match="has no token ids or checksum"):
        read_aux_spans(store, [(keys[0], 0, 4)])


# One object of 6 positions per kind of auxiliary output: the binding and the
# checksum do not depend on the arrays an object holds.
_ARTIFACTS = {
    "routed-experts": {"routed_experts": ((6, 3, 2), np.uint8)},
    "indexer-top-k": {"indexer_topk": ((6, 2, 8), np.int16)},
    # bfloat16 rows travel as their uint16 bits.
    "hidden-states": {"hidden_states": ((6, 16), np.uint16)},
    "top-k-logprobs": {
        "topk_logprobs": ((6, 4), np.float32),
        "topk_token_ids": ((6, 4), np.int32),
    },
    "multimodal-embedding": {"mm_embedding": ((6, 16), np.float16)},
}


@pytest.mark.parametrize("artifact", list(_ARTIFACTS), ids=list(_ARTIFACTS))
def test_every_artifact_is_bound_and_checksummed(artifact):
    rng = np.random.default_rng(0)
    fields = {
        name: rng.integers(0, 100, shape).astype(dtype)
        for name, (shape, dtype) in _ARTIFACTS[artifact].items()
    }
    tokens = np.array([5, 9, 9, 2, 7, 3, 8], np.int32)
    store = DictObjectStore()
    writer = AuxObjectWriter(store, timeout_s=5)
    writer.put([("job/g/0", {**fields, "token_ids": tokens[:6]}, {"start": "0"})])
    wait_for_writer(writer)
    writer.close()
    for name, array in fields.items():
        spans = [("job/g/0", 0, 6)]
        rows, row_tokens, row_positions = read_aux_spans(store, spans, name)
        np.testing.assert_array_equal(rows, array)
        verify_aux_rows(tokens[:6], np.arange(6), row_tokens, row_positions)
        with pytest.raises(AuxRowsMisaligned):
            verify_aux_rows(tokens[1:], np.arange(6), row_tokens, row_positions)
    store.objects["job/g/0"] = _corrupt(store.objects["job/g/0"], "payload")
    with pytest.raises(AuxObjectCorrupted, match="does not match"):
        read_aux_spans(store, [("job/g/0", 0, 6)], next(iter(fields)))
