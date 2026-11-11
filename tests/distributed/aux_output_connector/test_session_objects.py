# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.distributed.aux_output_connector.object_store import (
    AuxObjectStoreError,
    LocalAuxObjectStore,
)
from vllm.distributed.aux_output_connector.objects import read_aux_rows
from vllm.distributed.aux_output_connector.session_objects import (
    SessionObjectWriter,
)

HASHES = [f"h{i}" for i in range(8)]
VERSIONS = [(0, "v")]


def _block(start, size=4):
    return start, np.arange(start, start + size, dtype=np.uint8).reshape(-1, 1, 1)


class CountingStore(LocalAuxObjectStore):
    def __init__(self, fail=False):
        super().__init__()
        self.fail = fail
        self.writes = []

    def put_batch(self, keys, values):
        self.writes.append(list(keys))
        if self.fail:
            raise AuxObjectStoreError("pool is full")
        super().put_batch(keys, values)


def _writer(store):
    return SessionObjectWriter(
        store,
        key_prefix="job",
        object_size=8,
        hash_block_size=4,
        max_pending_batches=4,
    )


class TestSessionObjectWriter:
    def test_writes_complete_objects(self):
        store = CountingStore()
        writer = _writer(store)
        writer.add_blocks("r", "s", HASHES, [_block(0), _block(4), _block(8)], VERSIONS)
        writer.add_blocks("r", "s", HASHES, [_block(12)], VERSIONS)
        writer.flush()
        assert store.writes == [["job/s/h1@v"], ["job/s/h3@v"]]
        rows = read_aux_rows(store, ["job/s/h1@v", "job/s/h3@v"], "routed_experts")
        assert rows[:, 0, 0].tolist() == list(range(16))

    def test_incomplete_and_discarded_objects_are_not_written(self):
        store = CountingStore()
        writer = _writer(store)
        writer.add_blocks("r", "s", HASHES, [_block(4)], VERSIONS)
        writer.discard("r")
        writer.add_blocks("r", "s", HASHES, [_block(0)], VERSIONS)
        writer.flush()
        assert store.writes == []

    @pytest.mark.parametrize(
        "second_session,expected_writes",
        [("s", [["job/s/h1@v"]]), ("t", [["job/s/h1@v"], ["job/t/h1@v"]])],
        ids=["same-session", "other-session"],
    )
    def test_each_object_is_written_once_per_session(
        self, second_session, expected_writes
    ):
        store = CountingStore()
        writer = _writer(store)
        writer.add_blocks("a", "s", HASHES, [_block(0), _block(4)], VERSIONS)
        writer.add_blocks("b", second_session, HASHES, [_block(0), _block(4)], VERSIONS)
        writer.flush()
        assert store.writes == expected_writes

    def test_failed_write_fails_later_calls(self):
        writer = _writer(CountingStore(fail=True))
        writer.add_blocks("r", "s", HASHES, [_block(0), _block(4)], VERSIONS)
        with pytest.raises(AuxObjectStoreError, match="write failed"):
            writer.flush()
        with pytest.raises(AuxObjectStoreError, match="write failed"):
            writer.add_blocks("r", "s", HASHES, [_block(8)], VERSIONS)

    def test_forgotten_session_writes_again(self):
        store = CountingStore()
        writer = _writer(store)
        writer.add_blocks("a", "s", HASHES, [_block(0), _block(4)], VERSIONS)
        writer.forget_session("s")
        writer.add_blocks("b", "s", HASHES, [_block(0), _block(4)], VERSIONS)
        writer.flush()
        assert store.writes == [["job/s/h1@v"], ["job/s/h1@v"]]

    def test_objects_are_named_by_the_weight_version_of_their_last_row(self):
        store = CountingStore()
        writer = _writer(store)
        blocks = [_block(0), _block(4), _block(8), _block(12)]
        writer.add_blocks("a", "s", HASHES, blocks, [(0, "v1"), (6, "v2"), (12, "v3")])
        writer.add_blocks("b", "s", HASHES, blocks[:2], [(0, "v2")])
        writer.add_blocks("c", "s", HASHES, blocks[:2], [(0, "v4")])
        writer.flush()
        assert store.writes == [["job/s/h1@v2", "job/s/h3@v3"], ["job/s/h1@v4"]]

    def test_objects_carry_staged_rows_of_other_fields(self):
        store = CountingStore()
        writer = _writer(store)
        rows = np.arange(24, dtype=np.int32).reshape(12, 1, 2)
        writer.add_rows("r", "indexer_topk", 0, rows[:5])
        writer.add_rows("r", "indexer_topk", 5, rows[5:])
        start, staged = writer.read_rows("r", "indexer_topk", 3, 12)
        assert start == 3 and staged.tolist() == rows[3:].tolist()
        writer.add_blocks("r", "s", HASHES, [_block(0), _block(4)], VERSIONS)
        writer.flush()
        stored = read_aux_rows(store, ["job/s/h1@v"], "indexer_topk")
        assert stored.tolist() == rows[:8].tolist()
        start, staged = writer.read_rows("r", "indexer_topk", 3, 12)
        assert start == 8 and staged.tolist() == rows[8:].tolist()

    def test_object_with_incomplete_field_rows_is_rejected(self):
        writer = _writer(CountingStore())
        writer.add_rows("r", "indexer_topk", 0, np.zeros((5, 1, 2), dtype=np.int32))
        with pytest.raises(AssertionError, match="incomplete"):
            writer.add_blocks("r", "s", HASHES, [_block(0), _block(4)], VERSIONS)

    def test_reading_rows_that_were_not_staged_is_rejected(self):
        writer = _writer(CountingStore())
        writer.add_rows("r", "indexer_topk", 0, np.zeros((3, 1, 2), dtype=np.int32))
        writer.add_rows("r", "indexer_topk", 5, np.zeros((7, 1, 2), dtype=np.int32))
        with pytest.raises(AssertionError, match="were not staged"):
            writer.read_rows("r", "indexer_topk", 0, 12)
