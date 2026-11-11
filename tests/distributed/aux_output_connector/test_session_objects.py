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
        writer.add_blocks("r", "s", HASHES, [_block(0), _block(4), _block(8)])
        writer.add_blocks("r", "s", HASHES, [_block(12)])
        writer.flush()
        assert store.writes == [["job/s/h1"], ["job/s/h3"]]
        rows = read_aux_rows(store, ["job/s/h1", "job/s/h3"], "routed_experts")
        assert rows[:, 0, 0].tolist() == list(range(16))

    def test_incomplete_and_discarded_objects_are_not_written(self):
        store = CountingStore()
        writer = _writer(store)
        writer.add_blocks("r", "s", HASHES, [_block(4)])
        writer.discard("r")
        writer.add_blocks("r", "s", HASHES, [_block(0)])
        writer.flush()
        assert store.writes == []

    @pytest.mark.parametrize(
        "second_session,expected_writes",
        [("s", [["job/s/h1"]]), ("t", [["job/s/h1"], ["job/t/h1"]])],
        ids=["same-session", "other-session"],
    )
    def test_each_object_is_written_once_per_session(
        self, second_session, expected_writes
    ):
        store = CountingStore()
        writer = _writer(store)
        writer.add_blocks("a", "s", HASHES, [_block(0), _block(4)])
        writer.add_blocks("b", second_session, HASHES, [_block(0), _block(4)])
        writer.flush()
        assert store.writes == expected_writes

    def test_failed_write_fails_later_calls(self):
        writer = _writer(CountingStore(fail=True))
        writer.add_blocks("r", "s", HASHES, [_block(0), _block(4)])
        with pytest.raises(AuxObjectStoreError, match="write failed"):
            writer.flush()
        with pytest.raises(AuxObjectStoreError, match="write failed"):
            writer.add_blocks("r", "s", HASHES, [_block(8)])
