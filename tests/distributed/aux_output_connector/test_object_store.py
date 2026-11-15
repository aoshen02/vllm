# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading

import numpy as np
import pytest
from safetensors.numpy import save

from tests.distributed.aux_output_connector.utils import fake_store
from vllm.distributed.aux_output_connector.object_store import read_aux_rows
from vllm.distributed.aux_output_connector.store import BlockObjectStoreError


def _rows(start, n, dtype=np.uint8):
    return (np.arange(start, start + n)[:, None, None] + np.zeros((1, 2, 3))).astype(
        dtype
    )


class TestReadAuxRows:
    def _store(self, blocks):
        _, store = fake_store()
        keys = [f"s/{i}" for i in range(len(blocks))]
        store.put_batch(keys, [save({"routed_experts": b}) for b in blocks])
        return store, keys

    @pytest.mark.parametrize(
        "num_blocks,tail_len,dtype",
        [(3, 5, np.uint8), (3, 0, np.uint16), (0, 5, np.uint8)],
        ids=["blocks-and-tail", "blocks-only", "tail-only"],
    )
    def test_concatenates_in_key_order(self, num_blocks, tail_len, dtype):
        store, keys = self._store([_rows(4 * i, 4, dtype) for i in range(num_blocks)])
        tail = _rows(4 * num_blocks, tail_len, dtype) if tail_len else None
        rows = read_aux_rows(store, keys, "routed_experts", tail)
        assert rows.dtype == dtype
        np.testing.assert_array_equal(rows, _rows(0, 4 * num_blocks + tail_len))

    def test_waits_for_object_being_written(self):
        store, keys = self._store([_rows(0, 4)])
        late = save({"routed_experts": _rows(4, 4)})
        timer = threading.Timer(0.2, store.put_batch, (["s/late"], [late]))
        timer.start()
        rows = read_aux_rows(store, keys + ["s/late"], "routed_experts", timeout_s=10)
        timer.join()
        np.testing.assert_array_equal(rows, _rows(0, 8))

    def test_missing_object_times_out(self):
        store, keys = self._store([_rows(0, 4)])
        with pytest.raises(BlockObjectStoreError, match="'s/gone' is missing"):
            read_aux_rows(store, keys + ["s/gone"], "routed_experts", timeout_s=0.05)
