# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import sys
import threading
import types
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from safetensors.numpy import save

from tests.distributed.aux_output_connector.utils import fake_store
from vllm.config import AuxOutputConfig
from vllm.distributed.aux_output_connector.object_store import (
    MooncakeAuxObjectStore,
    read_aux_rows,
)
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
        "num_blocks,dtype",
        [(3, np.uint8), (3, np.uint16), (0, np.uint8)],
        ids=["u8", "u16", "no-keys"],
    )
    def test_concatenates_in_key_order(self, num_blocks, dtype):
        store, keys = self._store([_rows(4 * i, 4, dtype) for i in range(num_blocks)])
        rows = read_aux_rows(store, keys, "routed_experts")
        assert len(rows) == 4 * num_blocks
        if num_blocks:
            assert rows.dtype == dtype
            np.testing.assert_array_equal(rows, _rows(0, 12))

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

    def test_missing_field_is_named(self):
        store, keys = self._store([_rows(0, 4), _rows(4, 4)])
        with pytest.raises(BlockObjectStoreError, match="'s/0' has no field 'hidden"):
            read_aux_rows(store, keys, "hidden_states")


@pytest.fixture
def mooncake_masters(tmp_path, monkeypatch):
    """Config files of a KV and an object master; `MOONCAKE_CONFIG_PATH` names
    the KV one. Returns them and the masters that clients connect to."""
    paths = {}
    for name in ("kv", "objects"):
        path = tmp_path / f"{name}.json"
        path.write_text(
            json.dumps(
                {
                    "master_server_address": f"{name}-master:50051",
                    "mode": "standalone-store",
                    "global_segment_size": 0,
                }
            )
        )
        paths[name] = str(path)
    monkeypatch.setenv("MOONCAKE_CONFIG_PATH", paths["kv"])
    store_module = types.ModuleType("mooncake.store")
    store_module.MooncakeDistributedStore = Mock
    store_module.ReplicateConfig = SimpleNamespace
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", store_module)
    connected = []
    monkeypatch.setattr(
        "vllm.distributed.mooncake_store.setup_mooncake_store",
        lambda store, config, hostname: connected.append(config.master_server_address),
    )
    return paths, connected


@pytest.mark.parametrize("name", [None, "objects"], ids=["environment", "argument"])
def test_store_connects_to_the_master_of_its_config(mooncake_masters, name):
    paths, connected = mooncake_masters
    MooncakeAuxObjectStore.create(None if name is None else paths[name])
    assert connected == [f"{name or 'kv'}-master:50051"]


def test_engine_writes_objects_to_the_configured_master(mooncake_masters, monkeypatch):
    import vllm.distributed.aux_output_connector.connector as aux_connector
    import vllm.distributed.aux_output_connector.worker as aux_worker

    paths, connected = mooncake_masters
    group = SimpleNamespace(
        is_first_rank=True, is_last_rank=True, world_size=1, rank_in_group=0
    )
    monkeypatch.setattr(aux_worker, "get_tp_group", lambda: group)
    monkeypatch.setattr(aux_worker, "get_pp_group", lambda: group)
    monkeypatch.setattr(
        aux_connector, "resolve_kv_cache_block_sizes", lambda *_: (16, 16)
    )
    config = SimpleNamespace(
        aux_output_config=AuxOutputConfig(
            object_key_prefix="job",
            enable_return_hidden_states=True,
            mooncake_config_path=paths["objects"],
        ),
        kv_transfer_config=None,
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=16),
        max_concurrent_batches=2,
    )
    connector = aux_worker.AuxOutputWorkerConnector(vllm_config=config, model=None)
    connector.initialize_output_storage(SimpleNamespace(num_blocks=4))
    connector.close()
    assert connected == ["objects-master:50051"]
