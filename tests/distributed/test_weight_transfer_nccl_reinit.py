# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for idempotent communicator teardown on NCCL re-init."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer import nccl_engine, sparse_nccl_engine
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
)
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
    SparseNCCLWeightTransferEngine,
)


def _mock_vllm_config() -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
        model_config=SimpleNamespace(),
    )


def _init_info(port: int) -> NCCLWeightTransferInitInfo:
    return NCCLWeightTransferInitInfo(
        master_address="127.0.0.1",
        master_port=port,
        rank_offset=1,
        world_size=2,
    )


def test_nccl_reinit_destroys_previous_communicator(monkeypatch: pytest.MonkeyPatch):
    comm1 = MagicMock(name="comm1")
    comm2 = MagicMock(name="comm2")
    comms = iter([comm1, comm2])
    monkeypatch.setattr(
        nccl_engine, "worker_init_process_group", lambda *_a, **_kw: next(comms)
    )

    engine = NCCLWeightTransferEngine(
        WeightTransferConfig(backend="nccl"),
        _mock_vllm_config(),  # type: ignore[arg-type]
        torch.device("cpu"),
        MagicMock(spec=torch.nn.Module),
    )

    engine.init_transfer_engine(_init_info(29500))
    assert engine.model_update_group is comm1
    comm1.destroy.assert_not_called()

    # Populate cached state to verify shutdown() clears it as well.
    engine._last_update_info["default"] = MagicMock()
    engine._plan = MagicMock()
    engine._plan_metadata = MagicMock()
    engine._plan_manifest = MagicMock()

    # Re-initializing must destroy comm1 before creating comm2.
    engine.init_transfer_engine(_init_info(29501))
    comm1.destroy.assert_called_once()
    assert engine.model_update_group is comm2
    assert engine._last_update_info == {}
    assert engine._plan is None
    assert engine._plan_metadata is None
    assert engine._plan_manifest is None

    # Explicit shutdown destroys comm2 and is idempotent.
    engine.shutdown()
    comm2.destroy.assert_called_once()
    assert engine.model_update_group is None
    engine.shutdown()
    comm2.destroy.assert_called_once()


def test_sparse_nccl_reinit_destroys_previous_communicator(
    monkeypatch: pytest.MonkeyPatch,
):
    comm1 = MagicMock(name="comm1")
    comm2 = MagicMock(name="comm2")
    comms = iter([comm1, comm2])
    monkeypatch.setattr(
        sparse_nccl_engine, "worker_init_process_group", lambda *_a, **_kw: next(comms)
    )

    engine = SparseNCCLWeightTransferEngine(
        WeightTransferConfig(backend="sparse_nccl"),
        _mock_vllm_config(),  # type: ignore[arg-type]
        torch.device("cpu"),
        MagicMock(spec=torch.nn.Module),
    )

    engine.init_transfer_engine(_init_info(29600))
    assert engine.model_update_group is comm1
    comm1.destroy.assert_not_called()

    engine.init_transfer_engine(_init_info(29601))
    comm1.destroy.assert_called_once()
    assert engine.model_update_group is comm2

    engine.shutdown()
    comm2.destroy.assert_called_once()
    assert engine.model_update_group is None
