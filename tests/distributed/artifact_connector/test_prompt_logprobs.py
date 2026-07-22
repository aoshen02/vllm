# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.distributed.artifact_connector import (
    LocalSharedMemoryArtifactStore,
    PromptLogprobsArtifactRequest,
)
from vllm.distributed.artifact_connector.prompt_logprobs import (
    PromptLogprobsArrays,
    PromptLogprobsArtifactManager,
)
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample.prompt_logprob import PromptLogprobsWorker

pytestmark = pytest.mark.cpu_test


def _store(tmp_path, *, max_bytes: int = 1 << 20):
    return LocalSharedMemoryArtifactStore(
        str(tmp_path),
        "instance",
        0,
        max_bytes=max_bytes,
        ttl_seconds=60,
    )


def _spec(
    request_id: str,
    *,
    cached_tokens: int = 0,
    num_prompt_logprobs: int = 2,
    block_hashes: list[bytes] | None = None,
) -> PromptLogprobsArtifactRequest:
    return PromptLogprobsArtifactRequest(
        request_id=request_id,
        block_hashes=block_hashes
        or [bytes([block_index + 1]) * 32 for block_index in range(2)],
        num_prompt_tokens=10,
        num_prompt_logprobs=num_prompt_logprobs,
        num_cached_tokens=cached_tokens,
        hash_block_size=4,
        policy_epoch=0,
    )


def _arrays() -> PromptLogprobsArrays:
    token_ids = np.arange(9 * 3, dtype=np.int32).reshape(9, 3)
    return PromptLogprobsArrays(
        token_ids=token_ids,
        logprobs=token_ids.astype(np.float32) / 10,
        ranks=np.arange(9, dtype=np.int32),
    )


def _publish(manager: PromptLogprobsArtifactManager) -> PromptLogprobsArrays:
    spec = _spec("producer")
    arrays = _arrays()
    manager.store_completed_blocks(
        spec,
        arrays,
        completed_token_end=8,
        boundary_hidden={
            0: np.arange(6, dtype=np.float32),
            1: np.arange(6, dtype=np.float32) + 10,
        },
    )
    return manager.finalize(spec, arrays)


def test_prompt_logprobs_full_blocks_tail_and_manifest_round_trip(tmp_path):
    store = _store(tmp_path)
    manager = PromptLogprobsArtifactManager(
        store, namespace="test", logprobs_mode="raw_logprobs"
    )

    materialized = _publish(manager)

    expected = _arrays()
    np.testing.assert_array_equal(materialized.token_ids, expected.token_ids)
    np.testing.assert_array_equal(materialized.logprobs, expected.logprobs)
    np.testing.assert_array_equal(materialized.ranks, expected.ranks)
    assert len(list(store.blocks_dir.glob("*.bin"))) == 2
    assert len(list(store.tails_dir.glob("*.bin"))) == 1
    assert len(list(store.manifests_dir.glob("*.json"))) == 1
    store.close()


def test_kv_hit_restores_all_block_rows_and_boundary_hidden(tmp_path):
    store = _store(tmp_path)
    manager = PromptLogprobsArtifactManager(
        store, namespace="test", logprobs_mode="raw_logprobs"
    )
    _publish(manager)

    restored = manager.restore_cached_prefix(_spec("consumer", cached_tokens=8))

    assert restored is not None
    expected = _arrays()
    np.testing.assert_array_equal(restored.token_ids, expected.token_ids[:7])
    np.testing.assert_array_equal(restored.logprobs, expected.logprobs[:7])
    np.testing.assert_array_equal(restored.ranks, expected.ranks[:7])
    np.testing.assert_array_equal(
        restored.boundary_hidden, np.arange(6, dtype=np.float32) + 10
    )
    manager.discard("consumer")
    store.close()


def test_kv_hit_with_missing_prompt_logprobs_block_fails_closed(tmp_path):
    store = _store(tmp_path)
    manager = PromptLogprobsArtifactManager(
        store, namespace="test", logprobs_mode="raw_logprobs"
    )
    _publish(manager)
    missing_hashes = [bytes([1]) * 32, bytes([9]) * 32]

    with pytest.raises(
        RuntimeError,
        match="mandatory prompt-logprobs artifact is missing",
    ):
        manager.restore_cached_prefix(
            _spec(
                "consumer",
                cached_tokens=8,
                block_hashes=missing_hashes,
            )
        )
    store.close()


def test_kv_hit_does_not_fall_back_across_artifact_profiles(tmp_path):
    store = _store(tmp_path)
    manager = PromptLogprobsArtifactManager(
        store, namespace="test", logprobs_mode="raw_logprobs"
    )
    _publish(manager)

    with pytest.raises(
        RuntimeError,
        match="mandatory prompt-logprobs artifact is missing",
    ):
        manager.restore_cached_prefix(
            _spec("consumer", cached_tokens=8, num_prompt_logprobs=1)
        )
    store.close()


def test_kv_hit_with_evicted_prompt_logprobs_block_fails_closed(tmp_path):
    store = _store(tmp_path)
    manager = PromptLogprobsArtifactManager(
        store, namespace="test", logprobs_mode="raw_logprobs"
    )
    _publish(manager)
    block_path = next(store.blocks_dir.glob("*.bin"))
    block_path.unlink()

    with pytest.raises(
        RuntimeError,
        match="mandatory prompt-logprobs artifact is missing",
    ):
        manager.restore_cached_prefix(_spec("consumer", cached_tokens=8))
    store.close()


def test_mrv2_worker_tracks_prompt_logprobs_artifact_request():
    worker = PromptLogprobsWorker(max_num_reqs=2)
    spec = _spec("request-a", cached_tokens=0)

    worker.add_request(
        "request-a",
        0,
        SamplingParams(prompt_logprobs=2),
        spec,
    )

    assert worker.artifact_requests == {"request-a": spec}
    worker.remove_request("request-a")
    assert not worker.artifact_requests


def test_mrv2_worker_restores_mandatory_prefix_and_boundary():
    worker = PromptLogprobsWorker(max_num_reqs=1)
    spec = _spec("request-a", cached_tokens=8)
    worker.artifact_requests["request-a"] = spec
    restored = _arrays()
    connector = Mock()
    connector.restore_prompt_logprobs.return_value = PromptLogprobsArrays(
        token_ids=restored.token_ids[:7],
        logprobs=restored.logprobs[:7],
        ranks=restored.ranks[:7],
        boundary_hidden=np.arange(6, dtype=np.float32),
    )
    worker.artifact_connector = connector
    tp_group = SimpleNamespace(
        rank_in_group=0,
        broadcast_object=lambda value, src: value,
        broadcast=lambda tensor, src: None,
    )
    boundary = LogprobsTensors(
        logprob_token_ids=torch.tensor([[8, 9, 10]], dtype=torch.int64),
        logprobs=torch.tensor([[0.8, 0.9, 1.0]], dtype=torch.float32),
        selected_token_ranks=torch.tensor([2], dtype=torch.int64),
    )

    with (
        patch(
            "vllm.v1.worker.gpu.sample.prompt_logprob.get_tp_group",
            return_value=tp_group,
        ),
        patch(
            "vllm.v1.worker.gpu.sample.prompt_logprob.compute_topk_scores",
            return_value=boundary,
        ),
    ):
        result = worker._restore_artifact_prefix(
            req_id="request-a",
            req_idx=0,
            logits_fn=lambda hidden: torch.zeros((1, 16)),
            all_token_ids=torch.arange(10).reshape(1, 10),
            hidden_states=torch.zeros((2, 6)),
        )

    assert result is not None
    assert result.logprob_token_ids.shape == (8, 3)
    assert result.logprobs.shape == (8, 3)
    assert result.selected_token_ranks.shape == (8,)
    connector.restore_prompt_logprobs.assert_called_once_with(spec)


def test_mrv2_worker_fails_closed_without_rank0_connector():
    worker = PromptLogprobsWorker(max_num_reqs=1)
    spec = _spec("request-a", cached_tokens=8)
    worker.artifact_requests["request-a"] = spec
    tp_group = SimpleNamespace(
        rank_in_group=0,
        broadcast_object=lambda value, src: value,
    )

    with (
        patch(
            "vllm.v1.worker.gpu.sample.prompt_logprob.get_tp_group",
            return_value=tp_group,
        ),
        pytest.raises(RuntimeError, match="without an ArtifactWorkerConnector"),
    ):
        worker._restore_artifact_prefix(
            req_id="request-a",
            req_idx=0,
            logits_fn=lambda hidden: torch.zeros((1, 16)),
            all_token_ids=torch.arange(10).reshape(1, 10),
            hidden_states=torch.zeros((2, 6)),
        )


def test_mrv1_does_not_create_prompt_logprobs_artifact_request():
    scheduler = object.__new__(Scheduler)
    scheduler.use_v2_model_runner = False
    scheduler.artifact_connector = Mock()
    request = SimpleNamespace(
        sampling_params=SamplingParams(prompt_logprobs=2),
    )

    result = scheduler._make_prompt_logprobs_artifact_request(request)

    assert result is None
    scheduler.artifact_connector.make_prompt_logprobs_request.assert_not_called()
