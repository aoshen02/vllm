# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.config.model import LogprobsMode
from vllm.distributed.artifact_connector.prompt_logprobs import PromptLogprobsArrays
from vllm.distributed.artifact_connector.protocol import (
    PromptLogprobsArtifactRequest,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores

if TYPE_CHECKING:
    from vllm.distributed.artifact_connector import ArtifactWorkerConnector


class PromptLogprobsWorker:
    def __init__(self, max_num_reqs: int, logprobs_mode: LogprobsMode = "raw_logprobs"):
        self.max_num_reqs = max_num_reqs
        self.logprobs_mode = logprobs_mode

        self.uses_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=bool)
        self.num_prompt_logprobs = np.zeros(self.max_num_reqs, dtype=np.int32)
        # req_idx -> list of in-progress LogprobsTensors
        self.in_progress_prompt_logprobs: dict[str, list[LogprobsTensors]] = {}
        self.artifact_connector: ArtifactWorkerConnector | None = None
        self.artifact_requests: dict[str, PromptLogprobsArtifactRequest] = {}
        self.restored_artifact_requests: set[str] = set()

    def add_request(
        self,
        req_id: str,
        req_idx: int,
        sampling_params: SamplingParams,
        artifact_request: PromptLogprobsArtifactRequest | None = None,
    ):
        uses_prompt_logprobs = sampling_params.prompt_logprobs is not None
        self.uses_prompt_logprobs[req_idx] = uses_prompt_logprobs
        self.num_prompt_logprobs[req_idx] = sampling_params.prompt_logprobs or 0
        if uses_prompt_logprobs:
            self.in_progress_prompt_logprobs[req_id] = []
        if artifact_request is not None:
            if not uses_prompt_logprobs:
                raise RuntimeError(
                    "prompt-logprobs artifact metadata without prompt logprobs"
                )
            if artifact_request.request_id != req_id:
                raise RuntimeError("prompt-logprobs artifact request ID mismatch")
            self.artifact_requests[req_id] = artifact_request

    def remove_request(self, req_id: str) -> None:
        self.in_progress_prompt_logprobs.pop(req_id, None)
        self.artifact_requests.pop(req_id, None)
        self.restored_artifact_requests.discard(req_id)

    def _restore_artifact_prefix(
        self,
        *,
        req_id: str,
        req_idx: int,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        all_token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> LogprobsTensors | None:
        artifact_request = self.artifact_requests.get(req_id)
        if artifact_request is None or req_id in self.restored_artifact_requests:
            return None
        self.restored_artifact_requests.add(req_id)

        cached_tokens = artifact_request.num_cached_tokens
        if cached_tokens == 0:
            return None
        if cached_tokens >= artifact_request.num_prompt_tokens:
            raise RuntimeError(
                "prompt-logprobs KV hit leaves no boundary token: "
                f"request={req_id}, cached_tokens={cached_tokens}, "
                f"prompt_tokens={artifact_request.num_prompt_tokens}"
            )

        tp_group = get_tp_group()
        restored = None
        restore_error = None
        if tp_group.rank_in_group == 0:
            if self.artifact_connector is None:
                restore_error = (
                    "prompt-logprobs artifact metadata was provided without an "
                    "ArtifactWorkerConnector on TP rank 0"
                )
            else:
                try:
                    restored = self.artifact_connector.restore_prompt_logprobs(
                        artifact_request
                    )
                    expected_rows = cached_tokens - 1
                    expected_width = artifact_request.num_prompt_logprobs + 1
                    if restored is None:
                        raise RuntimeError(
                            "mandatory prompt-logprobs artifact was not restored"
                        )
                    if (
                        restored.token_ids.shape != (expected_rows, expected_width)
                        or restored.logprobs.shape != restored.token_ids.shape
                        or restored.ranks.shape != (expected_rows,)
                        or restored.boundary_hidden is None
                    ):
                        raise RuntimeError(
                            "invalid mandatory prompt-logprobs prefix artifact"
                        )
                except Exception as error:
                    restore_error = f"{type(error).__name__}: {error}"

        restore_error = tp_group.broadcast_object(restore_error, src=0)
        if restore_error is not None:
            raise RuntimeError(
                "failed to restore mandatory prompt-logprobs artifact: "
                f"request={req_id}, error={restore_error}"
            )

        expected_rows = cached_tokens - 1
        width = artifact_request.num_prompt_logprobs + 1
        device = hidden_states.device
        if restored is not None:
            assert restored.boundary_hidden is not None
            restored_token_ids = torch.from_numpy(restored.token_ids).to(device)
            restored_logprobs = torch.from_numpy(restored.logprobs).to(device)
            restored_ranks = torch.from_numpy(restored.ranks).to(device)
            boundary_hidden = torch.from_numpy(restored.boundary_hidden).to(
                device=device, dtype=hidden_states.dtype
            )
        else:
            restored_token_ids = torch.empty(
                (expected_rows, width), dtype=torch.int32, device=device
            )
            restored_logprobs = torch.empty(
                (expected_rows, width), dtype=torch.float32, device=device
            )
            restored_ranks = torch.empty(
                expected_rows, dtype=torch.int32, device=device
            )
            boundary_hidden = torch.empty(
                hidden_states.shape[-1], dtype=hidden_states.dtype, device=device
            )
        tp_group.broadcast(boundary_hidden, src=0)

        boundary_logits = logits_fn(boundary_hidden.unsqueeze(0))
        target_token = all_token_ids[req_idx, cached_tokens : cached_tokens + 1].to(
            torch.int64
        )
        boundary = compute_topk_scores(
            boundary_logits,
            artifact_request.num_prompt_logprobs,
            target_token,
            logits_mode=self.logprobs_mode in ("raw_logits", "processed_logits"),
        )
        return LogprobsTensors(
            logprob_token_ids=torch.cat(
                [restored_token_ids, boundary.logprob_token_ids]
            ),
            logprobs=torch.cat([restored_logprobs, boundary.logprobs]),
            selected_token_ranks=torch.cat(
                [restored_ranks, boundary.selected_token_ranks]
            ),
        )

    @staticmethod
    def _to_artifact_arrays(logprobs: LogprobsTensors) -> PromptLogprobsArrays:
        return PromptLogprobsArrays(
            token_ids=(logprobs.logprob_token_ids.detach().to("cpu").numpy().copy()),
            logprobs=logprobs.logprobs.detach().to("cpu").numpy().copy(),
            ranks=(logprobs.selected_token_ranks.detach().to("cpu").numpy().copy()),
        )

    @staticmethod
    def _merge_logprobs(parts: list[LogprobsTensors]) -> LogprobsTensors:
        if len(parts) == 1:
            return parts[0]
        return LogprobsTensors(
            logprob_token_ids=torch.cat([part.logprob_token_ids for part in parts]),
            logprobs=torch.cat([part.logprobs for part in parts]),
            selected_token_ranks=torch.cat(
                [part.selected_token_ranks for part in parts]
            ),
        )

    def compute_prompt_logprobs(
        self,
        logits_fn: Callable[[torch.Tensor], torch.Tensor],
        hidden_states: torch.Tensor,
        input_batch: InputBatch,
        # [max_num_reqs, max_model_len]
        all_token_ids: torch.Tensor,
        # [max_num_reqs]
        num_computed_tokens: torch.Tensor,
        # [max_num_reqs]
        prompt_lens: np.ndarray,
    ) -> dict[str, LogprobsTensors]:
        idx_mapping_np = input_batch.idx_mapping_np
        needs_prompt_logprobs = self.uses_prompt_logprobs[idx_mapping_np]
        if not np.any(needs_prompt_logprobs):
            # Common case: No request asks for prompt logprobs.
            return {}

        num_prompt_logprobs = self.num_prompt_logprobs[idx_mapping_np]
        prompt_lens = prompt_lens[idx_mapping_np]
        computed_prefill = input_batch.num_computed_prefill_tokens_np
        includes_prompt = computed_prefill < prompt_lens
        # NOTE(woosuk): If the request was resumed after preemption, its prompt
        # logprobs must have been computed before preemption. Skip.
        resumed_after_prompt = prompt_lens < input_batch.prefill_len_np
        needs_prompt_logprobs &= includes_prompt & ~resumed_after_prompt
        if not np.any(needs_prompt_logprobs):
            return {}

        # get the maximum number in this batch
        requested_num_prompt_logprobs = num_prompt_logprobs[needs_prompt_logprobs]
        max_num_prompt_logprobs = (
            -1
            if np.any(requested_num_prompt_logprobs == -1)
            else int(requested_num_prompt_logprobs.max())
        )

        # Get the prompt logprobs token_ids.
        prompt_logprobs_token_ids = get_prompt_logprobs_token_ids(
            input_batch.num_tokens,
            input_batch.query_start_loc,
            input_batch.idx_mapping,
            num_computed_tokens,
            all_token_ids,
        )
        prompt_token_ids, prompt_logprobs, prompt_ranks = (
            compute_prompt_logprobs_with_chunking(
                prompt_logprobs_token_ids,
                hidden_states[: input_batch.num_tokens],
                logits_fn,
                max_num_prompt_logprobs,
                self.logprobs_mode,
            )
        )

        pos_after_step = computed_prefill + input_batch.num_scheduled_tokens
        is_prompt_chunked = pos_after_step < prompt_lens

        query_start_loc_np = input_batch.query_start_loc_np
        prompt_logprobs_dict: dict[str, LogprobsTensors] = {}
        for i, req_id in enumerate(input_batch.req_ids):
            if not needs_prompt_logprobs[i]:
                continue

            restored_prefix = self._restore_artifact_prefix(
                req_id=req_id,
                req_idx=int(idx_mapping_np[i]),
                logits_fn=logits_fn,
                all_token_ids=all_token_ids,
                hidden_states=hidden_states,
            )
            prompt_logprobs_list = self.in_progress_prompt_logprobs[req_id]
            if restored_prefix is not None:
                prompt_logprobs_list.append(restored_prefix)

            req_is_prompt_chunked = is_prompt_chunked[i]
            req_num_prompt_logprobs = int(num_prompt_logprobs[i])
            start_idx = query_start_loc_np[i]
            end_idx = query_start_loc_np[i + 1]
            assert start_idx < end_idx, (
                f"start_idx ({start_idx}) >= end_idx ({end_idx})"
            )
            if not req_is_prompt_chunked:
                end_idx -= 1

            width = (
                prompt_logprobs.shape[1]
                if req_num_prompt_logprobs == -1
                else req_num_prompt_logprobs + 1
            )
            # no logprobs if start_idx >= end_idx
            logprobs = (
                None
                if start_idx >= end_idx
                else LogprobsTensors(
                    logprob_token_ids=prompt_token_ids[start_idx:end_idx, :width],
                    logprobs=prompt_logprobs[start_idx:end_idx, :width],
                    selected_token_ranks=prompt_ranks[start_idx:end_idx],
                )
            )

            if logprobs is not None and (req_is_prompt_chunked or prompt_logprobs_list):
                prompt_logprobs_list.append(logprobs)

            artifact_request = self.artifact_requests.get(req_id)
            if artifact_request is not None and self.artifact_connector is not None:
                completed_token_end = min(
                    int(pos_after_step[i]), artifact_request.num_prompt_tokens
                )
                pending_blocks = self.artifact_connector.pending_prompt_logprobs_blocks(
                    artifact_request, completed_token_end
                )
                if pending_blocks:
                    assembled_parts = list(prompt_logprobs_list)
                    if not assembled_parts and logprobs is not None:
                        assembled_parts.append(logprobs)
                    if not assembled_parts:
                        raise RuntimeError(
                            "completed prompt-logprobs block has no rows: "
                            f"request={req_id}"
                        )
                    boundary_hidden: dict[int, np.ndarray] = {}
                    logical_start = int(computed_prefill[i])
                    logical_end = logical_start + int(
                        input_batch.num_scheduled_tokens[i]
                    )
                    block_size = artifact_request.hash_block_size
                    for block_index in pending_blocks:
                        token_position = (block_index + 1) * block_size - 1
                        if not logical_start <= token_position < logical_end:
                            raise RuntimeError(
                                "completed prompt-logprobs block has no current "
                                "boundary hidden state: "
                                f"request={req_id}, block={block_index}, "
                                f"scheduled=[{logical_start}, {logical_end})"
                            )
                        hidden_index = start_idx + token_position - logical_start
                        boundary_hidden[block_index] = (
                            hidden_states[hidden_index]
                            .detach()
                            .to(device="cpu", dtype=torch.float32)
                            .numpy()
                            .copy()
                        )
                    self.artifact_connector.store_prompt_logprobs_blocks(
                        artifact_request,
                        self._to_artifact_arrays(self._merge_logprobs(assembled_parts)),
                        completed_token_end,
                        boundary_hidden,
                    )
            if req_is_prompt_chunked:
                # Prompt is chunked. Do not return the logprobs yet.
                continue

            if prompt_logprobs_list:
                # Merge the in-progress logprobs.
                logprobs = self._merge_logprobs(prompt_logprobs_list)
                prompt_logprobs_list.clear()

            if logprobs is None:
                continue

            if artifact_request is not None and self.artifact_connector is not None:
                materialized = self.artifact_connector.finalize_prompt_logprobs(
                    artifact_request,
                    self._to_artifact_arrays(logprobs),
                )
                logprobs = LogprobsTensors(
                    logprob_token_ids=torch.from_numpy(materialized.token_ids),
                    logprobs=torch.from_numpy(materialized.logprobs),
                    selected_token_ranks=torch.from_numpy(materialized.ranks),
                )
            prompt_logprobs_dict[req_id] = logprobs
        return prompt_logprobs_dict


@triton.jit
def _prompt_logprobs_token_ids_kernel(
    prompt_logprobs_token_ids_ptr,
    query_start_loc_ptr,
    idx_mapping_ptr,
    num_computed_tokens_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    BLOCK_SIZE: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + batch_idx)

    query_start = tl.load(query_start_loc_ptr + batch_idx)
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1)
    query_len = query_end - query_start

    num_computed_tokens = tl.load(num_computed_tokens_ptr + req_state_idx)
    for i in range(0, query_len, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        mask = block < query_len
        # NOTE(woosuk): We should shift the pos by one
        # because the logprob is computed for the next token.
        target_pos = num_computed_tokens + 1 + block
        token_ids = tl.load(
            all_token_ids_ptr + req_state_idx * all_token_ids_stride + target_pos,
            mask=mask,
        )
        tl.store(
            prompt_logprobs_token_ids_ptr + query_start + block, token_ids, mask=mask
        )


def get_prompt_logprobs_token_ids(
    num_tokens: int,
    query_start_loc: torch.Tensor,
    idx_mapping: torch.Tensor,
    num_computed_tokens: torch.Tensor,
    all_token_ids: torch.Tensor,
) -> torch.Tensor:
    token_ids = torch.empty(num_tokens, dtype=torch.int64, device=idx_mapping.device)
    num_reqs = idx_mapping.shape[0]
    _prompt_logprobs_token_ids_kernel[(num_reqs,)](
        token_ids,
        query_start_loc,
        idx_mapping,
        num_computed_tokens,
        all_token_ids,
        all_token_ids.stride(0),
        BLOCK_SIZE=1024,
    )
    return token_ids


def compute_prompt_logprobs_with_chunking(
    prompt_token_ids: torch.Tensor,
    prompt_hidden_states: torch.Tensor,
    logits_fn: Callable[[torch.Tensor], torch.Tensor],
    num_prompt_logprobs: int,
    logprobs_mode: LogprobsMode = "raw_logprobs",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Since materializing the full prompt logits can take too much memory,
    # we compute it in chunks.
    CHUNK_SIZE = 1024
    token_ids = []
    scores = []
    ranks = []
    logits_mode = logprobs_mode in ("raw_logits", "processed_logits")
    prompt_token_ids = prompt_token_ids.to(torch.int64)
    for start_idx in range(0, prompt_token_ids.shape[0], CHUNK_SIZE):
        end_idx = start_idx + CHUNK_SIZE
        # NOTE(woosuk): logits_fn can be slow because it involves all-gather.
        prompt_logits = logits_fn(prompt_hidden_states[start_idx:end_idx])
        requested_num = (
            prompt_logits.shape[-1]
            if num_prompt_logprobs == -1
            else num_prompt_logprobs
        )
        result = compute_topk_scores(
            prompt_logits,
            requested_num,
            prompt_token_ids[start_idx:end_idx],
            logits_mode=logits_mode,
        )
        token_ids.append(result.logprob_token_ids)
        scores.append(result.logprobs)
        ranks.append(result.selected_token_ranks)

    token_ids = torch.cat(token_ids, dim=0) if len(token_ids) > 1 else token_ids[0]
    scores = torch.cat(scores, dim=0) if len(scores) > 1 else scores[0]
    ranks = torch.cat(ranks, dim=0) if len(ranks) > 1 else ranks[0]
    return token_ids, scores, ranks
