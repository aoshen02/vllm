# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.outputs import MAX_COMPACT_SUPPORT, LogprobsTensors, SamplingMaskLists
from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

__all__ = ["MAX_COMPACT_SUPPORT", "SamplerOutput", "SamplingMaskTensors"]


@dataclass
class SamplerOutput:
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None
    num_nans: torch.Tensor | None
    num_sampled: torch.Tensor
    num_rejected: torch.Tensor
    sampling_mask_tensors: SamplingMaskTensors | None = None


@triton.jit
def _compact_sampling_mask_kernel(
    logits_ptr,
    logits_row_stride,
    logits_col_stride,
    num_sampled_tokens_ptr,
    token_ids_ptr,
    token_ids_row_stride,
    packed_mask_ptr,
    packed_mask_row_stride,
    counts_ptr,
    vocab_size,
    max_num_kept,
    BLOCK_SIZE: tl.constexpr,
    WRITE_PACKED_MASK: tl.constexpr,
):
    """Per row: first ``max_num_kept`` finite-logit ids, the full count, the bitmask."""
    req_idx = tl.program_id(0)
    is_active = tl.load(num_sampled_tokens_ptr + req_idx) > 0
    count = tl.zeros((), dtype=tl.int32)

    for start_idx in range(0, vocab_size, BLOCK_SIZE):
        offsets = start_idx + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(
            logits_ptr + req_idx * logits_row_stride + offsets * logits_col_stride,
            mask=offsets < vocab_size,
            other=-float("inf"),
        )
        keep = (logits > -float("inf")) & (logits < float("inf")) & is_active
        keep_i32 = keep.to(tl.int32)
        pos = count + tl.cumsum(keep_i32, axis=0) - keep_i32
        tl.store(
            token_ids_ptr + req_idx * token_ids_row_stride + pos,
            offsets.to(tl.int32),
            mask=keep & (pos < max_num_kept),
        )
        count += tl.sum(keep_i32)

        if WRITE_PACKED_MASK:
            bits = (
                tl.reshape(keep_i32, (BLOCK_SIZE // 8, 8)) << tl.arange(0, 8)[None, :]
            )
            byte_offsets = start_idx // 8 + tl.arange(0, BLOCK_SIZE // 8)
            tl.store(
                packed_mask_ptr + req_idx * packed_mask_row_stride + byte_offsets,
                tl.sum(bits, axis=1).to(tl.uint8),
                mask=byte_offsets < tl.cdiv(vocab_size, 8),
            )

    tl.store(counts_ptr + req_idx, count)


class SamplingMaskTensors(NamedTuple):
    """Device-side masks pending async D2H: compact ids, plus the bitmask as
    the exact fallback for rows wider than ``max_num_kept``.

    With ``logprobs`` the layout is fixed-capacity instead: ``token_ids`` and
    ``logprobs`` are ``[num_requests, max_num_kept]``, only the first
    ``counts[i]`` slots of row ``i`` are meaningful, no bitmask is written and
    nothing on this path synchronizes with the device.
    """

    # [num_requests, max_num_kept]
    token_ids: torch.Tensor
    # [num_requests, ceil(vocab_size / 8)]; [num_requests, 0] with ``logprobs``.
    packed_mask: torch.Tensor
    # [num_requests]
    counts: torch.Tensor
    vocab_size: int
    # [num_requests, max_num_kept] log-softmax over the processed logits,
    # gathered at ``token_ids``; slots past ``counts[i]`` are garbage.
    logprobs: torch.Tensor | None = None

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        num_sampled_tokens: torch.Tensor,
        max_num_kept: int,
        return_logprobs: bool = False,
    ) -> SamplingMaskTensors:
        """Capture the finite-logit support of every row with a sampled token."""
        num_reqs, vocab_size = logits.shape
        max_num_kept = min(max_num_kept, vocab_size, MAX_COMPACT_SUPPORT)
        device = logits.device

        if return_logprobs:
            # Padded slots are gathered by the logprob kernel, so they must
            # hold a valid (any) token id.
            token_ids = torch.zeros(
                (num_reqs, max_num_kept), dtype=torch.int32, device=device
            )
            packed_mask = torch.empty((num_reqs, 0), dtype=torch.uint8, device=device)
        else:
            token_ids = torch.empty(
                (num_reqs, max_num_kept), dtype=torch.int32, device=device
            )
            packed_mask = torch.empty(
                (num_reqs, (vocab_size + 7) // 8), dtype=torch.uint8, device=device
            )
        counts = torch.empty(num_reqs, dtype=torch.int32, device=device)
        _compact_sampling_mask_kernel[(num_reqs,)](
            logits,
            logits.stride(0),
            logits.stride(1),
            num_sampled_tokens,
            token_ids,
            token_ids.stride(0),
            packed_mask,
            packed_mask.stride(0),
            counts,
            vocab_size,
            max_num_kept,
            BLOCK_SIZE=8192,
            WRITE_PACKED_MASK=not return_logprobs,
        )
        logprobs = None
        if return_logprobs:
            # Same fused max/logsumexp + gather as sampled-token logprobs;
            # never materializes the full log-softmax.
            logprobs = compute_token_logprobs(logits, token_ids)
        return cls(token_ids, packed_mask, counts, vocab_size, logprobs)

    def to_cpu_nonblocking(self) -> SamplingMaskTensors:
        if self.token_ids.device.type == "cpu":
            return self
        logprobs = self.logprobs
        if logprobs is not None:
            logprobs = logprobs.to("cpu", non_blocking=True)
        return SamplingMaskTensors(
            self.token_ids.to("cpu", non_blocking=True),
            self.packed_mask.to("cpu", non_blocking=True),
            self.counts.to("cpu", non_blocking=True),
            self.vocab_size,
            logprobs,
        )

    def tolists(self) -> SamplingMaskLists:
        """CSR over all requests; rows without a sampled token are empty."""
        counts = self.counts.cpu().numpy()
        token_ids = self.token_ids.cpu().numpy()
        width = token_ids.shape[1]

        if self.logprobs is not None:
            # Fixed capacity: rows wider than the buffer (top-k boundary ties)
            # cannot be represented, so they are flagged rather than truncated
            # or padded silently; the scheduler fails those requests.
            kept = np.minimum(counts, width)
            valid = np.arange(width)[None, :] < kept[:, None]
            offsets = np.zeros(len(counts) + 1, dtype=np.int64)
            np.cumsum(kept, out=offsets[1:])
            overflow = counts > width
            return SamplingMaskLists(
                token_ids[valid],
                offsets,
                logprobs=self.logprobs.cpu().numpy()[valid],
                overflow=overflow if overflow.any() else None,
            )

        packed_mask = self.packed_mask.cpu().numpy()

        def support(row: int) -> np.ndarray:
            if counts[row] <= width:
                return token_ids[row, : counts[row]]
            # Wider than the compact row (ties or a huge top_k): use the bitmask.
            bits = np.unpackbits(
                packed_mask[row], count=self.vocab_size, bitorder="little"
            )
            return np.flatnonzero(bits).astype(np.int32, copy=False)

        supports = [support(row) for row in range(len(counts))]
        offsets = np.zeros(len(supports) + 1, dtype=np.int64)
        np.cumsum([len(s) for s in supports], out=offsets[1:])
        return SamplingMaskLists(np.concatenate(supports), offsets)
