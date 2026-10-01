# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-sided workspace with storage-owned, native DLPack metadata."""

import torch
from flashinfer.comm.mnnvl import MnnvlMemory
from flashinfer.comm.trtllm_moe_alltoall import MoeAlltoAll, moe_a2a_initialize

from vllm import _mnnvl_C


class NemotronMoeAlltoAll(MoeAlltoAll):
    # Do not reuse tensors built by FlashInfer's Python DLPack wrapper.
    _WORKSPACE_CACHE: dict = {}

    @classmethod
    def get_workspace(
        cls,
        workspace_size_per_rank: int,
        ep_rank: int,
        ep_size: int,
        max_num_tokens: int,
        mapping,
        eplb_stats_num_experts: int = 0,
    ) -> dict:
        key = (
            workspace_size_per_rank,
            ep_rank,
            ep_size,
            max_num_tokens,
            eplb_stats_num_experts,
        )
        if key not in cls._WORKSPACE_CACHE:
            memory = MnnvlMemory(mapping, workspace_size_per_rank)
            segments = MnnvlMemory.comm.Get_size()
            if segments != ep_size:
                raise ValueError("MNNVL communicator does not match expert parallelism")
            capsule = _mnnvl_C.make_capsule(
                memory.ptr,
                segments,
                memory.segment_size,
                memory.rank_stride,
                2,  # kDLCUDA
                MnnvlMemory.dev_id,
            )
            workspace = torch.utils.dlpack.from_dlpack(capsule)
            metainfo = moe_a2a_initialize(
                workspace, ep_rank, ep_size, max_num_tokens, eplb_stats_num_experts
            )
            # MnnvlMemory owns the allocation; the tensor owns only metadata.
            cls._WORKSPACE_CACHE[key] = {
                "workspace_size_per_rank": workspace_size_per_rank,
                "max_num_tokens": max_num_tokens,
                "ep_rank": ep_rank,
                "ep_size": ep_size,
                "eplb_stats_num_experts": eplb_stats_num_experts,
                "mnnvl_mem": memory,
                "workspace": workspace,
                "metainfo": metainfo,
            }
        return cls._WORKSPACE_CACHE[key]
