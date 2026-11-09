# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Snapshot the SWA window at request end on DeepSeek-V4.1 (VLLM_SWA_DECODE_SNAPSHOT=1).

With swa_bounded_replay=False and prefix_cache_retention_interval=0 the image hashes
only the SWA tail ending at the prompt-end boundary; decode blocks leave the window
unhashed. With the env var set, each prefix-cacheable sliding-window group keeps the
`need` blocks (128 rows) that end at the latest aligned committed point allocated, and
hashes them when the request is freed (finish, abort, preemption), so a later prompt
that extends this sequence token-for-token hits there. Unset, every patched path is a
no-op. Speculative decoding is refused.

Usage: python3 patch_swa_snapshot.py <site-packages/vllm dir>
"""

import pathlib
import sys

sp = sys.argv[1]
NL = chr(10)
ENV = "VLLM_SWA_DECODE_SNAPSHOT"


def sub(s, old, new):
    old = NL.join(old) + NL
    new = NL.join(new) + NL
    assert s.count(old) == 1, old
    return s.replace(old, new)


def load(p):
    return pathlib.Path(p).read_text(encoding="utf-8")


def save(p, s):
    pathlib.Path(p).write_text(s, encoding="utf-8", newline=NL)


p = f"{sp}/v1/core/single_type_kv_cache_manager.py"
s = load(p)
s = sub(
    s,
    ["import itertools", "from abc import ABC, abstractmethod"],
    ["import itertools", "import os", "from abc import ABC, abstractmethod"],
)
s = sub(
    s,
    ["        self.extra_retained_tokens = kv_cache_spec.extra_retained_tokens"],
    [
        "        self.extra_retained_tokens = kv_cache_spec.extra_retained_tokens",
        "        self.decode_snapshot = (",
        f'            os.environ.get("{ENV}", "0") == "1"',
        "            and self.enable_caching",
        "            and kv_cache_spec.prefix_cacheable",
        "        )",
    ],
)
s = sub(
    s,
    [
        "        return max(",
        "            0,",
        "            num_computed_tokens - self.sliding_window + 1 - "
        "self.extra_retained_tokens,",
        "        )",
        "",
        "    def get_num_hit_skipped_tokens(self, num_computed_tokens: int) -> int:",
        '        """A bounded replay longer than the window rewrites the hit\'s last',
        '        ``prefix_replay_tokens`` tokens, so their blocks are needed too."""',
        "        num_skipped_tokens = self.get_num_skipped_tokens(num_computed_tokens)",
    ],
    [
        "        num_skipped_tokens = self._window_skipped_tokens(num_computed_tokens)",
        "        if self.decode_snapshot:",
        "            snapshot_start = (",
        "                self._snapshot_point(num_computed_tokens) - "
        "self._snapshot_tokens()",
        "            )",
        "            num_skipped_tokens = min(num_skipped_tokens, max(0, "
        "snapshot_start))",
        "        return num_skipped_tokens",
        "",
        "    def _window_skipped_tokens(self, num_computed_tokens: int) -> int:",
        "        return max(",
        "            0,",
        "            num_computed_tokens - self.sliding_window + 1 - "
        "self.extra_retained_tokens,",
        "        )",
        "",
        "    def _snapshot_point(self, num_tokens: int) -> int:",
        "        alignment = self.cache_hit_alignment_tokens",
        "        return num_tokens // alignment * alignment",
        "",
        "    def _snapshot_tokens(self) -> int:",
        "        need = self._contiguous_blocks_for_hit(",
        "            self.sliding_window, self.block_size, use_eagle=False",
        "        )",
        "        return need * self.block_size",
        "",
        "    def cache_final_snapshot(self, request: Request, num_committed_tokens: "
        "int) -> None:",
        '        """Hash the blocks a hit at the last aligned committed point '
        'needs."""',
        "        if not self.decode_snapshot:",
        "            return",
        "        if self.cache_hit_alignment_tokens % self.block_size:",
        "            return",
        "        request_id = request.request_id",
        "        blocks = self.req_to_blocks.get(request_id)",
        "        if not blocks:",
        "            return",
        "        num_committed_tokens = min(",
        "            num_committed_tokens,",
        "            self.num_cached_block.get(request_id, 0) * self.block_size,",
        "        )",
        "        end = self._snapshot_point(num_committed_tokens) // self.block_size",
        "        start = end - self._snapshot_tokens() // self.block_size",
        "        if start < 0 or end > len(blocks):",
        "            return",
        "        tail = blocks[start:end]",
        "        if any(block.is_null for block in tail):",
        "            return",
        "        block_mask = [block.block_hash is None for block in tail]",
        "        if not any(block_mask):",
        "            return",
        "        self.block_pool.cache_full_blocks(",
        "            request=request,",
        "            blocks=blocks,",
        "            num_cached_blocks=start,",
        "            num_full_blocks=end,",
        "            block_size=self.block_size,",
        "            kv_cache_group_id=self.kv_cache_group_id,",
        "            block_mask=block_mask,",
        "        )",
        "",
        "    def get_num_hit_skipped_tokens(self, num_computed_tokens: int) -> int:",
        '        """A bounded replay longer than the window rewrites the hit\'s last',
        '        ``prefix_replay_tokens`` tokens, so their blocks are needed too."""',
        "        num_skipped_tokens = self._window_skipped_tokens(num_computed_tokens)",
    ],
)
save(p, s)

p = f"{sp}/v1/kv_cache_interface.py"
s = load(p)
s = sub(
    s,
    [
        "        num_tokens = min(",
        "            self.sliding_window - 1 + self.extra_retained_tokens + "
        "max_in_flight_tokens,",
        "            max_model_len,",
        "        )",
    ],
    [
        "        snapshot_hold = (",
        "            2 * self.sliding_window",
        f'            if __import__("os").environ.get("{ENV}", "0") == "1"',
        "            else 0",
        "        )",
        "        num_tokens = min(",
        "            self.sliding_window - 1 + self.extra_retained_tokens + "
        "max_in_flight_tokens",
        "            + snapshot_hold,",
        "            max_model_len,",
        "        )",
    ],
)
save(p, s)

p = f"{sp}/v1/core/kv_cache_coordinator.py"
s = load(p)
s = sub(
    s,
    ["from abc import ABC, abstractmethod", "from collections.abc import Sequence"],
    [
        "import os",
        "from abc import ABC, abstractmethod",
        "from collections.abc import Sequence",
    ],
)
s = sub(
    s,
    [
        "        _validate_prefix_cache_retention_interval(",
        "            self.retention_interval, self.scheduler_block_size, "
        "kv_cache_config",
        "        )",
        "",
        "    def get_num_blocks_to_allocate(",
    ],
    [
        "        _validate_prefix_cache_retention_interval(",
        "            self.retention_interval, self.scheduler_block_size, "
        "kv_cache_config",
        "        )",
        "        self.swa_decode_snapshot = (",
        f'            enable_caching and os.environ.get("{ENV}", "0") == "1"',
        "        )",
        "        if self.swa_decode_snapshot:",
        "            self._validate_swa_decode_snapshot(use_eagle, "
        "num_prefill_lookahead)",
        "",
        "    def _validate_swa_decode_snapshot(",
        "        self, use_eagle: bool, num_prefill_lookahead: int",
        "    ) -> None:",
        f'        name = "{ENV}=1"',
        "        if use_eagle or self.eagle_group_ids or num_prefill_lookahead > 0:",
        "            raise ValueError(",
        '                f"{name} does not support speculative decoding (EAGLE/MTP)."',
        "            )",
        "        if self.retention_interval != 0:",
        '            raise ValueError(f"{name} requires '
        'prefix_cache_retention_interval=0.")',
        "        managers = [",
        "            m",
        "            for m in self.single_type_managers",
        '            if getattr(m, "decode_snapshot", False)',
        "        ]",
        "        if not managers:",
        "            raise ValueError(",
        '                f"{name} needs a prefix-cacheable sliding-window KV cache '
        'group "',
        '                "(DeepSeek-V4.1: swa_bounded_replay=False)."',
        "            )",
        "        for m in managers:",
        "            alignment = m.cache_hit_alignment_tokens",
        "            hold = alignment + m._snapshot_tokens() - m.sliding_window",
        "            if (",
        "                m.extra_retained_tokens",
        "                or alignment % m.block_size",
        "                or hold > 2 * m.sliding_window",
        "            ):",
        "                raise ValueError(",
        '                    f"{name} does not support this sliding-window group: "',
        '                    f"window={m.sliding_window} block={m.block_size} "',
        '                    f"alignment={alignment} "',
        '                    f"extra_retained={m.extra_retained_tokens}."',
        "                )",
        "",
        "    def get_num_blocks_to_allocate(",
    ],
)
s = sub(
    s,
    [
        "    def emit_cached_block_events(",
        "        self, request: Request, computed_blocks: tuple[list[KVCacheBlock], "
        "...]",
        "    ) -> None:",
    ],
    [
        "    def cache_final_snapshot(self, request: Request) -> None:",
        '        """Hash each sliding-window group\'s tail at the committed point."""',
        '        if not getattr(self, "swa_decode_snapshot", False):',
        "            return",
        "        num_committed_tokens = min(",
        "            request.num_tokens - 1,",
        "            request.num_computed_tokens - request.num_in_flight_tokens,",
        "        )",
        "        if num_committed_tokens <= 0:",
        "            return",
        "        for manager in self.single_type_managers:",
        '            if getattr(manager, "decode_snapshot", False):',
        "                manager.cache_final_snapshot(request, num_committed_tokens)",
        "",
        "    def emit_cached_block_events(",
        "        self, request: Request, computed_blocks: tuple[list[KVCacheBlock], "
        "...]",
        "    ) -> None:",
    ],
)
save(p, s)

p = f"{sp}/v1/core/kv_cache_manager.py"
s = load(p)
s = sub(
    s,
    [
        "        self.coordinator.free(request.request_id)",
        "",
        "    def remove_skipped_blocks(",
    ],
    [
        "        self.coordinator.free(request.request_id)",
        "",
        "    def cache_final_snapshot(self, request: Request) -> None:",
        "        if self.enable_caching:",
        "            self.coordinator.cache_final_snapshot(request)",
        "",
        "    def remove_skipped_blocks(",
    ],
)
save(p, s)

p = f"{sp}/v1/core/sched/scheduler.py"
s = load(p)
s = sub(
    s,
    [
        "        pool when an in-flight GPU step may still write them.",
        '        """',
        "        if self._request_blocks_can_be_freed(request):",
        "            self.kv_cache_manager.free(request)",
    ],
    [
        "        pool when an in-flight GPU step may still write them.",
        '        """',
        '        if hasattr(self.kv_cache_manager, "cache_final_snapshot"):',
        "            self.kv_cache_manager.cache_final_snapshot(request)",
        "        if self._request_blocks_can_be_freed(request):",
        "            self.kv_cache_manager.free(request)",
    ],
)
save(p, s)
print("swa decode snapshot patched")
