# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Let R3 keep the decoder replay trim on DeepSeek-V4.1.

The image runs every row through layers 21-39 once R3 is on. Rows below both the prefix
hit and the request's first needed block are served from the pool, so only the rows from
there on must come out exact: each stage keeps its usual rows plus (needed rows -
window). Captures write trimmed layers by batch row, and only the needed rows: every
other row of such a layer stays -1 (dtype max after narrowing), including kept rows
whose results are inexact.
"""

import pathlib
import sys

sp = sys.argv[1]
NL = chr(10)


def sub(s, old, new):
    old = NL.join(old) + NL
    new = NL.join(new) + NL
    assert s.count(old) == 1, old
    return s.replace(old, new)


def load(p):
    return pathlib.Path(p).read_text(encoding="utf-8")


def save(p, s):
    pathlib.Path(p).write_text(s, encoding="utf-8", newline=NL)


p = f"{sp}/models/deepseek_v41/nvidia/model_state.py"
s = load(p)
s = sub(
    s,
    ["            self._keeps_rows_np = np.zeros(self.max_num_reqs, dtype=np.bool_)"],
    [
        "            self._keeps_rows_np = np.zeros(self.max_num_reqs, dtype=np.bool_)",
        "            self._r3_aux_np = np.zeros(self.max_num_reqs, dtype=np.int32)",
        "            self._r3_hit_np = np.zeros(self.max_num_reqs, dtype=np.int32)",
    ],
)
s = sub(
    s,
    [
        "            ) or "
        "self.vllm_config.aux_output_config.enable_return_routed_experts"
    ],
    [
        "            ) or self._r3_keeps_all()",
        "            aux = (None if params is None else "
        "params.routed_experts_prompt_start) or 0",
        "            prompt = new_req_data.prompt_token_ids",
        "            self._r3_aux_np[req_index] = aux if prompt is None else "
        "min(aux, len(prompt))",
        "            self._r3_hit_np[req_index] = new_req_data.r3_hit",
    ],
)
s = sub(
    s,
    [
        "    def add_request(self, req_index: int, new_req_data: NewRequestData) -> "
        "None:"
    ],
    [
        "    def _r3_trims(self) -> bool:",
        "        return (",
        "            self.vllm_config.aux_output_config.enable_return_routed_experts",
        '            and __import__("os").environ.get("OC_R3_TRIM", "1") != "0"',
        "            and self.replay_cudagraphs is None",
        "            and self._bi_replay_gather()",
        "            and self.vllm_config.model_config.get_num_experts() > 256",
        "        )",
        "",
        "    def _r3_keeps_all(self) -> bool:",
        "        return (",
        "            self.vllm_config.aux_output_config.enable_return_routed_experts",
        "            and not self._r3_trims()",
        "        )",
        "",
        "    def _r3_needed_rows(self, input_batch, query_lens):",
        '        """Trailing rows of each request that captures need exact."""',
        "        start = input_batch.num_computed_tokens_np",
        "        if start is None:",
        "            return query_lens",
        "        num_reqs = input_batch.num_reqs",
        "        index = input_batch.idx_mapping_np[:num_reqs]",
        "        block = 128",
        "        first = self._r3_aux_np[index] // block * block",
        "        need = np.minimum(self._r3_hit_np[index], first)",
        "        return np.clip(start[:num_reqs] + query_lens - need, 0, query_lens)",
        "",
        "    def _r3_exact_rows(self, lens, needed):",
        '        """Among a stage\'s kept rows, the ones holding needed rows."""',
        "        starts = np.repeat(np.cumsum(lens) - lens, lens)",
        "        offsets = np.arange(int(lens.sum())) - starts",
        "        first_exact = np.repeat(lens - np.minimum(needed, lens), lens)",
        "        return torch.from_numpy(np.flatnonzero(offsets >= "
        "first_exact)).to(self.device)",
        "",
        "    def add_request(self, req_index: int, new_req_data: NewRequestData) -> "
        "None:",
    ],
)
s = sub(
    s,
    ["        stage_rows = layers.stage_rows", "        if capture_desc is not None:"],
    [
        "        stage_rows = layers.stage_rows",
        "        needed = None",
        "        if capture_desc is not None:",
    ],
)
s = sub(
    s,
    [
        "            stage_lens = [",
        "                np.where(keeps_rows, query_lens, np.minimum(query_lens, "
        "rows))",
    ],
    [
        "            extra = 0",
        "            if self._r3_trims():",
        "                needed = self._r3_needed_rows(input_batch, query_lens)",
        "                extra = np.maximum(0, needed - layers.window)",
        '                if __import__("os").environ.get("OC_R3_PARTIAL", "0") == "1":',
        "                    extra = 0",
        "            stage_lens = [",
        "                np.where(keeps_rows, query_lens, np.minimum(query_lens, "
        "rows + extra))",
    ],
)
s = sub(
    s,
    [
        "                is_padding=is_padding,",
        "            )",
        "            if i == 0:",
        "                replay_batch = ReplayBatch(",
    ],
    [
        "                is_padding=is_padding,",
        "            )",
        "            if needed is not None:",
        "                forward_context.r3_rows = rows[:num_tokens]",
        '                if __import__("os").environ.get("OC_R3_PARTIAL", "0") != "1":',
        "                    forward_context.r3_exact = self._r3_exact_rows(lens, "
        "needed)",
        "            if i == 0:",
        "                replay_batch = ReplayBatch(",
    ],
)
save(p, s)

p = f"{sp}/model_executor/layers/fused_moe/routed_experts_capturer.py"
s = load(p)
s = sub(
    s,
    ["        self.device_buffer[: len(local_topk_ids), layer_id] = local_topk_ids"],
    [
        "        ctx = get_forward_context()",
        '        rows = getattr(ctx, "r3_rows", None)',
        "        if rows is not None:",
        "            column = self.device_buffer[:, layer_id]",
        "            column.fill_(-1)",
        '            exact = getattr(ctx, "r3_exact", None)',
        "            if exact is None:",
        "                column[rows] = local_topk_ids.to(column.dtype)",
        "            else:",
        "                column[rows[exact]] = local_topk_ids[exact].to(column.dtype)",
        "            return",
        "        self.device_buffer[: len(local_topk_ids), layer_id] = local_topk_ids",
    ],
)
save(p, s)

p = f"{sp}/v1/core/sched/output.py"
s = load(p)
s = sub(
    s,
    ["    replay_start: int = 0"],
    ["    replay_start: int = 0", "    r3_hit: int = 0"],
)
s = sub(
    s,
    ["            replay_start=request.replay_start,"],
    [
        "            replay_start=request.replay_start,",
        '            r3_hit=getattr(request, "r3_hit", 0),',
    ],
)
save(p, s)

p = f"{sp}/v1/core/sched/scheduler.py"
s = load(p)
s = sub(
    s,
    [
        "                        and 0 < num_computed_tokens <= "
        "self.prefix_replay_tokens",
        "                    ):",
    ],
    [
        "                        and 0 < num_computed_tokens <= "
        "self.prefix_replay_tokens",
        "                    ):",
        "                        request.r3_short_hit = num_computed_tokens",
    ],
)
s = sub(
    s,
    [
        "        if not self.prefix_replay_tokens:",
        "            return 0",
        "        if window_loaded:",
    ],
    [
        "        request.r3_hit = max(num_hit_tokens, getattr(request, "
        '"r3_short_hit", 0))',
        "        request.r3_short_hit = 0",
        "        if not self.prefix_replay_tokens:",
        "            return 0",
        "        if window_loaded:",
    ],
)
s = sub(
    s,
    [
        "        request.status = RequestStatus.PREEMPTED",
        "        request.num_computed_tokens = 0",
    ],
    [
        "        request.status = RequestStatus.PREEMPTED",
        "        request.num_computed_tokens = 0",
        "        request.r3_hit = 0",
        "        request.r3_short_hit = 0",
    ],
)
save(p, s)
print("r3 trim patched")
