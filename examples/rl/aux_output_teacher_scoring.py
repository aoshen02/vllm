# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Score a multi-turn conversation incrementally with a teacher model.

Each turn sends the whole conversation so far as one request with
`max_tokens=1`. The teacher returns the top-k next-token logprobs of every
position through a shared object store. Prefix-cache hits keep the rows of the
turns already scored, so each turn computes only its new tokens.

Requires a MoE teacher and a running Mooncake master, for example:

    mooncake_master --port=50051 &
    export MOONCAKE_CONFIG_PATH=mooncake.json  # points at 127.0.0.1:50051
    python examples/rl/aux_output_teacher_scoring.py --model Qwen/Qwen3-30B-A3B
"""

import argparse

import numpy as np

from vllm import LLM, SamplingParams
from vllm.distributed.aux_output_connector.object_store import (
    MooncakeAuxObjectStore,
)
from vllm.distributed.aux_output_connector.objects import read_aux_rows
from vllm.inputs import TokensPrompt

TURNS = [
    (
        "User: I am planning a week of hiking in the Alps in late September. What "
        "should I pack, how cold does it get at night above two thousand meters, and "
        "do I need to book the mountain huts in advance or can I just show up?\n"
    ),
    (
        "Assistant: Pack layers: a base layer, a fleece, a down jacket and a "
        "waterproof shell, plus a warm hat and gloves. Above two thousand meters, "
        "nights in late September often drop to around zero degrees Celsius, and "
        "early snow is possible. Many huts close for the season in mid or late "
        "September, so check their dates and book ahead; the ones that stay open "
        "fill up on weekends.\n"
    ),
    (
        "User: Thanks. If the weather turns bad for two or three days, what are "
        "good low-altitude alternatives, and is it realistic to switch between "
        "valleys by public transport without a car?\n"
    ),
    (
        "Assistant: Valley walks along lakes and gorges stay safe in bad weather, "
        "and many towns have museums and thermal baths. Switzerland and Austria "
        "have dense train and post bus networks that reach most trailheads, so "
        "moving between valleys without a car is easy; a regional travel pass "
        "usually pays for itself within a few days.\n"
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--moe-backend", default=None)
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        kernel_config={"moe_backend": args.moe_backend} if args.moe_backend else None,
        enable_prefix_caching=True,
        aux_output_config={
            "enable_return_routed_experts": True,
            "return_topk_logprobs": args.topk,
            "object_store": {"backend": "mooncake", "key_prefix": "teacher"},
        },
    )
    store = MooncakeAuxObjectStore.create(None, "127.0.0.1")
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(max_tokens=1, temperature=0.0, aux_session="conv-0")

    token_ids: list[int] = []
    keys: set[str] = set()
    previous = None
    for turn in TURNS:
        token_ids += tokenizer.encode(turn, add_special_tokens=False)
        output = llm.generate(
            [TokensPrompt(prompt_token_ids=token_ids)], params, use_tqdm=False
        )[0]
        completion = output.outputs[0]
        keys.update(completion.aux_output_keys)
        fields = completion.aux_fields
        topk_ids = read_aux_rows(
            store,
            completion.aux_output_keys,
            "topk_token_ids",
            fields["topk_token_ids"],
        )
        topk_logprobs = read_aux_rows(
            store, completion.aux_output_keys, "topk_logprobs", fields["topk_logprobs"]
        )
        assert topk_ids.shape == (len(token_ids), args.topk)
        cached = output.num_cached_tokens
        if previous is not None:
            assert np.array_equal(topk_ids[:cached], previous[:cached])
        previous = topk_ids
        target = np.array(token_ids[1:])
        in_topk = (topk_ids[:-1] == target[:, None]).any(axis=1).mean()
        print(
            f"{len(token_ids)} tokens, {cached} from cache; "
            f"next token in teacher top-{args.topk}: {in_topk:.0%}; "
            f"mean top-1 logprob {topk_logprobs[:, 0].mean():.3f}"
        )

    store.remove(sorted(keys))
    store.close()


if __name__ == "__main__":
    main()
