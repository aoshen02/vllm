#!/usr/bin/env python3
"""Teacher-union/student fixed-ID prefill benchmark.

The teacher run supplies a global union of natural prompt top-k IDs. The
student is then run with that frozen union, so candidate construction never
depends on student output. Use identical token IDs for both engines.
"""

from __future__ import annotations

import argparse
import math
import time

import torch

from vllm import LLM, SamplingParams


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--prompt-tokens", type=int, default=8192)
    p.add_argument("--output-tokens", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--max-union", type=int, default=128)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    args = p.parse_args()
    common = dict(
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.prompt_tokens + args.output_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    teacher = LLM(model=args.teacher, **common)
    seed_ids = teacher.get_tokenizer().encode(
        "OPD fixed-token prefill benchmark. The same tokenized prompt is used."
    )
    prompt_ids = (seed_ids * math.ceil(args.prompt_tokens / len(seed_ids)))[
        : args.prompt_tokens
    ]
    teacher_params = SamplingParams(
        prompt_logprobs=args.top_k, max_tokens=args.output_tokens, temperature=0
    )
    start = time.perf_counter()
    teacher_out = teacher.generate([{"prompt_token_ids": prompt_ids}], teacher_params)[
        0
    ]
    teacher_ms = (time.perf_counter() - start) * 1000
    if teacher_out.prompt_logprobs is None:
        raise RuntimeError("teacher did not return prompt_logprobs")
    union = list(
        dict.fromkeys(
            token_id for row in teacher_out.prompt_logprobs if row for token_id in row
        )
    )
    if len(union) > args.max_union:
        raise RuntimeError(
            f"teacher union has {len(union)} IDs, exceeding M1 limit "
            f"{args.max_union}; use per-position candidates (M2) or raise "
            "the limit deliberately."
        )

    del teacher
    student = LLM(model=args.student, **common)
    student_params = SamplingParams(
        prompt_logprob_token_ids=union,
        max_tokens=args.output_tokens,
        temperature=0,
    )
    start = time.perf_counter()
    student_out = student.generate([{"prompt_token_ids": prompt_ids}], student_params)[
        0
    ]
    student_ms = (time.perf_counter() - start) * 1000
    scores = student_out.prompt_token_logprobs
    if scores is None or len(scores) != args.prompt_tokens - 1:
        raise AssertionError(
            f"expected {args.prompt_tokens - 1} score rows, got "
            f"{None if scores is None else len(scores)}"
        )
    fixed_peak_memory = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    student.generate(
        [{"prompt_token_ids": prompt_ids}],
        SamplingParams(max_tokens=args.output_tokens, temperature=0),
    )
    baseline_ms = (time.perf_counter() - start) * 1000
    print(
        {
            "prompt_tokens": args.prompt_tokens,
            "output_tokens": args.output_tokens,
            "teacher_top_k": args.top_k,
            "union_size": len(union),
            "teacher_ms": teacher_ms,
            "student_fixed_ids_ms": student_ms,
            "student_baseline_ms": baseline_ms,
            "student_fixed_peak_memory_bytes": fixed_peak_memory,
            "student_baseline_peak_memory_bytes": torch.cuda.max_memory_allocated(),
        }
    )


if __name__ == "__main__":
    main()
