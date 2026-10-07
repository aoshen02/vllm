"""Construct engine logprob arrays; never construct an HTTP response body."""

import json
from pathlib import Path

import numpy as np


def load_token_pool(path: Path) -> np.ndarray:
    entries = json.loads(path.read_text())
    ids = [entry["id"] for entry in entries]
    decoded = [entry["decoded"] for entry in entries]
    if len(set(ids)) != len(ids) or len(set(decoded)) != len(decoded):
        raise ValueError("Token pool must contain distinct IDs and decoded strings")
    return np.asarray(ids, dtype=np.int64)


def make_chunk(pool: np.ndarray, start: int, count: int, top_k: int = 128):
    """Return sampled IDs and the actual LogprobsLists constituent arrays.

    Engine payloads have a sampled-token slot followed by top-k slots. The
    sampled token belongs to the top-k here, so the wire has k+1 slots but
    exactly k distinct candidates per position, as in a real sampled output.
    """
    if count < 1 or start < 0 or not 1 <= top_k <= len(pool):
        raise ValueError("Invalid chunk bounds or top-k")
    positions = np.arange(start, start + count, dtype=np.int64)
    candidate_indices = (positions[:, None] + np.arange(top_k, dtype=np.int64)) % len(
        pool
    )
    top_ids = pool[candidate_indices]
    sampled_indices = positions % top_k
    sampled = top_ids[np.arange(count), sampled_indices]
    token_ids = np.column_stack((sampled, top_ids))

    scores = -np.arange(top_k, dtype=np.float64) * 0.037
    scores -= np.log(np.exp(scores).sum())
    top_scores = np.broadcast_to(scores.astype(np.float32), (count, top_k))
    logprobs = np.column_stack(
        (top_scores[np.arange(count), sampled_indices], top_scores)
    )
    ranks = sampled_indices + 1
    return sampled.tolist(), token_ids, logprobs, ranks


def make_routed(row_start: int, rows: int, layers: int, topk: int = 8):
    """Deterministic uint8 expert ids, shape (rows, layers, topk); distinct per row/layer."""
    t = np.arange(row_start, row_start + rows, dtype=np.int64)[:, None, None]
    lyr = np.arange(layers, dtype=np.int64)[None, :, None]
    j = np.arange(topk, dtype=np.int64)[None, None, :]
    return ((t * 7 + lyr * 13 + j * 31) % 256).astype(np.uint8)


def make_engine_output(
    request_id: str,
    pool: np.ndarray,
    start: int,
    count: int,
    routed_layers: int = 0,
    prompt_tokens: int = 0,
):
    """Use the installed vLLM types and encoder at the engine boundary.

    routed_experts mirrors the vLLM scheduler: the first output carries the
    prompt routing (prompt_tokens rows) plus rows for the tokens forwarded in
    this chunk; later outputs carry one row per forwarded token. Total rows
    after N output tokens: prompt_tokens + N - 1.
    """
    from vllm.v1.engine import EngineCoreOutput
    from vllm.v1.outputs import LogprobsLists

    sampled, token_ids, logprobs, ranks = make_chunk(pool, start, count)
    routed = None
    if routed_layers:
        if start == 0:
            routed = make_routed(0, prompt_tokens + count - 1, routed_layers)
        else:
            routed = make_routed(prompt_tokens + start - 1, count, routed_layers)
    return EngineCoreOutput(
        request_id=request_id,
        new_token_ids=sampled,
        new_logprobs=LogprobsLists(token_ids, logprobs, ranks),
        routed_experts=routed,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("token_pool", type=Path)
    args = parser.parse_args()
    pool = load_token_pool(args.token_pool)
    samples, ids, scores, ranks = make_chunk(pool, 117, 257)
    assert ids.shape == scores.shape == (257, 129)
    assert scores.dtype == np.float32
    for index, sample in enumerate(samples):
        assert len(set(ids[index].tolist())) == 128
        rank = int(ranks[index])
        assert ids[index, rank] == sample == ids[index, 0]
        assert scores[index, rank] == scores[index, 0]
    print(
        json.dumps(
            {
                "positions": len(samples),
                "distinct_candidates": 128,
                "wire_slots": 129,
                "status": "payload-only PASS",
            }
        )
    )
