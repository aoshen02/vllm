# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare TransferQueue R3 delivery with SHM inline delivery."""

import argparse
import io
import json
import urllib.request
from typing import Any

import numpy as np
import pybase64 as base64

from vllm.distributed.artifact_connector import (
    TransferQueueArtifactStore,
    materialize_routed_experts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-queue-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--shm-url", default="http://127.0.0.1:18001/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument("--store-id", default="transfer-queue")
    parser.add_argument("--data-partition", default="vllm-artifact-data")
    parser.add_argument("--request-partition", default="vllm-artifact-requests")
    return parser.parse_args()


def request_completion(
    base_url: str,
    *,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/completions",
        data=json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0,
                "seed": 0,
                "return_token_ids": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.load(response)


def decode_inline_r3(response: dict[str, Any]) -> np.ndarray:
    choice = response["choices"][0]
    encoded = choice.get("routed_experts")
    if not isinstance(encoded, str):
        raise RuntimeError("SHM response did not contain routed_experts")
    if choice.get("artifact_sample_id") is not None:
        raise RuntimeError("SHM response unexpectedly contained a sample ID")
    return np.load(io.BytesIO(base64.b64decode(encoded)), allow_pickle=False)


def main() -> None:
    args = parse_args()
    tq_response = request_completion(
        args.transfer_queue_url,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    shm_response = request_completion(
        args.shm_url,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
    )
    tq_choice = tq_response["choices"][0]
    shm_choice = shm_response["choices"][0]
    if tq_choice["token_ids"] != shm_choice["token_ids"]:
        raise RuntimeError("the two servers generated different token IDs")
    sample_id = tq_choice.get("artifact_sample_id")
    if not isinstance(sample_id, str):
        raise RuntimeError("TransferQueue response is missing artifact_sample_id")
    if tq_choice.get("routed_experts") is not None:
        raise RuntimeError("TransferQueue unexpectedly returned routed_experts")

    store = TransferQueueArtifactStore(
        ray_address=args.ray_address,
        store_id=args.store_id,
        data_partition=args.data_partition,
        request_partition=args.request_partition,
    )
    try:
        transfer_queue_r3 = materialize_routed_experts(store, sample_id)
        shm_r3 = decode_inline_r3(shm_response)
        np.testing.assert_array_equal(transfer_queue_r3, shm_r3)
    finally:
        store.close()

    print(
        json.dumps(
            {
                "artifact_sample_id": sample_id,
                "shape": list(transfer_queue_r3.shape),
                "dtype": transfer_queue_r3.dtype.str,
                "equal": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
