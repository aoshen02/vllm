# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a TransferQueue SimpleStorage deployment for ArtifactConnector."""

import argparse
import signal
import threading
from pathlib import Path

import ray
import transfer_queue as tq
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray-address", default="auto")
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional TransferQueue config; defaults to SimpleStorage.",
    )
    parser.add_argument("--num-storage-units", type=int, default=2)
    parser.add_argument("--total-storage-size", type=int)
    return parser.parse_args()


def load_config(args: argparse.Namespace):
    if args.config is not None:
        return OmegaConf.load(args.config)
    if args.num_storage_units <= 0:
        raise ValueError("--num-storage-units must be positive")
    return OmegaConf.create(
        {
            "controller": {"polling_mode": True},
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": {
                    "total_storage_size": args.total_storage_size,
                    "num_data_storage_units": args.num_storage_units,
                },
            },
        }
    )


def main() -> None:
    args = parse_args()
    ray.init(address=args.ray_address, namespace="transfer_queue")
    config = load_config(args)
    tq.init(config)

    stopped = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stopped.set()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(
        f"TransferQueue is ready (backend={config.backend.storage_backend})",
        flush=True,
    )
    stopped.wait()
    tq.close()
    ray.shutdown()


if __name__ == "__main__":
    main()
