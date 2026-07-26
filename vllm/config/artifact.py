# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution artifacts."""

from typing import Literal

from pydantic import Field

from vllm.config.utils import config


@config
class ArtifactConfig:
    """Configuration for the execution-artifact connector."""

    backend: Literal["shm", "mooncake"] = "shm"
    """Artifact storage and delivery backend."""

    shm_dir: str = "/dev/shm/vllm-artifacts"
    """Trusted root for immutable artifact objects."""

    max_shm_bytes: int = Field(default=8 << 30, gt=0)
    """Maximum retained artifact bytes for one engine and DP rank."""

    shm_ttl_seconds: int = Field(default=3600, gt=0)
    """Retention time for immutable artifact objects."""

    mooncake_store_id: str = Field(
        default="vllm-r3",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    """Stable deployment namespace used in public Mooncake object keys."""

    mooncake_staging_buffer_bytes: int = Field(default=64 << 20, gt=0)
    """Registered CPU bytes reserved by each R3 writer for Mooncake I/O."""
