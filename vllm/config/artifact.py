# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution artifacts."""

from typing import Literal

from pydantic import Field

from vllm.config.utils import config


@config
class ArtifactConfig:
    """Configuration for the local shared-memory artifact connector."""

    backend: Literal["shm"] = "shm"
    """Artifact delivery backend. PR4 intentionally supports SHM only."""

    shm_dir: str = "/dev/shm/vllm-artifacts"
    """Trusted root for immutable artifact objects."""

    max_shm_bytes: int = Field(default=8 << 30, gt=0)
    """Maximum retained artifact bytes for one engine and DP rank."""

    shm_ttl_seconds: int = Field(default=3600, gt=0)
    """Retention time for immutable artifact objects."""
