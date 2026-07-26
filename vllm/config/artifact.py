# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for execution artifacts."""

from typing import Literal

from pydantic import Field, model_validator

from vllm.config.utils import config


@config
class ArtifactConfig:
    """Configuration for execution-artifact storage."""

    backend: Literal["shm", "transfer_queue"] = "shm"
    """Artifact delivery backend."""

    shm_dir: str = "/dev/shm/vllm-artifacts"
    """Trusted root for immutable artifact objects."""

    max_shm_bytes: int = Field(default=8 << 30, gt=0)
    """Maximum retained artifact bytes for one engine and DP rank."""

    shm_ttl_seconds: int = Field(default=3600, gt=0)
    """Retention time for completed artifact samples."""

    tq_ray_address: str = "auto"
    """Ray cluster containing an existing TransferQueue deployment."""

    tq_store_id: str = "transfer-queue"
    """Stable identity for this TransferQueue artifact namespace."""

    tq_data_partition: str = "vllm-artifact-data"
    """TransferQueue partition containing block and tail payloads."""

    tq_request_partition: str = "vllm-artifact-requests"
    """TransferQueue partition containing finalized request manifests."""

    tq_connect_timeout_seconds: float = Field(default=30.0, ge=0)
    """Maximum time to wait for an existing TransferQueue deployment."""

    @model_validator(mode="after")
    def validate_transfer_queue(self) -> "ArtifactConfig":
        if self.backend != "transfer_queue":
            return self
        for field_name in (
            "tq_ray_address",
            "tq_store_id",
            "tq_data_partition",
            "tq_request_partition",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(
                    f"artifact_config.{field_name} must not be empty "
                    "for the TransferQueue backend"
                )
        return self
