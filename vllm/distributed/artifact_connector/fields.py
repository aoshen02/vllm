# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Field descriptions for execution artifacts."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactField:
    """Immutable schema facts that differ between artifact fields."""

    name: str
    reuse_policy: str
    logical_coordinate: str


ROUTED_EXPERTS = ArtifactField(
    name="routed_experts",
    reuse_policy="PREFIX_BLOCK",
    logical_coordinate="EXECUTED_TOKEN",
)
