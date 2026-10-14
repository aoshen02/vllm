# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal, TypeAlias

import msgspec

from vllm.v1.serial_utils import PydanticMsgspecMixin

MetadataValue: TypeAlias = str | int | list[int]

NUMERIC_OBJECT_BYTES = 4 * 1024**2
DEFAULT_STAGING_BUDGET_BYTES = 128 * 1024**2


class AuxOutputRef(PydanticMsgspecMixin, msgspec.Struct):
    backend: Literal["mooncake"]
    manifest_key: str


class AuxChunkRef(msgspec.Struct):
    num_rows: int
    keys: dict[str, str]


class PayloadRef(msgspec.Struct):
    token_start: int
    num_rows: int
    metadata: dict[str, MetadataValue]
    chunks: list[AuxChunkRef]


class AuxOutputManifest(msgspec.Struct):
    payloads: dict[str, PayloadRef]
