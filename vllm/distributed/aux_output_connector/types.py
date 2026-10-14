# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Annotated, TypeAlias

import msgspec
import numpy as np
import numpy.typing as npt

from vllm.v1.serial_utils import PydanticMsgspecMixin

RoutingArray: TypeAlias = npt.NDArray[np.uint8] | npt.NDArray[np.uint16]
RoutingDtype: TypeAlias = np.dtype[np.uint8] | np.dtype[np.uint16]
MetadataValue: TypeAlias = str | int
RowCount: TypeAlias = Annotated[int, msgspec.Meta(ge=0)]


class AuxChunkRef(PydanticMsgspecMixin, msgspec.Struct):
    num_rows: RowCount
    key: str

    def __post_init__(self) -> None:
        if type(self.num_rows) is not int or self.num_rows <= 0:
            raise ValueError("Invalid auxiliary chunk row count")
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("Invalid auxiliary object key")


class AuxOutputPayload(PydanticMsgspecMixin, msgspec.Struct):
    token_start: RowCount
    num_rows: RowCount
    metadata: dict[str, MetadataValue]
    chunks: list[AuxChunkRef]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if (
            type(self.token_start) is not int
            or self.token_start < 0
            or type(self.num_rows) is not int
            or self.num_rows < 0
        ):
            raise ValueError(
                f"Invalid auxiliary token range: "
                f"{self.token_start!r}, {self.num_rows!r}"
            )
        if not isinstance(self.chunks, list):
            raise ValueError("Auxiliary chunks must be a list")
        covered = 0
        for chunk in self.chunks:
            chunk.__post_init__()
            if covered >= self.num_rows:
                raise ValueError("Auxiliary chunks exceed the visible prefix")
            covered += chunk.num_rows
        if covered < self.num_rows:
            raise ValueError("Auxiliary chunks do not cover the requested rows")


AuxOutputs: TypeAlias = dict[str, AuxOutputPayload]


@dataclass
class AuxOutputResult:
    routed_experts: RoutingArray | None = None
    aux_outputs: AuxOutputs | None = None


def validate_aux_outputs(value: AuxOutputs) -> AuxOutputs:
    result = msgspec.convert(value, type=AuxOutputs)
    for name, payload in result.items():
        payload.validate()
        if name == "routed_experts":
            routing_layout(payload.metadata)
    return result


def routing_layout(
    metadata: dict[str, MetadataValue],
) -> tuple[RoutingDtype, tuple[int, int]]:
    name = metadata.get("dtype")
    layers = metadata.get("num_layers")
    experts = metadata.get("experts_per_token")
    if (
        name not in ("|u1", "<u2")
        or type(layers) is not int
        or type(experts) is not int
        or layers <= 0
        or experts <= 0
    ):
        raise ValueError("Invalid routed-experts layout")
    dtype: RoutingDtype = (
        np.dtype(np.uint8) if name == "|u1" else np.dtype(np.uint16).newbyteorder("<")
    )
    return dtype, (layers, experts)


def routing_metadata(rows: RoutingArray) -> dict[str, MetadataValue]:
    if rows.ndim != 3:
        raise ValueError("Routed experts must have three dimensions")
    metadata: dict[str, MetadataValue] = {
        "dtype": rows.dtype.str,
        "num_layers": rows.shape[1],
        "experts_per_token": rows.shape[2],
    }
    routing_layout(metadata)
    return metadata


def append_aux_outputs(accumulated: AuxOutputs, delta: AuxOutputs) -> None:
    for name, payload in delta.items():
        payload.validate()
        if sum(chunk.num_rows for chunk in payload.chunks) != payload.num_rows:
            raise ValueError("Auxiliary deltas must describe physical rows")
        if name == "routed_experts":
            routing_layout(payload.metadata)
        previous = accumulated.get(name)
        if previous is None:
            accumulated[name] = AuxOutputPayload(
                payload.token_start,
                payload.num_rows,
                dict(payload.metadata),
                [AuxChunkRef(chunk.num_rows, chunk.key) for chunk in payload.chunks],
            )
        else:
            if (
                previous.metadata != payload.metadata
                or previous.token_start + previous.num_rows != payload.token_start
            ):
                raise ValueError("Inconsistent auxiliary output delta")
            previous.chunks.extend(
                AuxChunkRef(chunk.num_rows, chunk.key) for chunk in payload.chunks
            )
            previous.num_rows += payload.num_rows
