# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Versioned, token-aligned auxiliary-output objects."""

from dataclasses import dataclass
from enum import IntEnum, IntFlag
from struct import Struct
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

RoutingArray: TypeAlias = NDArray[np.uint8] | NDArray[np.uint16]
RoutingArrayDtype: TypeAlias = np.dtype[np.uint8] | np.dtype[np.uint16]

HEADER = Struct("<8sHHIQIIHBBI")
MAGIC = b"VLLMAUX\0"
VERSION = 1
MAX_OBJECT_BYTES = 64 * 1024**2
R3_KEY_PREFIX = "vllm-aux/v1/r3"
OUTPUT_KEY_PREFIX = "vllm-aux/v1/output"


class AuxField(IntFlag):
    ROUTED_EXPERTS = 1
    TOKEN_LOGPROBS = 2


class RoutingDtypeCode(IntEnum):
    UINT8 = 1
    UINT16 = 2


@dataclass(frozen=True)
class AuxBlock:
    token_start: int
    routed_experts: RoutingArray

    @property
    def num_rows(self) -> int:
        return len(self.routed_experts)

    @property
    def fields(self) -> dict[str, RoutingArray]:
        return {"routed_experts": self.routed_experts}


def _validate_dimensions(start: int, rows: int, layers: int, experts: int) -> None:
    if type(start) is not int or not 0 <= start < 2**64:
        raise ValueError("Invalid auxiliary token start")
    if not 0 < rows < 2**32 or start + rows >= 2**64:
        raise ValueError("Invalid auxiliary token range")
    if not 0 < layers < 2**32 or not 0 < experts < 2**16:
        raise ValueError("Invalid routed-experts shape")


def encode_parts(block: AuxBlock) -> tuple[bytes, ...]:
    rows = block.routed_experts
    if rows.ndim != 3 or rows.dtype.kind != "u" or rows.dtype.itemsize not in (1, 2):
        raise ValueError("Routed experts must be a uint8 or uint16 [T,L,E] array")
    count, layers, experts = rows.shape
    _validate_dimensions(block.token_start, count, layers, experts)
    if HEADER.size + rows.nbytes > MAX_OBJECT_BYTES:
        raise ValueError("Auxiliary object exceeds size limit")
    dtype = np.dtype("u1" if rows.dtype.itemsize == 1 else "<u2")
    header = HEADER.pack(
        MAGIC,
        VERSION,
        AuxField.ROUTED_EXPERTS,
        HEADER.size,
        block.token_start,
        count,
        layers,
        experts,
        RoutingDtypeCode.UINT8 if dtype.itemsize == 1 else RoutingDtypeCode.UINT16,
        0,
        0,
    )
    return header, rows.astype(dtype, copy=False).tobytes(order="C")


def decode_block(raw: bytes) -> AuxBlock:
    if not HEADER.size <= len(raw) <= MAX_OBJECT_BYTES:
        raise ValueError("Invalid auxiliary object size")
    magic, version, mask, size, start, count, layers, experts, code, mode, k = (
        HEADER.unpack_from(raw)
    )
    if magic != MAGIC or version != VERSION or size != HEADER.size:
        raise ValueError("Unsupported auxiliary object header")
    if mask != AuxField.ROUTED_EXPERTS or mode != 0 or k != 0:
        raise ValueError(
            "Unsupported auxiliary fields; only routed experts are enabled"
        )
    _validate_dimensions(start, count, layers, experts)
    if code not in (RoutingDtypeCode.UINT8, RoutingDtypeCode.UINT16):
        raise ValueError("Unsupported routed-experts dtype")
    dtype = np.dtype("u1" if code == RoutingDtypeCode.UINT8 else "<u2")
    if len(raw) != HEADER.size + count * layers * experts * dtype.itemsize:
        raise ValueError("Auxiliary object length does not match its shape")
    rows = np.frombuffer(raw, dtype=dtype, offset=HEADER.size).reshape(
        count, layers, experts
    )
    return AuxBlock(start, cast(RoutingArray, rows))
