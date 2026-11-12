# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Format of shared auxiliary-output objects and the reader that assembles rows."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence

import numpy as np
from safetensors.numpy import load

from vllm.distributed.aux_output_connector.object_store import (
    AuxObjectStore,
    AuxObjectStoreError,
)

AUX_OBJECT_FORMAT = "vllm-aux-object-v1"
_SAFETENSORS_DTYPES = {
    "bool": "BOOL",
    "uint8": "U8",
    "int8": "I8",
    "uint16": "U16",
    "int16": "I16",
    "float16": "F16",
    "uint32": "U32",
    "int32": "I32",
    "float32": "F32",
    "uint64": "U64",
    "int64": "I64",
    "float64": "F64",
}


def encode_aux_object(fields: Mapping[str, np.ndarray]) -> memoryview:
    """Serialize the rows of one block, one array per output field, in the
    safetensors format. The rows are copied once.

    Every field has one row per position of the block.
    """
    if not fields:
        raise ValueError("an auxiliary-output object needs at least one field")
    num_rows = {len(rows) for rows in fields.values()}
    if len(num_rows) != 1:
        raise ValueError(f"fields have different row counts {sorted(num_rows)}")
    header: dict[str, object] = {"__metadata__": {"format": AUX_OBJECT_FORMAT}}
    size = 0
    for name, rows in fields.items():
        header[name] = {
            "dtype": _SAFETENSORS_DTYPES[rows.dtype.name],
            "shape": list(rows.shape),
            "data_offsets": [size, size + rows.nbytes],
        }
        size += rows.nbytes
    encoded_header = json.dumps(header, separators=(",", ":")).encode()
    encoded_header += b" " * (-len(encoded_header) % 8)
    start = 8 + len(encoded_header)
    data = np.empty(start + size, dtype=np.uint8)
    data[:8] = np.frombuffer(len(encoded_header).to_bytes(8, "little"), np.uint8)
    data[8:start] = np.frombuffer(encoded_header, np.uint8)
    for rows in fields.values():
        data[start : start + rows.nbytes] = (
            np.ascontiguousarray(rows).view(np.uint8).reshape(-1)
        )
        start += rows.nbytes
    return memoryview(data)


def decode_aux_object(data: bytes | memoryview) -> dict[str, np.ndarray]:
    """Inverse of `encode_aux_object`."""
    data = bytes(data)
    header_len = int.from_bytes(data[:8], "little")
    metadata = json.loads(data[8 : 8 + header_len]).get("__metadata__", {})
    if metadata.get("format") != AUX_OBJECT_FORMAT:
        raise AuxObjectStoreError(
            f"unknown auxiliary-output object format {metadata.get('format')!r}"
        )
    return load(data)


def read_aux_rows(
    store: AuxObjectStore,
    keys: Sequence[str],
    field: str,
    tail: np.ndarray | None = None,
    timeout_s: float = 60.0,
    poll_interval_s: float = 0.05,
) -> np.ndarray:
    """Concatenate `field` from the objects `keys`, in order, then `tail`.

    An object the engine is still writing is waited for, up to `timeout_s`
    in total; objects still missing then raise `AuxObjectStoreError`.
    """
    objects: dict[str, bytes] = {}
    deadline = time.monotonic() + timeout_s
    while True:
        for key in keys:
            if key not in objects and (data := store.get(key)) is not None:
                objects[key] = data
        missing = [key for key in keys if key not in objects]
        if not missing:
            break
        if time.monotonic() >= deadline:
            raise AuxObjectStoreError(
                f"{len(missing)} of {len(keys)} auxiliary-output objects are "
                f"missing after {timeout_s}s, first {missing[0]!r}"
            )
        time.sleep(poll_interval_s)
    parts = []
    for key in keys:
        fields = decode_aux_object(objects[key])
        if field not in fields:
            raise AuxObjectStoreError(f"object {key!r} has no field {field!r}")
        parts.append(fields[field])
    if tail is not None:
        parts.append(np.asarray(tail))
    if not parts:
        raise ValueError("no objects and no tail to read")
    if len({(part.dtype, part.shape[1:]) for part in parts}) != 1:
        raise AuxObjectStoreError(
            f"field {field!r} has different dtypes or row shapes across objects"
        )
    return np.concatenate(parts)
