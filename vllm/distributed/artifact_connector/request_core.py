# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Backend-independent request lifecycle and object format."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vllm.distributed.artifact_connector.protocol import (
    ArtifactCommitRequest,
    ArtifactFinalizeRequest,
)
from vllm.distributed.artifact_connector.store import (
    ArtifactCorruptionError,
    ArtifactObject,
    ArtifactReader,
    ArtifactStore,
)

if TYPE_CHECKING:
    from vllm.distributed.artifact_connector.buffer import (
        RoutedExpertsArtifactBuffer,
    )

_SCHEMA_VERSION = 3
_FIELD_NAME = "routed_experts"
_HEADER_LENGTH = struct.Struct("<I")
_MAX_HEADER_BYTES = 4096


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(*parts: bytes) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(len(part).to_bytes(8, "big"))
        digest.update(part)
    return digest.hexdigest()


def encode_artifact_array(
    *,
    key: str,
    kind: str,
    field_profile_id: str,
    array: np.ndarray,
    source_token_start: int,
) -> bytes:
    """Encode one self-describing immutable array object."""
    contiguous = np.ascontiguousarray(array)
    raw = contiguous.tobytes(order="C")
    header = {
        "schema_version": _SCHEMA_VERSION,
        "key": key,
        "kind": kind,
        "field": _FIELD_NAME,
        "field_profile_id": field_profile_id,
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "source_token_start": source_token_start,
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }
    encoded_header = _canonical_json(header)
    if len(encoded_header) > _MAX_HEADER_BYTES:
        raise ValueError("artifact object header is too large")
    return _HEADER_LENGTH.pack(len(encoded_header)) + encoded_header + raw


def decode_artifact_array(
    payload: bytes,
    *,
    expected_key: str,
    expected_profile_id: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode and validate one self-describing immutable array object."""
    if len(payload) < _HEADER_LENGTH.size:
        raise ArtifactCorruptionError("artifact object is truncated")
    (header_length,) = _HEADER_LENGTH.unpack(payload[: _HEADER_LENGTH.size])
    if header_length <= 0 or header_length > _MAX_HEADER_BYTES:
        raise ArtifactCorruptionError("invalid artifact header length")
    header_end = _HEADER_LENGTH.size + header_length
    if header_end > len(payload):
        raise ArtifactCorruptionError("artifact header is truncated")
    try:
        header = json.loads(payload[_HEADER_LENGTH.size : header_end])
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ArtifactCorruptionError("invalid artifact object header") from error
    if not isinstance(header, dict):
        raise ArtifactCorruptionError("artifact object header must be an object")
    if (
        header.get("schema_version") != _SCHEMA_VERSION
        or header.get("key") != expected_key
        or header.get("kind") not in ("block", "tail")
        or header.get("field") != _FIELD_NAME
    ):
        raise ArtifactCorruptionError("artifact object identity mismatch")
    if (
        expected_profile_id is not None
        and header.get("field_profile_id") != expected_profile_id
    ):
        raise ArtifactCorruptionError("artifact field profile mismatch")

    raw = payload[header_end:]
    if header.get("payload_sha256") != hashlib.sha256(raw).hexdigest():
        raise ArtifactCorruptionError("artifact payload checksum mismatch")
    try:
        dtype = np.dtype(header["dtype"])
        shape = tuple(int(value) for value in header["shape"])
        source_token_start = int(header["source_token_start"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactCorruptionError("invalid artifact array metadata") from error
    if (
        not shape
        or any(dimension < 0 for dimension in shape)
        or dtype.kind not in "biuf"
        or shape[0] <= 0
        or source_token_start < 0
        or math.prod(shape) * dtype.itemsize != len(raw)
    ):
        raise ArtifactCorruptionError("invalid artifact array shape")
    try:
        array = np.frombuffer(raw, dtype=dtype).reshape(shape)
    except (TypeError, ValueError) as error:
        raise ArtifactCorruptionError("invalid artifact array payload") from error
    return array, header


class ArtifactKeySpace:
    """Derive backend-independent immutable keys for one artifact field."""

    def __init__(
        self,
        dtype: np.dtype[Any],
        shape_per_token: tuple[int, ...],
    ) -> None:
        self.dtype = np.dtype(dtype)
        self.shape_per_token = shape_per_token

    def profile_id(self, hash_block_size: int, weight_version: str) -> str:
        profile = {
            "schema_version": _SCHEMA_VERSION,
            "field": _FIELD_NAME,
            "dtype": self.dtype.str,
            "shape_per_token": list(self.shape_per_token),
            "hash_block_size": hash_block_size,
            "weight_version": weight_version,
        }
        return hashlib.sha256(_canonical_json(profile)).hexdigest()

    def block_key(
        self, block_hash: bytes, hash_block_size: int, weight_version: str
    ) -> str:
        profile_id = self.profile_id(hash_block_size, weight_version)
        digest = _digest(
            b"vllm.artifact.field-block.v3",
            profile_id.encode(),
            block_hash,
        )
        return f"vllm-artifact/{_FIELD_NAME}/block/{digest}"

    def tail_key(
        self,
        *,
        request_id: str,
        request_attempt_id: str,
        source_start: int,
        source_end: int,
        hash_block_size: int,
        weight_version: str,
    ) -> str:
        digest = _digest(
            b"vllm.artifact.field-tail.v3",
            self.profile_id(hash_block_size, weight_version).encode(),
            request_id.encode(),
            request_attempt_id.encode(),
            source_start.to_bytes(8, "big"),
            source_end.to_bytes(8, "big"),
        )
        return f"vllm-artifact/{_FIELD_NAME}/tail/{digest}"


def materialize_routed_experts(
    store: ArtifactReader,
    artifact_keys: list[str],
    *,
    expected_profile_id: str | None = None,
) -> np.ndarray:
    """Read ordered keys and materialize complete routed-experts rows."""
    if not artifact_keys:
        raise ValueError("routed-experts artifact key list must not be empty")
    payloads = store.get(artifact_keys)
    if len(payloads) != len(artifact_keys):
        raise ArtifactCorruptionError(
            "artifact backend returned the wrong object count"
        )
    arrays: list[np.ndarray] = []
    next_start = 0
    shape_per_token: tuple[int, ...] | None = None
    dtype: np.dtype[Any] | None = None
    profile_id: str | None = expected_profile_id
    for key, payload in zip(artifact_keys, payloads, strict=True):
        array, header = decode_artifact_array(
            payload,
            expected_key=key,
            expected_profile_id=profile_id,
        )
        current_profile = str(header["field_profile_id"])
        if profile_id is None:
            profile_id = current_profile
        elif current_profile != profile_id:
            raise ArtifactCorruptionError("artifact key list mixes field profiles")
        source_start = int(header["source_token_start"])
        if source_start != next_start:
            raise ArtifactCorruptionError(
                "artifact keys do not cover one contiguous logical range"
            )
        if shape_per_token is None:
            shape_per_token = array.shape[1:]
            dtype = array.dtype
        elif array.shape[1:] != shape_per_token or array.dtype != dtype:
            raise ArtifactCorruptionError("artifact key list mixes array schemas")
        next_start += array.shape[0]
        arrays.append(array)
    return np.concatenate(arrays, axis=0)


@dataclass(frozen=True)
class PreparedCommit:
    request_id: str
    block_end: int
    objects: list[ArtifactObject]


class ArtifactRequestCore:
    """Encode immutable R3 objects and produce terminal key lists."""

    def __init__(
        self,
        store: ArtifactStore,
        source: RoutedExpertsArtifactBuffer,
    ) -> None:
        self.store = store
        self.source = source
        self.key_space = ArtifactKeySpace(
            source.dtype,
            source.shape_per_token,
        )

    def prepare_commit(self, request: ArtifactCommitRequest) -> PreparedCommit:
        if (
            request.block_start < 0
            or request.block_start % request.hash_block_size
            or not request.block_hashes
        ):
            raise ValueError("invalid artifact full-block commit")
        block_end = request.block_start + (
            len(request.block_hashes) * request.hash_block_size
        )
        field_profile_id = self.key_space.profile_id(
            request.hash_block_size, request.weight_version
        )
        array = self.source.read(
            request.request_id,
            request.block_start,
            block_end,
        )
        if (
            array.dtype != self.source.dtype
            or array.shape[1:] != self.source.shape_per_token
        ):
            raise RuntimeError("routed-experts capture profile changed")

        objects: list[ArtifactObject] = []
        for block_offset, block_hash in enumerate(request.block_hashes):
            source_start = request.block_start + block_offset * request.hash_block_size
            key = self.key_space.block_key(
                block_hash,
                request.hash_block_size,
                request.weight_version,
            )
            local_start = source_start - request.block_start
            block_array = array[local_start : local_start + request.hash_block_size]
            objects.append(
                ArtifactObject(
                    key=key,
                    payload=encode_artifact_array(
                        key=key,
                        kind="block",
                        field_profile_id=field_profile_id,
                        array=block_array,
                        source_token_start=source_start,
                    ),
                )
            )
        return PreparedCommit(
            request_id=request.request_id,
            block_end=block_end,
            objects=objects,
        )

    def publish_commits(self, commits: list[PreparedCommit]) -> None:
        objects = [obj for commit in commits for obj in commit.objects]
        self.store.put(objects)
        for commit in commits:
            self.source.release_through(commit.request_id, commit.block_end)

    def _put_tail(
        self,
        request: ArtifactFinalizeRequest,
        *,
        field_profile_id: str,
        source_start: int,
        source_end: int,
    ) -> str:
        try:
            array = self.source.read(
                request.request_id,
                source_start,
                source_end,
            )
        except RuntimeError as buffer_error:
            # Frontend stop matching can finalize behind async scheduler
            # progress. In that case this partial range was released from the
            # worker buffer after its containing full block became immutable.
            block_index = source_start // request.hash_block_size
            if block_index >= len(request.block_hashes):
                raise
            block_hash = request.block_hashes[block_index]
            block_key = self.key_space.block_key(
                block_hash,
                request.hash_block_size,
                request.weight_version,
            )
            payloads = self.store.get([block_key])
            if len(payloads) != 1:
                raise RuntimeError(
                    "artifact backend returned the wrong object count"
                ) from buffer_error
            block, header = decode_artifact_array(
                payloads[0],
                expected_key=block_key,
                expected_profile_id=field_profile_id,
            )
            if (
                header["kind"] != "block"
                or header["source_token_start"] != source_start
                or block.shape[0] != request.hash_block_size
                or source_end - source_start >= request.hash_block_size
            ):
                raise ArtifactCorruptionError(
                    "artifact block cannot reconstruct the terminal tail"
                ) from buffer_error
            array = block[: source_end - source_start]
        key = self.key_space.tail_key(
            request_id=request.request_id,
            request_attempt_id=request.request_attempt_id,
            source_start=source_start,
            source_end=source_end,
            hash_block_size=request.hash_block_size,
            weight_version=request.weight_version,
        )
        self.store.put(
            [
                ArtifactObject(
                    key=key,
                    payload=encode_artifact_array(
                        key=key,
                        kind="tail",
                        field_profile_id=field_profile_id,
                        array=array,
                        source_token_start=source_start,
                    ),
                )
            ]
        )
        return key

    def finalize(self, request: ArtifactFinalizeRequest) -> list[str]:
        if request.token_end <= 0:
            raise ValueError(f"invalid artifact token range: [0, {request.token_end})")
        field_profile_id = self.key_space.profile_id(
            request.hash_block_size, request.weight_version
        )
        full_end = (
            request.token_end // request.hash_block_size * request.hash_block_size
        )
        full_block_count = full_end // request.hash_block_size
        if len(request.block_hashes) < full_block_count:
            raise RuntimeError(
                "terminal artifact is missing KV-compatible block hashes: "
                f"request={request.request_id}, expected={full_block_count}, "
                f"actual={len(request.block_hashes)}"
            )
        keys = [
            self.key_space.block_key(
                request.block_hashes[block_index],
                request.hash_block_size,
                request.weight_version,
            )
            for block_index in range(full_block_count)
        ]
        if full_end < request.token_end:
            keys.append(
                self._put_tail(
                    request,
                    field_profile_id=field_profile_id,
                    source_start=full_end,
                    source_end=request.token_end,
                )
            )

        self.source.discard(request.request_id)
        return keys
