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

from vllm.distributed.artifact_connector.fields import ROUTED_EXPERTS, ArtifactField
from vllm.distributed.artifact_connector.protocol import (
    ArtifactCommitRequest,
    ArtifactFinalizeRequest,
)
from vllm.distributed.artifact_connector.store import (
    ArtifactCorruptionError,
    ArtifactObject,
    ArtifactStore,
)

if TYPE_CHECKING:
    from vllm.distributed.artifact_connector.buffer import (
        RoutedExpertsArtifactBuffer,
    )

_SCHEMA_VERSION = 3
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
    field_spec: ArtifactField,
    field_profile_id: str,
    array: np.ndarray,
    source_token_start: int,
    valid_len: int,
    kv_block_hash: bytes | None = None,
) -> bytes:
    """Encode one self-describing immutable array object."""
    contiguous = np.ascontiguousarray(array)
    raw = contiguous.tobytes(order="C")
    header = {
        "schema_version": _SCHEMA_VERSION,
        "key": key,
        "kind": kind,
        "field": field_spec.name,
        "reuse_policy": field_spec.reuse_policy,
        "logical_coordinate": field_spec.logical_coordinate,
        "field_profile_id": field_profile_id,
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "source_token_start": source_token_start,
        "valid_len": valid_len,
        "kv_block_hash": kv_block_hash.hex() if kv_block_hash is not None else None,
        "payload_nbytes": len(raw),
        "payload_sha256": hashlib.sha256(raw).hexdigest(),
    }
    header["header_sha256"] = hashlib.sha256(_canonical_json(header)).hexdigest()
    encoded_header = _canonical_json(header)
    if len(encoded_header) > _MAX_HEADER_BYTES:
        raise ValueError("artifact object header is too large")
    return _HEADER_LENGTH.pack(len(encoded_header)) + encoded_header + raw


def decode_artifact_array(
    payload: bytes,
    *,
    expected_key: str,
    expected_field: ArtifactField = ROUTED_EXPERTS,
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
    header_sha256 = header.get("header_sha256")
    unsigned = dict(header)
    unsigned.pop("header_sha256", None)
    if header_sha256 != hashlib.sha256(_canonical_json(unsigned)).hexdigest():
        raise ArtifactCorruptionError("artifact header checksum mismatch")
    if (
        header.get("schema_version") != _SCHEMA_VERSION
        or header.get("key") != expected_key
        or header.get("kind") not in ("block", "tail")
        or header.get("field") != expected_field.name
        or header.get("reuse_policy") != expected_field.reuse_policy
        or header.get("logical_coordinate") != expected_field.logical_coordinate
    ):
        raise ArtifactCorruptionError("artifact object identity mismatch")
    if (
        expected_profile_id is not None
        and header.get("field_profile_id") != expected_profile_id
    ):
        raise ArtifactCorruptionError("artifact field profile mismatch")

    raw = payload[header_end:]
    if header.get("payload_nbytes") != len(raw):
        raise ArtifactCorruptionError("artifact payload size mismatch")
    if header.get("payload_sha256") != hashlib.sha256(raw).hexdigest():
        raise ArtifactCorruptionError("artifact payload checksum mismatch")
    try:
        dtype = np.dtype(header["dtype"])
        shape = tuple(int(value) for value in header["shape"])
        valid_len = int(header["valid_len"])
        source_token_start = int(header["source_token_start"])
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactCorruptionError("invalid artifact array metadata") from error
    if (
        not shape
        or any(dimension < 0 for dimension in shape)
        or dtype.kind not in "biuf"
        or valid_len <= 0
        or valid_len != shape[0]
        or source_token_start < 0
        or math.prod(shape) * dtype.itemsize != len(raw)
    ):
        raise ArtifactCorruptionError("invalid artifact array shape")
    try:
        array = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    except (TypeError, ValueError) as error:
        raise ArtifactCorruptionError("invalid artifact array payload") from error
    array.setflags(write=False)
    return array, header


def max_artifact_object_bytes(
    dtype: np.dtype[Any],
    shape_per_token: tuple[int, ...],
    block_size: int,
) -> int:
    """Return a safe registered-buffer capacity for one block object."""
    payload_bytes = block_size * math.prod(shape_per_token) * dtype.itemsize
    return _HEADER_LENGTH.size + _MAX_HEADER_BYTES + payload_bytes


def materialize_routed_experts(
    store: ArtifactStore,
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
    request: ArtifactCommitRequest
    objects: list[ArtifactObject]
    keys: list[str]


@dataclass(frozen=True)
class FinalizedArtifact:
    keys: list[str]
    value: np.ndarray


class ArtifactRequestCore:
    """Encode immutable R3 objects and materialize terminal artifacts."""

    def __init__(
        self,
        store: ArtifactStore,
        source: RoutedExpertsArtifactBuffer,
        *,
        namespace: str,
    ) -> None:
        self.store = store
        self.source = source
        self.namespace = namespace
        self.field_spec = ROUTED_EXPERTS

    def _profile_id(self, hash_block_size: int) -> str:
        profile = {
            "schema_version": _SCHEMA_VERSION,
            "field": self.field_spec.name,
            "namespace": self.namespace,
            "dtype": self.source.dtype.str,
            "shape_per_token": list(self.source.shape_per_token),
            "reuse_policy": self.field_spec.reuse_policy,
            "logical_coordinate": self.field_spec.logical_coordinate,
            "hash_block_size": hash_block_size,
        }
        return hashlib.sha256(_canonical_json(profile)).hexdigest()

    def block_key(
        self,
        *,
        block_hash: bytes,
        hash_block_size: int,
    ) -> str:
        profile_id = self._profile_id(hash_block_size)
        digest = _digest(
            b"vllm.artifact.field-block.v3",
            self.namespace.encode(),
            self.field_spec.name.encode(),
            profile_id.encode(),
            block_hash,
        )
        return (
            f"vllm-artifact/{self.store.store_id}/{self.field_spec.name}/block/{digest}"
        )

    def _tail_key(
        self,
        *,
        field_profile_id: str,
        request_id: str,
        request_attempt_id: str,
        source_start: int,
        source_end: int,
    ) -> str:
        digest = _digest(
            b"vllm.artifact.field-tail.v3",
            self.namespace.encode(),
            self.field_spec.name.encode(),
            field_profile_id.encode(),
            request_id.encode(),
            request_attempt_id.encode(),
            source_start.to_bytes(8, "big"),
            source_end.to_bytes(8, "big"),
        )
        return (
            f"vllm-artifact/{self.store.store_id}/{self.field_spec.name}/tail/{digest}"
        )

    def prepare_commit(self, request: ArtifactCommitRequest) -> PreparedCommit:
        if (
            request.block_start < 0
            or request.block_end <= request.block_start
            or request.block_start % request.hash_block_size
            or request.block_end % request.hash_block_size
        ):
            raise ValueError(
                "invalid artifact full-block range: "
                f"[{request.block_start}, {request.block_end})"
            )
        field_profile_id = self._profile_id(request.hash_block_size)
        array = self.source.read(
            request.request_id,
            request.block_start,
            request.block_end,
        )
        if (
            array.dtype != self.source.dtype
            or array.shape[1:] != self.source.shape_per_token
        ):
            raise RuntimeError("routed-experts capture profile changed")

        objects: list[ArtifactObject] = []
        keys: list[str] = []
        for source_start in range(
            request.block_start, request.block_end, request.hash_block_size
        ):
            block_index = source_start // request.hash_block_size
            if block_index >= len(request.block_hashes):
                raise RuntimeError(
                    "missing KV-compatible block hash for routed-experts artifact: "
                    f"request={request.request_id}, block_index={block_index}, "
                    f"num_hashes={len(request.block_hashes)}"
                )
            block_hash = request.block_hashes[block_index]
            key = self.block_key(
                block_hash=block_hash,
                hash_block_size=request.hash_block_size,
            )
            local_start = source_start - request.block_start
            block_array = array[local_start : local_start + request.hash_block_size]
            objects.append(
                ArtifactObject(
                    key=key,
                    payload=encode_artifact_array(
                        key=key,
                        kind="block",
                        field_spec=self.field_spec,
                        field_profile_id=field_profile_id,
                        array=block_array,
                        source_token_start=source_start,
                        valid_len=request.hash_block_size,
                        kv_block_hash=block_hash,
                    ),
                )
            )
            keys.append(key)
        return PreparedCommit(request=request, objects=objects, keys=keys)

    def publish_commits(self, commits: list[PreparedCommit]) -> dict[str, str | None]:
        objects = [obj for commit in commits for obj in commit.objects]
        results = self.store.put(objects)
        if len(results) != len(objects):
            raise RuntimeError("artifact store returned the wrong put result count")
        errors: dict[str, str | None] = {}
        offset = 0
        for commit in commits:
            commit_results = results[offset : offset + len(commit.objects)]
            offset += len(commit.objects)
            error = next(
                (result.error for result in commit_results if result.error),
                None,
            )
            errors[commit.request.operation_id] = error
            if error is None:
                self.source.release_through(
                    commit.request.request_id, commit.request.block_end
                )
        return errors

    def _put_tail(
        self,
        request: ArtifactFinalizeRequest,
        *,
        field_profile_id: str,
        source_start: int,
        source_end: int,
    ) -> str:
        array = self.source.read(
            request.request_id,
            source_start,
            source_end,
        )
        key = self._tail_key(
            field_profile_id=field_profile_id,
            request_id=request.request_id,
            request_attempt_id=request.request_attempt_id,
            source_start=source_start,
            source_end=source_end,
        )
        result = self.store.put(
            [
                ArtifactObject(
                    key=key,
                    payload=encode_artifact_array(
                        key=key,
                        kind="tail",
                        field_spec=self.field_spec,
                        field_profile_id=field_profile_id,
                        array=array,
                        source_token_start=source_start,
                        valid_len=source_end - source_start,
                    ),
                )
            ]
        )
        if len(result) != 1 or result[0].error is not None:
            message = result[0].error if result else "missing put result"
            raise RuntimeError(f"failed to publish artifact tail: {message}")
        return key

    def finalize(self, request: ArtifactFinalizeRequest) -> FinalizedArtifact:
        if request.token_end <= 0:
            raise ValueError(f"invalid artifact token range: [0, {request.token_end})")
        field_profile_id = self._profile_id(request.hash_block_size)
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
            self.block_key(
                block_hash=request.block_hashes[block_index],
                hash_block_size=request.hash_block_size,
            )
            for block_index in range(full_block_count)
        ]
        ready = self.store.exists(keys)
        if len(ready) != len(keys) or not all(ready):
            missing_block = next(
                (index for index, exists in enumerate(ready) if not exists),
                len(ready),
            )
            raise RuntimeError(
                "terminal artifact is missing a reusable full block: "
                f"request={request.request_id}, "
                f"block_start={missing_block * request.hash_block_size}"
            )
        if full_end < request.token_end:
            keys.append(
                self._put_tail(
                    request,
                    field_profile_id=field_profile_id,
                    source_start=full_end,
                    source_end=request.token_end,
                )
            )

        expected_key_count = (
            request.token_end + request.hash_block_size - 1
        ) // request.hash_block_size
        if len(keys) != expected_key_count:
            raise RuntimeError(
                "artifact key count does not match the terminal token range"
            )
        value = materialize_routed_experts(
            self.store,
            keys,
            expected_profile_id=field_profile_id,
        )
        self.source.discard(request.request_id)
        return FinalizedArtifact(keys=keys, value=value)

    def close(self) -> None:
        self.store.close()
