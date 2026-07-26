# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Artifact storage implemented through an existing TransferQueue service."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from vllm.distributed.artifact_connector.store import (
    ArtifactArray,
    ArtifactCorruptionError,
    ArtifactNotReadyError,
)


def _canonical_json(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


class TransferQueueArtifactStore:
    """Opaque ArtifactStore backed by TransferQueue key-value operations."""

    backend_name = "transfer_queue"

    def __init__(
        self,
        *,
        ray_address: str,
        store_id: str,
        data_partition: str,
        request_partition: str,
        connect_timeout_seconds: float = 30.0,
    ) -> None:
        try:
            import ray
            import transfer_queue as tq
            from tensordict import TensorDict
            from tensordict.tensorclass import NonTensorStack
            from transfer_queue import (
                TransferQueueObjectNotFoundError,
                TransferQueueObjectNotReadyError,
            )
        except ImportError as error:
            raise ImportError(
                "The TransferQueue artifact backend requires transfer_queue, "
                "Ray, and TensorDict"
            ) from error

        self._owns_ray = not ray.is_initialized()
        if self._owns_ray:
            ray.init(
                address=ray_address,
                namespace="transfer_queue",
                logging_level="ERROR",
            )
        try:
            tq.connect(timeout=connect_timeout_seconds)
        except Exception:
            if self._owns_ray:
                ray.shutdown()
            raise

        self._ray = ray
        self._tq = tq
        self._TensorDict = TensorDict
        self._NonTensorStack = NonTensorStack
        self._not_found_error = TransferQueueObjectNotFoundError
        self._not_ready_error = TransferQueueObjectNotReadyError
        self._closed = False
        self.store_id = store_id
        self.data_partition = data_partition
        self.request_partition = request_partition

    @staticmethod
    def _array_header(
        kind: str,
        artifact: ArtifactArray,
    ) -> dict[str, Any]:
        contiguous = np.ascontiguousarray(artifact.array)
        header = {
            "schema_version": 1,
            "kind": kind,
            "object_id": artifact.object_id,
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "payload_sha256": hashlib.sha256(contiguous.view(np.uint8)).hexdigest(),
            "metadata": artifact.metadata,
        }
        header["header_sha256"] = hashlib.sha256(_canonical_json(header)).hexdigest()
        return header

    @staticmethod
    def _validate_array(
        header: dict[str, Any],
        *,
        kind: str,
        object_id: str,
        array: np.ndarray,
    ) -> None:
        unsigned = dict(header)
        expected_header_sha256 = unsigned.pop("header_sha256", None)
        if (
            expected_header_sha256
            != hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        ):
            raise ArtifactCorruptionError("artifact header checksum mismatch")
        if (
            header.get("schema_version") != 1
            or header.get("kind") != kind
            or header.get("object_id") != object_id
        ):
            raise ArtifactCorruptionError("artifact header identity mismatch")
        try:
            expected_dtype = np.dtype(header["dtype"])
            expected_shape = tuple(int(value) for value in header["shape"])
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactCorruptionError("invalid artifact array header") from error
        if array.dtype != expected_dtype or array.shape != expected_shape:
            raise ArtifactCorruptionError("artifact array dtype or shape mismatch")
        if (
            header.get("payload_sha256")
            != hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()
        ):
            raise ArtifactCorruptionError("artifact payload checksum mismatch")

    def _get(
        self,
        *,
        keys: list[str] | str,
        partition_id: str,
        select_fields: list[str] | str,
    ) -> Any:
        try:
            return self._tq.kv_batch_get(
                keys=keys,
                partition_id=partition_id,
                select_fields=select_fields,
            )
        except self._not_ready_error as error:
            raise ArtifactNotReadyError(str(error)) from error
        except self._not_found_error as error:
            raise FileNotFoundError(str(error)) from error

    def _put_array_batch(self, kind: str, artifacts: list[ArtifactArray]) -> None:
        if not artifacts:
            return
        fields = self._TensorDict(
            {
                "header": self._NonTensorStack(
                    *(self._array_header(kind, artifact) for artifact in artifacts)
                ),
                "payload": torch.stack(
                    [
                        torch.from_numpy(np.ascontiguousarray(artifact.array))
                        for artifact in artifacts
                    ]
                ),
            },
            batch_size=[len(artifacts)],
        )
        self._tq.kv_batch_put(
            keys=[artifact.object_id for artifact in artifacts],
            partition_id=self.data_partition,
            fields=fields,
        )

    def put_blocks(self, blocks: list[ArtifactArray]) -> None:
        """Batch full blocks with equal tensor profiles."""
        unique: dict[str, ArtifactArray] = {}
        for block in blocks:
            existing = unique.get(block.object_id)
            if existing is not None and (
                existing.array.dtype != block.array.dtype
                or existing.array.shape != block.array.shape
                or not np.array_equal(existing.array, block.array)
            ):
                raise ArtifactCorruptionError(
                    f"conflicting artifact payload for {block.object_id}"
                )
            unique[block.object_id] = block

        groups: dict[tuple[str, tuple[int, ...]], list[ArtifactArray]] = defaultdict(
            list
        )
        for block in unique.values():
            groups[(block.array.dtype.str, block.array.shape)].append(block)
        for group in groups.values():
            self._put_array_batch("block", group)

    def retain_blocks(self, object_ids: list[str]) -> None:
        """Fail closed if a scheduler-ready block disappeared."""
        if not object_ids:
            return
        data = self._get(
            keys=object_ids,
            partition_id=self.data_partition,
            select_fields="header",
        )
        if len(data["header"]) != len(object_ids):
            raise ArtifactNotReadyError(
                "TransferQueue returned an incomplete artifact block batch"
            )

    def release_blocks(self, object_ids: list[str]) -> None:
        """Let the TransferQueue deployment own object retention."""

    def put_array(
        self,
        kind: str,
        object_id: str,
        array: np.ndarray,
        metadata: dict[str, Any],
    ) -> None:
        if kind not in ("block", "tail"):
            raise ValueError(f"invalid artifact array kind: {kind}")
        self._put_array_batch(
            kind,
            [
                ArtifactArray(
                    object_id=object_id,
                    array=np.ascontiguousarray(array),
                    metadata=metadata,
                )
            ],
        )

    def put_manifest(self, sample_id: str, manifest: dict[str, Any]) -> str:
        signed = dict(manifest)
        signed.pop("manifest_sha256", None)
        manifest_sha256 = hashlib.sha256(_canonical_json(signed)).hexdigest()
        signed["manifest_sha256"] = manifest_sha256
        self._tq.kv_put(
            key=sample_id,
            partition_id=self.request_partition,
            fields={"manifest": signed},
        )
        return manifest_sha256

    def read_array(self, kind: str, object_id: str) -> np.ndarray:
        data = self._get(
            keys=object_id,
            partition_id=self.data_partition,
            select_fields=["header", "payload"],
        )
        if len(data["header"]) != 1 or len(data["payload"]) != 1:
            raise ArtifactCorruptionError("expected one TransferQueue artifact row")
        header = data["header"][0]
        tensor = data["payload"][0]
        if not isinstance(header, dict) or not isinstance(tensor, torch.Tensor):
            raise ArtifactCorruptionError("invalid TransferQueue artifact row")
        array = tensor.detach().cpu().numpy().copy()
        self._validate_array(
            header,
            kind=kind,
            object_id=object_id,
            array=array,
        )
        return array

    def read_manifest(self, sample_id: str) -> dict[str, Any]:
        data = self._get(
            keys=sample_id,
            partition_id=self.request_partition,
            select_fields="manifest",
        )
        if len(data["manifest"]) != 1:
            raise ArtifactCorruptionError("expected one TransferQueue manifest row")
        manifest = data["manifest"][0]
        if not isinstance(manifest, dict):
            raise ArtifactCorruptionError("invalid TransferQueue manifest row")
        unsigned = dict(manifest)
        expected = unsigned.pop("manifest_sha256", None)
        if expected != hashlib.sha256(_canonical_json(unsigned)).hexdigest():
            raise ArtifactCorruptionError("manifest checksum mismatch")
        return manifest

    def close(self) -> None:
        """Disconnect without changing the external deployment."""
        if self._closed:
            return
        self._closed = True
        try:
            self._tq.disconnect()
        finally:
            if self._owns_ray:
                self._ray.shutdown()
