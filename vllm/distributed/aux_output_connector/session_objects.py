# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Write auxiliary outputs of session requests to a shared object store."""

from __future__ import annotations

import json
import queue
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from safetensors.numpy import save

from vllm.distributed.aux_output_connector.object_store import (
    MooncakeAuxObjectStore,
)
from vllm.distributed.aux_output_connector.store import BlockObjectStoreError
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.aux_output_connector.worker import _WorkerRequestState
    from vllm.sampling_params import SamplingParams
    from vllm.v1.engine import EngineCoreRequest

logger = init_logger(__name__)


def validate_aux_request(
    vllm_config: VllmConfig, sampling_params: SamplingParams, resumable: bool
) -> None:
    """Check the auxiliary-output parameters of a request."""
    session = sampling_params.aux_session
    if session is None:
        return
    if vllm_config.aux_output_config.object_key_prefix is None:
        reason = "requires AuxOutputConfig.object_key_prefix"
    elif not session or "/" in session:
        reason = "must be a non-empty string without '/'"
    elif sampling_params.stop:
        reason = "does not support stop strings; use stop_token_ids"
    elif resumable:
        reason = "does not support streaming input"
    else:
        return
    raise VLLMValidationError(f"aux_session {reason}.", parameter="aux_session")


_SESSION_SALT = "aux_session:"
_PLAIN_SALT = "aux_salt:"


def salt_aux_session(vllm_config: VllmConfig, request: EngineCoreRequest) -> None:
    """Fold a request's `aux_session` into its cache salt, so that only
    requests of the session compute or hit its KV blocks."""
    if vllm_config.aux_output_config.object_key_prefix is None:
        return
    params = request.sampling_params
    salt = request.cache_salt
    if params is not None and params.aux_session is not None:
        salt = _SESSION_SALT + json.dumps([params.aux_session, salt])
    elif salt is not None and salt.startswith((_SESSION_SALT, _PLAIN_SALT)):
        salt = _PLAIN_SALT + salt
    else:
        return
    request.cache_salt = salt
    request.block_hashes = None


def aux_object_key(
    key_prefix: str, session: str, block_hash_hex: str, weight_version: str
) -> str:
    """Key of the object that ends with the hash block `block_hash_hex` and
    whose last row was computed with weights `weight_version`."""
    return f"{key_prefix}/{session}/{block_hash_hex}@{weight_version}"


def record_weight_version(
    versions: list[tuple[int, str]], start: int, weight_version: str
) -> None:
    """Record that positions from `start` on are computed with `weight_version`."""
    while versions and versions[-1][0] >= start:
        versions.pop()
    if not versions or versions[-1][1] != weight_version:
        versions.append((start, weight_version))


def weight_version_at(versions: Sequence[tuple[int, str]], position: int) -> str:
    """Weight version that computed `position`."""
    for start, weight_version in reversed(versions):
        if start <= position:
            return weight_version
    raise ValueError(f"no weight version recorded for position {position}")


class SessionObjectWriter:
    """Write the objects of session requests in the background. A failed
    write is logged and not retried: `check` raises from then on."""

    def __init__(
        self,
        store: MooncakeAuxObjectStore,
        *,
        key_prefix: str,
        max_pending_batches: int,
    ) -> None:
        self._store = store
        self._key_prefix = key_prefix
        self._queue: queue.Queue[tuple[list[str], list[bytes]] | None] = queue.Queue(
            maxsize=max_pending_batches
        )
        self._error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="vllm-aux-object-writer"
        )
        self._thread.start()

    def _run(self) -> None:
        while (batch := self._queue.get()) is not None:
            try:
                if self._error is None:
                    self._store.put_batch(*batch)
            except BaseException as error:
                logger.exception("Auxiliary output object write failed.")
                self._error = error

    def check(self) -> None:
        """Raise `BlockObjectStoreError` if a write failed."""
        if self._error is not None:
            raise BlockObjectStoreError(
                "auxiliary output object write failed"
            ) from self._error

    def add_blocks(
        self,
        state: _WorkerRequestState,
        blocks: Sequence[tuple[int, np.ndarray]],
        object_size: int,
    ) -> None:
        """Stage the next full hash blocks of a request; write completed objects."""
        keys: list[str] = []
        values: list[bytes] = []
        for start, rows in blocks:
            state.staged.append((start, np.array(rows)))
            end = start + len(rows)
            if end % object_size:
                continue
            assert state.session is not None
            keys.append(
                aux_object_key(
                    self._key_prefix,
                    state.session,
                    state.block_hashes[end // len(rows) - 1].hex(),
                    weight_version_at(state.weight_versions, end - 1),
                )
            )
            pieces = np.concatenate([piece for _, piece in state.staged])
            values.append(save({"routed_experts": pieces}))
            state.staged.clear()
        if keys:
            self._queue.put((keys, values))

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()
        self._store.close()
