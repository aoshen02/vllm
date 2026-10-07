# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sample-logprobs container of compact requests: the top-k of each engine
row base64-encoded as the rows arrive, so building the response after an
abort only stitches the segments together."""

from typing import ClassVar

import numpy as np
import pybase64

from vllm.entrypoints.scale_out.token_in_token_out.array_logprobs import (
    ArrayLogprobs,
)


class _Base64Stream:
    """Incremental base64 of a byte stream, as standalone segments: bytes are
    encoded in pieces whose length is a multiple of 3, so the concatenation
    of :meth:`parts` is the ``b64encode`` of the whole stream."""

    FLUSH_BYTES: ClassVar[int] = 3 << 18  # 768 KiB of input per segment

    def __init__(self) -> None:
        self.segments: list[bytes] = []
        self.pending = bytearray()

    def write(self, data: np.ndarray) -> None:
        if data.size == 0:
            return  # memoryview.cast() rejects zero-size shapes
        self.pending += memoryview(data).cast("B")
        if len(self.pending) >= self.FLUSH_BYTES:
            cut = len(self.pending) - len(self.pending) % 3
            self.segments.append(pybase64.b64encode(self.pending[:cut]))
            del self.pending[:cut]

    def parts(self) -> list[bytes]:
        if not self.pending:
            return list(self.segments)
        return [*self.segments, pybase64.b64encode(bytes(self.pending))]


class WireLogprobs:
    """Little-endian int32 ids and float32 logprobs of engine slots
    ``1..S-1``, base64-encoded as rows arrive. Rows of another width make it
    irregular (the compact format cannot represent them)."""

    def __init__(self) -> None:
        self.num_slots: int | None = None  # engine row width (incl. slot 0)
        self.num_positions = 0
        self.is_regular = True
        self.token_ids = _Base64Stream()
        self.logprobs = _Base64Stream()

    def append_rows(
        self, token_ids: np.ndarray, logprobs: np.ndarray, ranks: np.ndarray
    ) -> None:
        ArrayLogprobs._check_rows(token_ids, logprobs, ranks, len(ranks))
        width = token_ids.shape[1]
        if self.num_slots is not None and width != self.num_slots:
            self.is_regular = False
        if not self.is_regular:
            return
        ids = np.ascontiguousarray(token_ids[:, 1:], dtype="<i4")
        values = np.ascontiguousarray(logprobs[:, 1:], dtype="<f4")
        self.num_slots = width
        self.token_ids.write(ids)
        self.logprobs.write(values)
        self.num_positions += len(ranks)

    def __len__(self) -> int:
        return self.num_positions
