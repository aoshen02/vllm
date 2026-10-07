# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sample logprobs of non-streaming ``/inference/v1/generate`` requests kept as
the engine's numpy rows (``ArrayLogprobs``), a container for the
sample-logprobs hook (``vllm.logprobs.register_sample_logprobs_container``):
the core ``LogprobsProcessor`` feeds it through ``append_rows``, and
``logprobs_render`` renders the response from the rows without per-entry
objects."""

import itertools
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from vllm.logprobs import Logprob, LogprobsOnePosition

_INT32 = np.dtype("<i4")
_FLOAT32 = np.dtype("<f4")
_INT32_INFO = np.iinfo(np.int32)


def _fits_int32(values: np.ndarray) -> bool:
    """Integers that int32 holds exactly (engine ranks, and the ids of some
    model runners, are int64)."""
    dtype = values.dtype
    if dtype.kind not in "iu":
        return False
    if dtype.itemsize < 4 or (dtype.kind == "i" and dtype.itemsize == 4):
        return True
    return values.size == 0 or (
        _INT32_INFO.min <= values.min() and values.max() <= _INT32_INFO.max
    )


@dataclass
class ArrayLogprobs:
    """
    Sample logprobs of a request kept as the engine's ``LogprobsLists`` rows.

    Row ``i`` is exactly the engine row: slot 0 holds the sampled token,
    slots ``1..S-1`` the top-k candidates in engine order. Candidate tokens
    are never detokenized. Rows are copied into ``int32`` / ``float32`` /
    ``int32`` numpy blocks whose capacity grows geometrically from the first
    append (1, 2, 4, ... rows, or the size of the incoming engine chunk if
    larger) up to ``BLOCK_BYTES``, so a request reserves at most about twice
    its data (or one block), and ``N`` positions create
    ``O(log N + N * row_bytes / BLOCK_BYTES)`` Python objects instead of
    ``O(N * S)``. Logprobs keep the raw engine values, non-finite included.

    Rows must have the engine's dtypes: integer ids and ranks within int32,
    float logprobs of at most 4 bytes; ``append_rows`` raises otherwise (the
    core's ``SampleLogprobsHandle`` then fails the request).

    Irregular engine output (a row whose width differs from the stored
    width, e.g. when a co-batched request's ``logprob_token_ids`` replaced
    the batch's logprob tensors) never raises: from that position on, rows
    are kept as legacy ``dict[int, Logprob]`` entries (same truncation and
    overwrite semantics as the list path) and :attr:`is_regular` becomes
    False, so renderers fall back to the legacy path.

    """

    BLOCK_BYTES: ClassVar[int] = 8 << 20

    # Blocks; only the first ``_tail_fill`` rows of the last block are used.
    token_id_chunks: list[np.ndarray] = field(default_factory=list)
    logprob_chunks: list[np.ndarray] = field(default_factory=list)
    rank_chunks: list[np.ndarray] = field(default_factory=list)
    num_positions: int = 0
    _tail_fill: int = 0
    # Positions after the array rows, once irregular rows were seen.
    _legacy: list[LogprobsOnePosition] | None = None

    @property
    def is_regular(self) -> bool:
        """Whether every position is stored as an array row."""
        return self._legacy is None

    @staticmethod
    def _row_dict(
        token_ids: np.ndarray, logprobs: np.ndarray, rank: int
    ) -> LogprobsOnePosition:
        ids = token_ids.tolist()
        ranks = itertools.chain((int(rank),), range(1, len(ids)))
        # Same insertion/overwrite semantics as the list[dict] path.
        return {
            token_id: Logprob(logprob=value, rank=r)
            for token_id, value, r in zip(ids, logprobs.tolist(), ranks)
        }

    def append_rows(
        self, token_ids: np.ndarray, logprobs: np.ndarray, ranks: np.ndarray
    ) -> None:
        """Append ``n`` positions given as ``[n, S]``, ``[n, S]`` and ``[n]``
        arrays. Values are copied, so engine buffers are not retained."""
        n = len(ranks)
        self._check_rows(token_ids, logprobs, ranks, n)
        width = token_ids.shape[1]
        if self._legacy is not None or (
            self.token_id_chunks and width != self.token_id_chunks[0].shape[1]
        ):
            if self._legacy is None:
                self._legacy = []
            self._legacy.extend(
                self._row_dict(token_ids[i], logprobs[i], ranks[i]) for i in range(n)
            )
            self.num_positions += n
            return
        pos = 0
        while pos < n:
            if not self.rank_chunks or self._tail_fill == len(self.rank_chunks[-1]):
                self._new_block(n - pos, width)
            fill = self._tail_fill
            take = min(n - pos, len(self.rank_chunks[-1]) - fill)
            self.token_id_chunks[-1][fill : fill + take] = token_ids[pos : pos + take]
            self.logprob_chunks[-1][fill : fill + take] = logprobs[pos : pos + take]
            self.rank_chunks[-1][fill : fill + take] = ranks[pos : pos + take]
            self._tail_fill = fill + take
            pos += take
        self.num_positions += n

    @staticmethod
    def _check_rows(
        token_ids: np.ndarray, logprobs: np.ndarray, ranks: np.ndarray, n: int
    ) -> None:
        # Numpy would broadcast fewer rows; surplus rows (which the base path
        # silently truncates) mean inconsistent engine output too.
        id_shape = token_ids.shape
        if (
            len(id_shape) != 2
            or logprobs.shape != id_shape
            or id_shape[0] != n
            or ranks.shape != (n,)
        ):
            raise ValueError(
                f"Inconsistent logprob rows: token_ids {id_shape}, "
                f"logprobs {logprobs.shape}, ranks {ranks.shape}"
            )
        if not (
            logprobs.dtype.kind == "f"
            and logprobs.dtype.itemsize <= 4
            and _fits_int32(token_ids)
            and _fits_int32(ranks)
        ):
            raise TypeError(
                f"Unsupported logprob rows: token_ids {token_ids.dtype}, "
                f"logprobs {logprobs.dtype}, ranks {ranks.dtype} (expected "
                "integers within int32 and float32)"
            )

    def _new_block(self, remaining: int, width: int) -> None:
        max_rows = max(1, self.BLOCK_BYTES // (width * 8 + 4))
        previous = len(self.rank_chunks[-1]) if self.rank_chunks else 0
        rows = max(remaining, min(max_rows, max(1, 2 * previous)))
        self.token_id_chunks.append(np.empty((rows, width), dtype=_INT32))
        self.logprob_chunks.append(np.empty((rows, width), dtype=_FLOAT32))
        self.rank_chunks.append(np.empty((rows,), dtype=_INT32))
        self._tail_fill = 0

    def _filled_blocks(self) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """The used ``(token_ids, logprobs, ranks)`` rows, per block."""
        last, fill = len(self.rank_chunks) - 1, self._tail_fill
        for i, (t, lp, r) in enumerate(
            zip(self.token_id_chunks, self.logprob_chunks, self.rank_chunks)
        ):
            yield (t[:fill], lp[:fill], r[:fill]) if i == last else (t, lp, r)

    @property
    def num_slots(self) -> int | None:
        """Slots per position, or None if no position was stored yet."""
        if not self.token_id_chunks:
            return None
        return self.token_id_chunks[0].shape[1]

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return contiguous ``(token_ids[N, S], logprobs[N, S], ranks[N])``.

        Blocks are concatenated once and kept as a single exact-sized block
        afterwards. Returns ``[0, 0]``-shaped arrays when there are no
        positions. Raises ValueError when the container is not
        :attr:`is_regular`.
        """
        if self._legacy is not None:
            raise ValueError("Logprob rows have inconsistent widths")
        if not self.token_id_chunks:
            return (
                np.empty((0, 0), dtype=_INT32),
                np.empty((0, 0), dtype=_FLOAT32),
                np.empty((0,), dtype=_INT32),
            )
        if len(self.rank_chunks) > 1 or self._tail_fill != len(self.rank_chunks[0]):
            ids, values, ranks = zip(*self._filled_blocks())
            self.token_id_chunks = [np.concatenate(ids)]
            self.logprob_chunks = [np.concatenate(values)]
            self.rank_chunks = [np.concatenate(ranks)]
            self._tail_fill = len(self.rank_chunks[0])
        return self.token_id_chunks[0], self.logprob_chunks[0], self.rank_chunks[0]

    def __len__(self) -> int:
        """Gets number of positions stored in the container"""
        return self.num_positions

    def __iter__(self) -> Iterator[LogprobsOnePosition]:
        """Positions in order as ``dict[int, Logprob]`` (``decoded_token``
        None), as the list representation holds them."""
        for t, lp, r in self._filled_blocks():
            for j in range(len(r)):
                yield self._row_dict(t[j], lp[j], r[j])
        if self._legacy is not None:
            yield from list(self._legacy)
