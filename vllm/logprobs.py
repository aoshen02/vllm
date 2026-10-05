# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import operator
from collections.abc import Iterable, Iterator, MutableSequence
from dataclasses import InitVar, dataclass, field
from typing import ClassVar, overload

import numpy as np
import pybase64


# We use dataclass for now because it is used for
# openai server output, and msgspec is not serializable.
# TODO(sang): Fix it.
@dataclass
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


LogprobsOnePosition = dict[int, Logprob]


@dataclass
class FlatLogprobs(MutableSequence[LogprobsOnePosition | None]):
    """
    Flat logprobs of a request into multiple primitive type lists.

    Compared to list[dict[int, Logprob]], this data structure reduced GC
    overhead significantly. As it flattened logprob information for
    all positions and ranks in to multiple primitive type lists (i.e.
    logprobs, token_ids, ranks per token_ids, decoded_tokens).
    So regardless of the sequence length and top_logprobs setup,
    FlatLogprobs would only introduce a constant amount of objects.

    As each position might contains different amount of ranks,
    start_indices_per_position would be used to access the logprob ranges
    for different positions.

    NOTE: To reduce the migration overhead and improve backward compatibility,
    we support the key Sequence APIs of list, so it could act as
    list[LogprobsOnePosition]
    """

    # Start / end indices to indicate the range of logprobs for each position.
    start_indices: list[int] = field(default_factory=list)
    end_indices: list[int] = field(default_factory=list)

    # Flatten Logprob information for (each position, rank).
    # For position <i>, the logprobs are ranged
    # from self.start_indices[i] to self.end_indices[i] (exclusive).
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    ranks: list[int | None] = field(default_factory=list)
    decoded_tokens: list[str | None] = field(default_factory=list)

    def append(self, logprobs_one_position: LogprobsOnePosition | None) -> None:
        """Appends the container with logprobs for the next position"""
        self.start_indices.append(len(self.logprobs))
        if logprobs_one_position:
            for token_id, logprob in logprobs_one_position.items():
                self.token_ids.append(token_id)
                self.logprobs.append(logprob.logprob)
                self.ranks.append(logprob.rank)
                self.decoded_tokens.append(logprob.decoded_token)
        self.end_indices.append(len(self.logprobs))

    def append_fast(
        self,
        token_ids: list[int],
        logprobs: list[float],
        ranks: itertools.chain[int],
        decoded_tokens: Iterable[str | None],
    ) -> None:
        """
        Appends logprobs for the next position without creating
        the intermediate logprob dictionary.
        """
        self.start_indices.append(len(self.logprobs))
        for token_id, logprob, rank, decoded_token in zip(
            token_ids, logprobs, ranks, decoded_tokens
        ):
            self.token_ids.append(token_id)
            self.logprobs.append(logprob)
            self.ranks.append(rank)
            self.decoded_tokens.append(decoded_token)
        self.end_indices.append(len(self.logprobs))

    def extend(self, logprobs_multi_positions) -> None:
        """Extends the container with logprobs for the next multiple positions"""
        for logprobs_one_position in logprobs_multi_positions:
            self.append(logprobs_one_position)

    def __len__(self) -> int:
        """Gets number of positions stored in the container"""
        return len(self.start_indices)

    @overload
    def __getitem__(self, position: int) -> LogprobsOnePosition: ...

    @overload
    def __getitem__(self, s: slice, /) -> "FlatLogprobs": ...

    def __getitem__(self, index: int | slice):
        """Extracts logprobs of a given position or slice"""
        if isinstance(index, int):
            return {
                self.token_ids[i]: Logprob(
                    logprob=self.logprobs[i],
                    rank=self.ranks[i],
                    decoded_token=self.decoded_tokens[i],
                )
                for i in range(self.start_indices[index], self.end_indices[index])
            }
        elif isinstance(index, slice):
            selected_starts = self.start_indices[index]
            selected_ends = self.end_indices[index]
            # Empty slices have no source offset to normalize.
            if not selected_starts:
                return FlatLogprobs()
            min_index = selected_starts[0]
            max_index = selected_ends[-1]
            return FlatLogprobs(
                # Shift updated start_indices and end_indices to
                # be 0-indexed
                start_indices=[i - min_index for i in selected_starts],
                end_indices=[i - min_index for i in selected_ends],
                token_ids=self.token_ids[min_index:max_index],
                logprobs=self.logprobs[min_index:max_index],
                ranks=self.ranks[min_index:max_index],
                decoded_tokens=self.decoded_tokens[min_index:max_index],
            )
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    def __setitem__(self, item, value) -> None:
        raise TypeError("Cannot set logprobs in FlatLogprobs")

    def __delitem__(self, item) -> None:
        raise TypeError("Cannot delete logprobs from FlatLogprobs")

    def insert(self, index: int, value: dict[int, Logprob] | None) -> None:
        raise TypeError("Cannot insert logprobs to FlatLogprobs")

    def __iter__(self) -> Iterator[LogprobsOnePosition]:
        """
        Iterates the container and yields LogprobsOnePosition for
        each position.
        """
        for i in range(0, len(self.start_indices)):
            yield self.__getitem__(i)


_INT32 = np.dtype("<i4")
_INT64 = np.dtype("<i8")
_FLOAT32 = np.dtype("<f4")
_FLOAT64 = np.dtype("<f8")
_INT32_INFO = np.iinfo(np.int32)


_NARROW_INTS = frozenset(
    np.dtype(t) for t in (np.int8, np.int16, np.int32, np.uint8, np.uint16)
)
_MIN_REDUCE = np.minimum.reduce
_MAX_REDUCE = np.maximum.reduce


def _storage_int_dtype(values: np.ndarray) -> np.dtype:
    """int32 unless a value does not fit (then int64; never wraps).

    Called per engine step: the engine's int32 ids need no scan, and tiny
    arrays (e.g. one rank per step) are checked in Python, which is several
    times cheaper than numpy reductions at that size.
    """
    if values.dtype in _NARROW_INTS or values.size == 0:
        return _INT32
    flat = values.reshape(-1)
    if flat.size <= 8:
        items = flat.tolist()
        lo, hi = min(items), max(items)
    else:
        lo, hi = int(_MIN_REDUCE(flat)), int(_MAX_REDUCE(flat))
    if lo >= _INT32_INFO.min and hi <= _INT32_INFO.max:
        return _INT32
    return _INT64


def _storage_float_dtype(values: np.ndarray) -> np.dtype:
    """float32 (the engine dtype; float16 widens exactly) or float64."""
    return _FLOAT64 if values.dtype == np.float64 else _FLOAT32


class _Base64Stream:
    """Incremental base64 of a byte stream, as standalone segments.

    Bytes are buffered and encoded in pieces whose length is a multiple of
    3, so the concatenation of :meth:`parts` equals ``b64encode`` of the
    whole stream (only the last part can carry ``=`` padding). The pending
    buffer stays below ``FLUSH_BYTES`` + one write.
    """

    FLUSH_BYTES: ClassVar[int] = 3 << 18  # 768 KiB of input per segment

    def __init__(self) -> None:
        self.segments: list[bytes] = []
        self.pending = bytearray()

    def write(self, data: np.ndarray) -> None:
        self.pending += memoryview(data).cast("B")
        if len(self.pending) >= self.FLUSH_BYTES:
            cut = len(self.pending) - len(self.pending) % 3
            self.segments.append(pybase64.b64encode(self.pending[:cut]))
            del self.pending[:cut]

    def parts(self) -> list[bytes]:
        """Base64 text of everything written so far (does not consume)."""
        if not self.pending:
            return list(self.segments)
        return [*self.segments, pybase64.b64encode(bytes(self.pending))]

    def decode(self) -> bytes:
        return pybase64.b64decode(b"".join(self.parts()))


class _WireEncoder:
    """Rows encoded at append time in the compact wire format: base64 of
    little-endian ``int32`` token ids, ``float32`` logprobs, ``int32``
    ranks.

    With ``topk_only`` the id/logprob streams hold engine slots ``1..S-1``
    only (``compact_include_sampled=false``); slot 0 is then kept in two
    small side streams so the rows can still be decoded losslessly.
    """

    def __init__(self, topk_only: bool = False) -> None:
        self.topk_only = topk_only
        self.slots: int | None = None  # engine row width (incl. slot 0)
        self.num_rows = 0
        self.token_ids = _Base64Stream()
        self.logprobs = _Base64Stream()
        self.ranks = _Base64Stream()
        self.sampled_ids = _Base64Stream()
        self.sampled_logprobs = _Base64Stream()

    def try_write(
        self, token_ids: np.ndarray, logprobs: np.ndarray, ranks: np.ndarray
    ) -> bool:
        """Encode the rows, or return False if they are not representable
        (width change, ids/ranks beyond int32, logprobs wider than float32,
        which the container then keeps losslessly in array mode)."""
        if logprobs.dtype not in (_FLOAT32, np.dtype(np.float16)):
            return False
        width = token_ids.shape[1]
        if self.slots is not None and width != self.slots:
            return False
        if _storage_int_dtype(token_ids) != _INT32 or (
            _storage_int_dtype(ranks) != _INT32
        ):
            return False
        self.slots = width
        if self.topk_only:
            self.token_ids.write(np.ascontiguousarray(token_ids[:, 1:], dtype=_INT32))
            self.logprobs.write(np.ascontiguousarray(logprobs[:, 1:], dtype=_FLOAT32))
            self.sampled_ids.write(np.ascontiguousarray(token_ids[:, 0], dtype=_INT32))
            self.sampled_logprobs.write(
                np.ascontiguousarray(logprobs[:, 0], dtype=_FLOAT32)
            )
        else:
            self.token_ids.write(np.ascontiguousarray(token_ids, dtype=_INT32))
            self.logprobs.write(np.ascontiguousarray(logprobs, dtype=_FLOAT32))
        self.ranks.write(np.ascontiguousarray(ranks, dtype=_INT32))
        self.num_rows += len(ranks)
        return True

    def decode(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        slots = self.slots or 0
        n = self.num_rows
        stored = slots - 1 if self.topk_only and slots else slots
        token_ids = np.frombuffer(self.token_ids.decode(), dtype=_INT32)
        logprobs = np.frombuffer(self.logprobs.decode(), dtype=_FLOAT32)
        ranks = np.frombuffer(self.ranks.decode(), dtype=_INT32)
        token_ids = token_ids.reshape(n, stored)
        logprobs = logprobs.reshape(n, stored)
        if self.topk_only and slots:
            sampled_ids = np.frombuffer(self.sampled_ids.decode(), dtype=_INT32)
            sampled = np.frombuffer(self.sampled_logprobs.decode(), dtype=_FLOAT32)
            token_ids = np.column_stack((sampled_ids, token_ids))
            logprobs = np.column_stack((sampled, logprobs))
        return token_ids, logprobs, ranks


@dataclass
class ArrayLogprobs(MutableSequence[LogprobsOnePosition | None]):
    """
    Sample logprobs of a request kept as the engine's ``LogprobsLists`` rows.

    Row ``i`` is exactly the engine row: slot 0 holds the sampled token,
    slots ``1..S-1`` the top-k candidates in engine order. Candidate tokens
    are never detokenized. Rows are copied into numpy blocks whose capacity
    grows geometrically from the first append (1, 2, 4, ... rows, or the
    size of the incoming engine chunk if larger) up to ``BLOCK_BYTES``, so
    a request reserves at most about twice its data (or one block), and
    ``N`` positions create ``O(log N + N * row_bytes / BLOCK_BYTES)``
    Python objects instead of ``O(N * S)``.

    Each block has its own dtypes: ``token_ids`` / ``ranks`` are ``int32``
    unless a value needs ``int64`` (never wrapped); ``logprobs`` keep the
    raw engine values, including non-finite ones, as ``float32`` (the engine
    dtype) or ``float64`` if the engine produced float64. :meth:`arrays`
    promotes losslessly across blocks.

    Slices are exact-sized views of the stored rows (no allocation).

    Irregular engine output (a row whose width differs from the stored
    width, e.g. when a co-batched request's ``logprob_token_ids`` replaced
    the batch's logprob tensors) never raises: from that position on, rows
    are kept as legacy ``dict[int, Logprob]`` entries (same truncation and
    overwrite semantics as the list path) and :attr:`is_regular` becomes
    False, so renderers fall back to the legacy path.

    Positional access (``container[i]``) materializes the same
    ``dict[int, Logprob]`` the list representation would hold (with
    ``decoded_token=None``), so generic consumers keep working; fast
    consumers use :meth:`arrays` instead.

    With ``wire_base64=True`` (non-streaming compact responses) rows are
    not stored as arrays: they are encoded immediately into the compact
    wire format (base64 of ``<i4``/``<f4``/``<i4``), about 1.33x the raw
    int32/float32 size, so building the response only stitches the
    segments together (:meth:`wire_parts`). Any other access first decodes
    the rows back into an array block (values as on the wire, i.e.
    float32), and rows the wire cannot represent switch the container
    back to array storage.
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
    # For slices: positions the source container held when it was sliced
    # (DELTA outputs slice suffixes of the cumulative container).
    source_positions: int | None = None
    wire_base64: InitVar[bool] = False
    wire_topk_only: InitVar[bool] = False
    # Fast path for appends into the tail block (see _tail_can_hold).
    _tail_block: np.ndarray | None = field(default=None, init=False, repr=False)
    _tail_dtypes: tuple[np.dtype, np.dtype, np.dtype] | None = field(
        default=None, init=False, repr=False
    )
    _wire: _WireEncoder | None = field(default=None, init=False, repr=False)

    def __post_init__(self, wire_base64: bool, wire_topk_only: bool) -> None:
        if wire_base64:
            self._wire = _WireEncoder(topk_only=wire_topk_only)

    def _unwire(self) -> None:
        """Leave wire mode: decode the encoded rows into an array block."""
        wire = self._wire
        if wire is None:
            return
        self._wire = None
        if wire.num_rows:
            token_ids, logprobs, ranks = wire.decode()
            # Writable copies, like array-mode storage.
            self.token_id_chunks = [token_ids.copy()]
            self.logprob_chunks = [logprobs.copy()]
            self.rank_chunks = [ranks.copy()]
            self._tail_fill = wire.num_rows

    def wire_parts(
        self,
    ) -> tuple[int, int | None, list[bytes], list[bytes], list[bytes]] | None:
        """``(N, S, token_ids, logprobs, ranks)`` base64 parts, or None if the
        container is not (or no longer) in wire mode. ``S`` is the engine row
        width; with :attr:`wire_topk_only` the id/logprob parts hold slots
        ``1..S-1`` only."""
        wire = self._wire
        if wire is None:
            return None
        return (
            wire.num_rows,
            wire.slots,
            wire.token_ids.parts(),
            wire.logprobs.parts(),
            wire.ranks.parts(),
        )

    @property
    def is_wire_topk_only(self) -> bool:
        """Whether rows are wire-encoded with the top-k-only layout."""
        return self._wire is not None and self._wire.topk_only

    @property
    def is_regular(self) -> bool:
        """Whether every position is stored as an array row."""
        return self._legacy is None

    @property
    def _array_positions(self) -> int:
        return self.num_positions - (len(self._legacy) if self._legacy else 0)

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

    def _append_legacy(self, positions: list[LogprobsOnePosition]) -> None:
        if self._legacy is None:
            self._legacy = []
        self._legacy.extend(positions)
        self.num_positions += len(positions)

    def append_rows(
        self, token_ids: np.ndarray, logprobs: np.ndarray, ranks: np.ndarray
    ) -> None:
        """Append ``n`` positions given as ``[n, S]``, ``[n, S]`` and ``[n]``
        arrays. Values are copied, so engine buffers are not retained."""
        n = len(ranks)
        if n == 0:
            return
        if self._wire is not None:
            if self._wire.try_write(token_ids, logprobs, ranks):
                self.num_positions += n
                return
            self._unwire()
        width = token_ids.shape[1]
        if self._legacy is not None or (
            self.token_id_chunks and width != self.token_id_chunks[0].shape[1]
        ):
            self._append_legacy(
                [self._row_dict(token_ids[i], logprobs[i], ranks[i]) for i in range(n)]
            )
            return
        dtypes = (
            _storage_int_dtype(token_ids),
            _storage_float_dtype(logprobs),
            _storage_int_dtype(ranks),
        )
        pos = 0
        while pos < n:
            if not self._tail_can_hold(dtypes):
                self._new_block(n - pos, width, dtypes)
            fill = self._tail_fill
            take = min(n - pos, len(self.rank_chunks[-1]) - fill)
            self.token_id_chunks[-1][fill : fill + take] = token_ids[pos : pos + take]
            self.logprob_chunks[-1][fill : fill + take] = logprobs[pos : pos + take]
            self.rank_chunks[-1][fill : fill + take] = ranks[pos : pos + take]
            self._tail_fill = fill + take
            pos += take
        self.num_positions += n

    def _tail_can_hold(self, dtypes: tuple[np.dtype, np.dtype, np.dtype]) -> bool:
        if not self.rank_chunks or self._tail_fill == len(self.rank_chunks[-1]):
            return False
        if self.rank_chunks[-1] is self._tail_block and dtypes == self._tail_dtypes:
            return True  # same dtypes as the block was created with
        blocks = (
            self.token_id_chunks[-1],
            self.logprob_chunks[-1],
            self.rank_chunks[-1],
        )
        # Only lossless (widening) stores into the existing block.
        return all(
            np.can_cast(dtype, block.dtype, casting="safe")
            for dtype, block in zip(dtypes, blocks)
        )

    def _new_block(
        self, remaining: int, width: int, dtypes: tuple[np.dtype, np.dtype, np.dtype]
    ) -> None:
        tid_dtype, lp_dtype, rank_dtype = dtypes
        if self.rank_chunks and self._tail_fill != len(self.rank_chunks[-1]):
            # The tail block is only partly initialized (e.g. a dtype
            # widening starts a new block early): shrink it to its used rows,
            # since every block but the last is treated as fully used.
            fill = self._tail_fill
            self.token_id_chunks[-1] = self.token_id_chunks[-1][:fill].copy()
            self.logprob_chunks[-1] = self.logprob_chunks[-1][:fill].copy()
            self.rank_chunks[-1] = self.rank_chunks[-1][:fill].copy()
        row_bytes = width * (tid_dtype.itemsize + lp_dtype.itemsize)
        row_bytes += rank_dtype.itemsize
        max_rows = max(1, self.BLOCK_BYTES // row_bytes)
        previous = len(self.rank_chunks[-1]) if self.rank_chunks else 0
        rows = max(remaining, min(max_rows, max(1, 2 * previous)))
        self.token_id_chunks.append(np.empty((rows, width), dtype=tid_dtype))
        self.logprob_chunks.append(np.empty((rows, width), dtype=lp_dtype))
        self.rank_chunks.append(np.empty((rows,), dtype=rank_dtype))
        self._tail_fill = 0
        self._tail_block = self.rank_chunks[-1]
        self._tail_dtypes = dtypes

    def _filled_block(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The used ``(token_ids, logprobs, ranks)`` rows of block ``i``."""
        t, lp, r = self.token_id_chunks[i], self.logprob_chunks[i], self.rank_chunks[i]
        if i == len(self.rank_chunks) - 1 and self._tail_fill != len(r):
            fill = self._tail_fill
            return t[:fill], lp[:fill], r[:fill]
        return t, lp, r

    def _filled_blocks(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Snapshot of all used rows, per block."""
        return [self._filled_block(i) for i in range(len(self.rank_chunks))]

    @property
    def num_slots(self) -> int | None:
        """Slots per position, or None if no position was stored yet."""
        if self._wire is not None:
            return self._wire.slots
        if not self.token_id_chunks:
            return None
        return self.token_id_chunks[0].shape[1]

    def reserved_bytes(self) -> int:
        """Bytes allocated by the blocks (used and unused rows)."""
        return sum(
            a.nbytes
            for a in itertools.chain(
                self.token_id_chunks, self.logprob_chunks, self.rank_chunks
            )
        )

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return contiguous ``(token_ids[N, S], logprobs[N, S], ranks[N])``.

        Blocks are concatenated once (promoting dtypes losslessly) and kept
        as a single exact-sized block afterwards. Returns ``[0, 0]``-shaped
        arrays when there are no positions. Raises ValueError when the
        container is not :attr:`is_regular`.
        """
        self._unwire()
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
            self._tail_fill = self.num_positions
        return self.token_id_chunks[0], self.logprob_chunks[0], self.rank_chunks[0]

    def extend(self, values) -> None:
        if isinstance(values, ArrayLogprobs):
            if values.source_positions is not None:
                # Merged DELTA outputs: the newest slice's source count.
                self.source_positions = values.source_positions
            values._unwire()
            # Snapshot first: ``values`` may be ``self``.
            blocks = values._filled_blocks()
            legacy = list(values._legacy) if values._legacy is not None else None
            for t, lp, r in blocks:
                self.append_rows(t, lp, r)
            if legacy is not None:
                self._append_legacy(legacy)
            return
        # Other position containers (list, FlatLogprobs): keep legacy dicts.
        self._append_legacy(list(values))

    def __len__(self) -> int:
        """Gets number of positions stored in the container"""
        return self.num_positions

    @overload
    def __getitem__(self, position: int) -> LogprobsOnePosition: ...

    @overload
    def __getitem__(self, s: slice, /) -> "ArrayLogprobs": ...

    def __getitem__(self, index: int | slice):
        self._unwire()
        if isinstance(index, slice):
            return self._slice(index)
        try:
            index = operator.index(index)
        except TypeError:
            raise TypeError(f"Invalid index type: {type(index)}") from None
        if index < 0:
            index += self.num_positions
        if not 0 <= index < self.num_positions:
            raise IndexError("ArrayLogprobs index out of range")
        array_positions = self._array_positions
        if index >= array_positions:
            assert self._legacy is not None
            return self._legacy[index - array_positions]
        for i in range(len(self.rank_chunks)):
            t, lp, r = self._filled_block(i)
            if index < len(r):
                return self._row_dict(t[index], lp[index], r[index])
            index -= len(r)
        raise AssertionError("unreachable")

    def _slice(self, index: slice) -> "ArrayLogprobs":
        start, stop, step = index.indices(self.num_positions)
        if step != 1:
            raise ValueError("ArrayLogprobs only supports contiguous slices")
        result = ArrayLogprobs(source_positions=self.num_positions)
        if stop <= start:
            return result
        array_positions = self._array_positions
        if self._legacy is not None:
            legacy_part = self._legacy[
                max(start - array_positions, 0) : max(stop - array_positions, 0)
            ]
            if legacy_part:
                result._legacy = legacy_part
                result.num_positions = len(legacy_part)
            stop = min(stop, array_positions)
            if stop <= start:
                return result
        pieces = []
        # Walk blocks backward lazily: the output processor slices suffixes.
        end = array_positions
        for i in range(len(self.rank_chunks) - 1, -1, -1):
            if end <= start:
                break
            t, lp, r = self._filled_block(i)
            begin = end - len(r)
            lo, hi = max(start, begin) - begin, min(stop, end) - begin
            if lo < hi:
                pieces.append((t[lo:hi], lp[lo:hi], r[lo:hi]))
            end = begin
        # Exact-sized views; the result's blocks are full, so appends to it
        # allocate new blocks and never write into this container.
        if not pieces:
            return result
        for t, lp, r in reversed(pieces):
            result.token_id_chunks.append(t)
            result.logprob_chunks.append(lp)
            result.rank_chunks.append(r)
            result.num_positions += len(r)
        result._tail_fill = len(result.rank_chunks[-1])
        return result

    def __setitem__(self, item, value) -> None:
        raise TypeError("Cannot set logprobs in ArrayLogprobs")

    def __delitem__(self, item) -> None:
        raise TypeError("Cannot delete logprobs from ArrayLogprobs")

    def insert(self, index: int, value: dict[int, Logprob] | None) -> None:
        raise TypeError("Cannot insert logprobs to ArrayLogprobs")

    def __iter__(self) -> Iterator[LogprobsOnePosition]:
        """Positions in order, in one pass over the blocks."""
        self._unwire()
        for i in range(len(self.rank_chunks)):
            t, lp, r = self._filled_block(i)
            for j in range(len(r)):
                yield self._row_dict(t[j], lp[j], r[j])
        if self._legacy is not None:
            yield from list(self._legacy)


# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = FlatLogprobs | list[LogprobsOnePosition | None]
# {token_id -> logprob} for each sequence group.
SampleLogprobs = FlatLogprobs | ArrayLogprobs | list[LogprobsOnePosition]


def create_prompt_logprobs(flat_logprobs: bool) -> PromptLogprobs:
    """Creates a container to store prompt logprobs for a request"""
    logprobs: PromptLogprobs = FlatLogprobs() if flat_logprobs else []
    # NOTE: logprob of first prompt token is None.
    logprobs.append(None)
    return logprobs


def create_sample_logprobs(
    flat_logprobs: bool,
    array_logprobs: bool = False,
    wire_base64: bool = False,
    wire_topk_only: bool = False,
) -> SampleLogprobs:
    """Creates a container to store decode logprobs for a request"""
    if array_logprobs:
        return ArrayLogprobs(wire_base64=wire_base64, wire_topk_only=wire_topk_only)
    return FlatLogprobs() if flat_logprobs else []


def append_logprobs_for_next_position(
    request_logprobs: PromptLogprobs | SampleLogprobs,
    token_ids: list[int],
    logprobs: list[float],
    decoded_tokens: Iterable[str | None],
    rank: int,
    num_logprobs: int,
) -> None:
    """Appends logprobs for the next position"""
    if num_logprobs == -1:
        num_logprobs = len(logprobs)
    # We do not need a special case for the sampled token
    # being in the topk, since inserting duplicated data
    # into a dictionary twice is the same as doing it once.
    topk_ranks = range(1, num_logprobs + 1)
    ranks = itertools.chain((rank,), topk_ranks)

    if isinstance(request_logprobs, FlatLogprobs):
        request_logprobs.append_fast(token_ids, logprobs, ranks, decoded_tokens)
    else:
        request_logprobs.append(
            {
                token_id: Logprob(
                    logprob=logprob,
                    rank=rank,
                    decoded_token=token,
                )
                for token_id, logprob, rank, token in zip(
                    token_ids, logprobs, ranks, decoded_tokens
                )
            }
        )
