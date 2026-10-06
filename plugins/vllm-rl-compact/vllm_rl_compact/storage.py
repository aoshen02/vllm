# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sample logprobs of compact requests: the core's ``ArrayLogprobs`` engine
rows, plus what compact needs on top of them: the DELTA slices and merges of
streaming responses, and the wire encoder (rows base64-encoded as they
arrive, for non-streaming responses).

Registered with ``vllm.logprobs.register_sample_logprobs_container`` and fed
by the core ``LogprobsProcessor`` through ``append_rows``.
"""

import operator
from collections.abc import Iterator
from dataclasses import InitVar, dataclass, field
from typing import ClassVar, overload

import numpy as np
import pybase64

from vllm.entrypoints.scale_out.token_in_token_out import array_logprobs
from vllm.logger import init_logger
from vllm.logprobs import LogprobsOnePosition

logger = init_logger(__name__)

_INT32 = np.dtype("<i4")
_FLOAT32 = np.dtype("<f4")


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
        if data.size == 0:
            return  # memoryview.cast() rejects zero-size shapes
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
    little-endian ``int32`` token ids and ``float32`` logprobs of engine slots
    ``1..S-1`` (the top-k; the rows were checked to fit). Slot 0 and the
    ranks are kept in small side streams, so the rows can still be decoded
    losslessly.
    """

    def __init__(self) -> None:
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
        """Encode the rows, or return False if their width changed (the
        container then keeps them in array mode)."""
        width = token_ids.shape[1]
        if self.slots is not None and width != self.slots:
            return False
        # Convert everything first: a failure here leaves the streams intact.
        writes = [
            (self.token_ids, np.ascontiguousarray(token_ids[:, 1:], dtype=_INT32)),
            (self.logprobs, np.ascontiguousarray(logprobs[:, 1:], dtype=_FLOAT32)),
            (self.sampled_ids, np.ascontiguousarray(token_ids[:, 0], dtype=_INT32)),
            (
                self.sampled_logprobs,
                np.ascontiguousarray(logprobs[:, 0], dtype=_FLOAT32),
            ),
            (self.ranks, np.ascontiguousarray(ranks, dtype=_INT32)),
        ]
        self.slots = width
        for stream, data in writes:
            stream.write(data)
        self.num_rows += len(ranks)
        return True

    def decode(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        slots = self.slots or 0
        n = self.num_rows
        stored = max(slots - 1, 0)
        token_ids = np.frombuffer(self.token_ids.decode(), dtype=_INT32)
        logprobs = np.frombuffer(self.logprobs.decode(), dtype=_FLOAT32)
        ranks = np.frombuffer(self.ranks.decode(), dtype=_INT32)
        token_ids = token_ids.reshape(n, stored)
        logprobs = logprobs.reshape(n, stored)
        if slots:
            sampled_ids = np.frombuffer(self.sampled_ids.decode(), dtype=_INT32)
            sampled = np.frombuffer(self.sampled_logprobs.decode(), dtype=_FLOAT32)
            token_ids = np.column_stack((sampled_ids, token_ids))
            logprobs = np.column_stack((sampled, logprobs))
        return token_ids, logprobs, ranks


@dataclass
class ArrayLogprobs(array_logprobs.ArrayLogprobs):
    """The core's engine-row storage (see its docstring) for compact
    requests, plus:

    - contiguous slices, exact-sized views of the stored rows (no
      allocation), and ``extend``, for DELTA (streaming) outputs; a slice
      records the positions its source held (:attr:`source_positions`);
    - positional access (``container[i]``), materializing the same
      ``dict[int, Logprob]`` the list representation would hold;
    - with ``wire_base64=True`` (non-streaming compact responses) rows are
      not stored as arrays: they are encoded immediately into the compact
      wire format (base64 of ``<i4``/``<f4``/``<i4``), about 1.33x the raw
      int32/float32 size, so building the response only stitches the
      segments together (:meth:`wire_parts`). Any other access first decodes
      the rows back into an array block, and rows the wire cannot represent
      (a width change) switch the container back to array storage.
    """

    # For slices: positions the source container held when it was sliced
    # (DELTA outputs slice suffixes of the cumulative container).
    source_positions: int | None = None
    wire_base64: InitVar[bool] = False
    _wire: _WireEncoder | None = field(default=None, init=False, repr=False)

    def __post_init__(self, wire_base64: bool) -> None:
        if wire_base64:
            self._wire = _WireEncoder()

    def _unwire(self) -> None:
        """Leave wire mode: decode the encoded rows into an array block."""
        wire = self._wire
        if wire is None:
            return
        if wire.num_rows:
            token_ids, logprobs, ranks = wire.decode()
            # Writable copies, like array-mode storage.
            self.token_id_chunks = [token_ids.copy()]
            self.logprob_chunks = [logprobs.copy()]
            self.rank_chunks = [ranks.copy()]
            self._tail_fill = wire.num_rows
        self._wire = None

    def wire_parts(self) -> tuple[int, int | None, list[bytes], list[bytes]] | None:
        """``(N, S, token_ids, logprobs)``: base64 parts of the top-k slots
        ``1..S-1`` (``S`` is the engine row width), or None if the container
        is not (or no longer) in wire mode."""
        wire = self._wire
        if wire is None:
            return None
        return wire.num_rows, wire.slots, wire.token_ids.parts(), wire.logprobs.parts()

    def mark_broken(self) -> None:
        self._wire = None
        super().mark_broken()

    def _append_rows(
        self, token_ids: np.ndarray, logprobs: np.ndarray, ranks: np.ndarray, n: int
    ) -> None:
        if self._wire is not None and not self.broken:
            self._check_rows(token_ids, logprobs, ranks, n)
            if self._wire.try_write(token_ids, logprobs, ranks):
                self.num_positions += n
                return
            self._unwire()
        super()._append_rows(token_ids, logprobs, ranks, n)

    @property
    def num_slots(self) -> int | None:
        if self._wire is not None:
            return self._wire.slots
        return super().num_slots

    def reserved_bytes(self) -> int:
        """Bytes allocated by the blocks (used and unused rows)."""
        return sum(
            a.nbytes
            for chunks in (self.token_id_chunks, self.logprob_chunks, self.rank_chunks)
            for a in chunks
        )

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self._check_usable()
        self._unwire()
        return super().arrays()

    def __iter__(self) -> Iterator[LogprobsOnePosition]:
        self._check_usable()
        self._unwire()
        return super().__iter__()

    def _filled_block(self, i: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The used ``(token_ids, logprobs, ranks)`` rows of block ``i``."""
        t, lp, r = self.token_id_chunks[i], self.logprob_chunks[i], self.rank_chunks[i]
        if i == len(self.rank_chunks) - 1:
            fill = self._tail_fill
            return t[:fill], lp[:fill], r[:fill]
        return t, lp, r

    def extend(self, values) -> None:
        """Append the positions of ``values`` (DELTA aggregation in
        ``RequestOutput.add``). Never raises for ArrayLogprobs sources: a
        failure marks this container broken (the request fails at render)."""
        if isinstance(values, ArrayLogprobs):
            count = self.num_positions + len(values)
            try:
                self._extend(values)
            except Exception:
                logger.exception("Merging sample logprobs failed; failing the request")
                self.mark_broken()
            if self.broken:
                self.num_positions = count
            if values.source_positions is not None:
                # Merged DELTA outputs: the newest slice's source count.
                self.source_positions = values.source_positions
            return
        # Other position containers (list, FlatLogprobs): keep legacy dicts.
        positions = list(values)
        if not self.broken:
            self._legacy = [*(self._legacy or []), *positions]
        self.num_positions += len(positions)

    def _extend(self, values: "ArrayLogprobs") -> None:
        if values.broken or self.broken:
            # Merging with a broken side: the merged positions are unusable,
            # so nothing is decoded, copied or re-attached (the caller counts
            # the positions).
            if not self.broken:
                self.mark_broken()
            return
        values._unwire()
        # Snapshot first: ``values`` may be ``self``.
        blocks = list(values._filled_blocks())
        legacy = list(values._legacy) if values._legacy is not None else None
        for t, lp, r in blocks:
            self.append_rows(t, lp, r)
            if self.broken:  # became broken mid-merge: stop copying
                return
        if legacy is not None:
            self._legacy = [*(self._legacy or []), *legacy]
            self.num_positions += len(legacy)

    @overload
    def __getitem__(self, position: int) -> LogprobsOnePosition: ...

    @overload
    def __getitem__(self, s: slice, /) -> "ArrayLogprobs": ...

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.num_positions)
            if step != 1:
                raise ValueError("ArrayLogprobs only supports contiguous slices")
            if not self.broken:
                # DELTA slicing runs while building request outputs inside
                # the OutputProcessor loop: never raise from here.
                try:
                    self._unwire()
                    return self._slice(index)
                except Exception:
                    logger.exception(
                        "Slicing sample logprobs failed; failing the request"
                    )
                    count = self.num_positions
                    self.mark_broken()
                    self.num_positions = count
            # Slices of a broken container stay broken.
            result = ArrayLogprobs(source_positions=self.num_positions)
            result.broken = True
            result.num_positions = max(stop - start, 0)
            return result
        self._check_usable()
        self._unwire()
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

    @property
    def _array_positions(self) -> int:
        return self.num_positions - (len(self._legacy) if self._legacy else 0)

    def _slice(self, index: slice) -> "ArrayLogprobs":
        start, stop, _ = index.indices(self.num_positions)
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
        for t, lp, r in reversed(pieces):
            result.token_id_chunks.append(t)
            result.logprob_chunks.append(lp)
            result.rank_chunks.append(r)
            result.num_positions += len(r)
        if pieces:
            result._tail_fill = len(result.rank_chunks[-1])
        return result
