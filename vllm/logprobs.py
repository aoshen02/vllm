# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Iterable, Iterator, MutableSequence
from dataclasses import dataclass, field
from typing import overload

import numpy as np


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


@dataclass
class ArrayLogprobs(MutableSequence[LogprobsOnePosition | None]):
    """
    Sample logprobs of a request kept as the engine's ``LogprobsLists`` rows.

    Positions are stored as a list of numpy chunks (one per engine output),
    so accumulating ``N`` positions with ``S`` slots creates ``O(chunks)``
    Python objects instead of ``O(N * S)``. Row ``i`` is exactly the engine
    row: slot 0 holds the sampled token, slots ``1..S-1`` the top-k
    candidates in engine order. Candidate tokens are never detokenized.

    ``token_ids`` / ``ranks`` chunks are ``int32``; ``logprobs`` chunks keep
    the raw engine values, including non-finite ones, as ``float32`` (the
    engine dtype) or ``float64`` if the engine produced float64.

    Positional access (``container[i]``) materializes the same
    ``dict[int, Logprob]`` the list representation would hold (with
    ``decoded_token=None``), so generic consumers keep working; fast
    consumers use :meth:`arrays` instead.
    """

    token_id_chunks: list[np.ndarray] = field(default_factory=list)
    logprob_chunks: list[np.ndarray] = field(default_factory=list)
    rank_chunks: list[np.ndarray] = field(default_factory=list)
    num_positions: int = 0

    def append_rows(
        self, token_ids: np.ndarray, logprobs: np.ndarray, ranks: np.ndarray
    ) -> None:
        """Append ``n`` positions given as ``[n, S]``, ``[n, S]`` and ``[n]``
        arrays. The arrays are copied, so engine buffers are not retained."""
        n = len(ranks)
        if n == 0:
            return
        if self.token_id_chunks and (
            token_ids.shape[1] != self.token_id_chunks[0].shape[1]
        ):
            raise ValueError("All positions must have the same number of slots")
        self.token_id_chunks.append(np.array(token_ids, dtype="<i4", order="C"))
        float_dtype = "<f8" if logprobs.dtype == np.float64 else "<f4"
        self.logprob_chunks.append(np.array(logprobs, dtype=float_dtype, order="C"))
        self.rank_chunks.append(np.array(ranks, dtype="<i4", order="C"))
        self.num_positions += n

    @property
    def num_slots(self) -> int | None:
        """Slots per position, or None if no position was stored yet."""
        if not self.token_id_chunks:
            return None
        return self.token_id_chunks[0].shape[1]

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return contiguous ``(token_ids[N, S], logprobs[N, S], ranks[N])``.

        Chunks are concatenated once and kept as a single chunk afterwards.
        Returns ``[0, 0]``-shaped arrays when there are no positions.
        """
        if not self.token_id_chunks:
            return (
                np.empty((0, 0), dtype="<i4"),
                np.empty((0, 0), dtype="<f4"),
                np.empty((0,), dtype="<i4"),
            )
        if len(self.token_id_chunks) > 1:
            self.token_id_chunks = [np.concatenate(self.token_id_chunks)]
            self.logprob_chunks = [np.concatenate(self.logprob_chunks)]
            self.rank_chunks = [np.concatenate(self.rank_chunks)]
        return self.token_id_chunks[0], self.logprob_chunks[0], self.rank_chunks[0]

    def extend(self, values) -> None:
        if isinstance(values, ArrayLogprobs):
            for t, lp, r in zip(
                values.token_id_chunks, values.logprob_chunks, values.rank_chunks
            ):
                self.append_rows(t, lp, r)
            return
        raise TypeError("ArrayLogprobs can only be extended with ArrayLogprobs")

    def __len__(self) -> int:
        """Gets number of positions stored in the container"""
        return self.num_positions

    @overload
    def __getitem__(self, position: int) -> LogprobsOnePosition: ...

    @overload
    def __getitem__(self, s: slice, /) -> "ArrayLogprobs": ...

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.num_positions)
            if step != 1:
                raise ValueError("ArrayLogprobs only supports contiguous slices")
            result = ArrayLogprobs()
            if stop <= start:
                return result
            # Walk chunks; slices used by the output processor are suffixes.
            offset = 0
            for t, lp, r in zip(
                self.token_id_chunks, self.logprob_chunks, self.rank_chunks
            ):
                n = len(r)
                lo, hi = max(start - offset, 0), min(stop - offset, n)
                if lo < hi:
                    result.token_id_chunks.append(t[lo:hi])
                    result.logprob_chunks.append(lp[lo:hi])
                    result.rank_chunks.append(r[lo:hi])
                    result.num_positions += hi - lo
                offset += n
                if offset >= stop:
                    break
            return result
        if not isinstance(index, int):
            raise TypeError(f"Invalid index type: {type(index)}")
        if index < 0:
            index += self.num_positions
        if not 0 <= index < self.num_positions:
            raise IndexError("ArrayLogprobs index out of range")
        for t, lp, r in zip(
            self.token_id_chunks, self.logprob_chunks, self.rank_chunks
        ):
            if index < len(r):
                ids = t[index].tolist()
                values = lp[index].tolist()
                ranks = itertools.chain((int(r[index]),), range(1, len(ids)))
                # Same insertion/overwrite semantics as the list[dict] path.
                return {
                    token_id: Logprob(logprob=value, rank=rank)
                    for token_id, value, rank in zip(ids, values, ranks)
                }
            index -= len(r)
        raise AssertionError("unreachable")

    def __setitem__(self, item, value) -> None:
        raise TypeError("Cannot set logprobs in ArrayLogprobs")

    def __delitem__(self, item) -> None:
        raise TypeError("Cannot delete logprobs from ArrayLogprobs")

    def insert(self, index: int, value: dict[int, Logprob] | None) -> None:
        raise TypeError("Cannot insert logprobs to ArrayLogprobs")

    def __iter__(self) -> Iterator[LogprobsOnePosition]:
        for i in range(self.num_positions):
            yield self.__getitem__(i)


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
    flat_logprobs: bool, array_logprobs: bool = False
) -> SampleLogprobs:
    """Creates a container to store decode logprobs for a request"""
    if array_logprobs:
        return ArrayLogprobs()
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
