# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from collections.abc import Callable, Iterable, Iterator, MutableSequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, final, overload

from vllm.logger import init_logger

if TYPE_CHECKING:
    import numpy as np

    from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)


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


# name -> (factory, skip_sampled_text)
_SAMPLE_LOGPROBS_CONTAINERS: dict[str, tuple[Callable[..., Any], bool]] = {}


def register_sample_logprobs_container(
    name: str, factory: Callable[["SamplingParams"], Any], *, skip_sampled_text: bool
) -> None:
    """Register a frontend-only sample-logprobs container (API server process),
    selected per request with :func:`set_sample_logprobs_container`, for
    ``FINAL_ONLY`` requests. ``factory(sampling_params)`` returns the request's
    container, which receives ``append_rows(token_ids [n, k + 1], logprobs
    [n, k + 1], ranks [n])`` (slot 0 = sampled token). A failing factory or
    container fails its request only. ``skip_sampled_text``: responses carry
    no sampled text (it is detokenized only for stop strings)."""
    current = _SAMPLE_LOGPROBS_CONTAINERS.get(name)
    if current is not None and current[0] is not factory:
        raise ValueError(f"Sample logprobs container {name!r} already registered")
    _SAMPLE_LOGPROBS_CONTAINERS[name] = (factory, skip_sampled_text)


def set_sample_logprobs_container(params: "SamplingParams", name: str | None) -> None:
    """Select a registered container for this request (server-side only: the
    selector is private, so clients and the EngineCore wire never see it)."""
    if name is not None and name not in _SAMPLE_LOGPROBS_CONTAINERS:
        raise ValueError(f"Unknown sample logprobs container {name!r}")
    params._sample_logprobs_container = name


def sample_logprobs_skip_text(params: "SamplingParams") -> bool:
    """Whether the selected container's responses carry no sampled text."""
    entry = _SAMPLE_LOGPROBS_CONTAINERS.get(params._sample_logprobs_container or "")
    return entry is not None and entry[1]


@final  # (lets type checkers narrow on `type(x) is SampleLogprobsHandle`)
class SampleLogprobsHandle:
    """Core-owned holder of a registered container: a failure of the container
    turns the handle broken and fails its request only (the endpoint checks
    :attr:`broken` and calls :meth:`unwrap`). ``len()`` counts the positions
    the engine emitted."""

    def __init__(self, container: Any) -> None:
        self._container = container  # None: broken
        self._count = 0

    @property
    def broken(self) -> bool:
        return self._container is None

    def unwrap(self) -> Any:
        if self._container is None:
            raise ValueError("Sample logprobs are unavailable for this request")
        return self._container

    def append_engine_rows(
        self,
        token_ids: "np.ndarray",
        logprobs: "np.ndarray",
        ranks: "np.ndarray",
        num_slots: int | None,
        cumulative_logprob: float,
    ) -> float:
        """Append this request's rows of an engine step, ``num_slots`` (k + 1)
        columns or all if None; returns the cumulative logprob plus the
        sampled tokens' logprobs, added row by row like the default path."""
        try:
            self._count += len(ranks)
            if self._container is None:
                return cumulative_logprob
            if not len(token_ids) == len(logprobs) == len(ranks):
                raise ValueError("Engine logprob rows of different lengths")
            if num_slots is not None:
                token_ids, logprobs = token_ids[:, :num_slots], logprobs[:, :num_slots]
            for value in logprobs[:, 0].tolist():
                cumulative_logprob += value
            self._container.append_rows(token_ids, logprobs, ranks)
        except Exception:
            logger.exception("Storing sample logprobs failed; failing the request")
            self._container = None
        return cumulative_logprob

    def __len__(self) -> int:
        return self._count


# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = FlatLogprobs | list[LogprobsOnePosition | None]
# {token_id -> logprob} for each sequence group.
SampleLogprobs = FlatLogprobs | list[LogprobsOnePosition]
# What the frontend stores per request (a handle: for a registered container).
SampleLogprobsStorage = SampleLogprobs | SampleLogprobsHandle


def create_prompt_logprobs(flat_logprobs: bool) -> PromptLogprobs:
    """Creates a container to store prompt logprobs for a request"""
    logprobs: PromptLogprobs = FlatLogprobs() if flat_logprobs else []
    # NOTE: logprob of first prompt token is None.
    logprobs.append(None)
    return logprobs


def create_sample_logprobs(
    flat_logprobs: bool, sampling_params: "SamplingParams | None" = None
) -> SampleLogprobsStorage:
    """Creates a container to store decode logprobs for a request"""
    from vllm.sampling_params import RequestOutputKind

    params = sampling_params
    name = params._sample_logprobs_container if params else None
    if params is not None and name is not None:
        if params.output_kind == RequestOutputKind.FINAL_ONLY:
            handle = SampleLogprobsHandle(None)
            try:
                handle._container = _SAMPLE_LOGPROBS_CONTAINERS[name][0](params)
            except Exception:
                logger.exception("Sample logprobs container %r failed", name)
            return handle
        logger.error("Sample logprobs container %r ignored: not FINAL_ONLY", name)
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
