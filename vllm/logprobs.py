# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import itertools
import threading
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


# name -> (factory, skip_sampled_text), see register_sample_logprobs_container
_SAMPLE_LOGPROBS_CONTAINERS: dict[str, tuple[Callable[..., Any], bool]] = {}
_REGISTRY_LOCK = threading.Lock()


def register_sample_logprobs_container(
    name: str, factory: Callable[["SamplingParams"], Any], *, skip_sampled_text: bool
) -> None:
    """Register a frontend-only sample-logprobs container (e.g. from an endpoint
    plugin, in the API server process), selected per request with
    :func:`set_sample_logprobs_container`.

    ``factory(sampling_params)`` returns the request's container. It receives
    ``append_rows(token_ids [n, k + 1], logprobs [n, k + 1], ranks [n])``
    (slot 0 = sampled token) and must support ``len()``, contiguous slices and
    ``extend(other_container)``. Core only ever holds it inside a
    :class:`SampleLogprobsHandle`: any failure of the factory or the container
    fails that request alone. ``skip_sampled_text``: responses carry no
    sampled text (detokenized only for stop strings). Registering the same
    factory and flag again is a no-op; a different one raises ValueError."""
    if not isinstance(name, str) or not name:
        raise ValueError(f"Invalid sample logprobs container name {name!r}")
    entry = (factory, bool(skip_sampled_text))
    with _REGISTRY_LOCK:  # no third-party code runs under the lock
        current = _SAMPLE_LOGPROBS_CONTAINERS.setdefault(name, entry)
    if current[0] is not factory or current[1] != entry[1]:
        raise ValueError(f"Sample logprobs container {name!r} already registered")


def set_sample_logprobs_container(params: "SamplingParams", name: str | None) -> None:
    """Select a registered container for this request (server-side only: the
    selector is private, so clients and the EngineCore wire never see it)."""
    if name is not None and name not in _SAMPLE_LOGPROBS_CONTAINERS:
        raise ValueError(f"Unknown sample logprobs container {name!r}")
    params._sample_logprobs_container = name


def sample_logprobs_skip_text(params: "SamplingParams") -> bool:
    """Whether the selected container's responses carry no sampled text."""
    name = params._sample_logprobs_container
    entry = None if name is None else _SAMPLE_LOGPROBS_CONTAINERS.get(name)
    return entry is not None and entry[1]


# Container failures contained by the handle. CancelledError is included: in
# the synchronous output loop it can only come from the container itself and
# would otherwise end the output handler. KeyboardInterrupt / SystemExit
# propagate (process-level).
_CONTAINER_ERRORS = (Exception, asyncio.CancelledError)


@final
class SampleLogprobsHandle:
    """Core-owned handle of a registered container: the only object the
    OutputProcessor and ``RequestOutput.add`` see. Positions are counted by
    core (positions the engine emitted for the request); every container call
    is guarded, and a failure turns the handle broken (request-local: the
    request fails when its endpoint unwraps it). Endpoints call
    :meth:`unwrap` outside shared processing."""

    __slots__ = ("_container", "_count")

    def __init__(self, container: Any, count: int = 0) -> None:
        self._container = container  # None: broken
        self._count = count

    @property
    def broken(self) -> bool:
        return self._container is None

    def unwrap(self) -> Any:
        """The container; ValueError if it failed for this request."""
        if self._container is None:
            raise ValueError("Sample logprobs are unavailable for this request")
        return self._container

    def _call(self, method: str, *args: Any) -> Any:
        if self._container is not None:
            try:
                return getattr(self._container, method)(*args)
            except _CONTAINER_ERRORS:
                logger.exception("Sample logprobs container failed; failing request")
                self._container = None
        return None

    def append_rows(self, token_ids: Any, logprobs: Any, ranks: "np.ndarray") -> None:
        try:
            self._count += len(ranks)
        except _CONTAINER_ERRORS:  # malformed engine rows: positions unknown
            self._container = None
        self._call("append_rows", token_ids, logprobs, ranks)

    def extend(self, other: Any) -> None:
        """DELTA aggregation in ``RequestOutput.add`` (never raises)."""
        if type(other) is not SampleLogprobsHandle:  # cannot happen for one request
            self._count += len(other) if type(other) in (list, FlatLogprobs) else 0
            self._container = None
        elif other._container is None:
            self._count += other._count
            self._container = None
        else:
            self._count += other._count
            self._call("extend", other._container)

    def __len__(self) -> int:
        return self._count

    def __bool__(self) -> bool:
        return self._count > 0

    def __getitem__(self, index: slice) -> "SampleLogprobsHandle":
        if not isinstance(index, slice):
            raise TypeError("SampleLogprobsHandle supports slices only; unwrap() it")
        count = len(range(*index.indices(self._count)))
        return SampleLogprobsHandle(self._call("__getitem__", index), count)

    def __iter__(self) -> Iterator[Any]:
        raise TypeError("SampleLogprobsHandle is not iterable; unwrap() it")

    def __repr__(self) -> str:
        return f"SampleLogprobsHandle(positions={self._count}, broken={self.broken})"


# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = FlatLogprobs | list[LogprobsOnePosition | None]
# {token_id -> logprob} for each sequence group.
SampleLogprobs = FlatLogprobs | list[LogprobsOnePosition]


def create_prompt_logprobs(flat_logprobs: bool) -> PromptLogprobs:
    """Creates a container to store prompt logprobs for a request"""
    logprobs: PromptLogprobs = FlatLogprobs() if flat_logprobs else []
    # NOTE: logprob of first prompt token is None.
    logprobs.append(None)
    return logprobs


def create_sample_logprobs(
    flat_logprobs: bool, sampling_params: "SamplingParams | None" = None
) -> SampleLogprobs | SampleLogprobsHandle:
    """Creates a container to store decode logprobs for a request"""
    name = sampling_params._sample_logprobs_container if sampling_params else None
    if sampling_params is None or name is None:
        return FlatLogprobs() if flat_logprobs else []
    entry = _SAMPLE_LOGPROBS_CONTAINERS.get(name)
    if entry is None:  # only via the private field, bypassing the setter
        raise ValueError(f"Unknown sample logprobs container {name!r}")
    container = None  # a factory failing (or returning None) breaks the handle
    try:
        container = entry[0](sampling_params)
    except _CONTAINER_ERRORS:
        logger.exception("Sample logprobs container %r failed", name)
    return SampleLogprobsHandle(container)


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
