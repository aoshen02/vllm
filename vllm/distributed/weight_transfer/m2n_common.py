# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the NCCL M2N (`nccl_m2n`) weight transfer backend.

M2N reshards a tensor between two disjoint meshes of ranks that live in one
communicator: the trainer occupies ranks `[0, T)` and the inference workers
`[T, T + N)`. This module holds everything both sides need — the optional
runtime import, layout descriptors, and the conversion to `nccl.m2n` types —
so the engine module stays about the transfer itself.
"""

import hashlib
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.distributed.weight_transfer.base import ParamMeta, Shard
from vllm.distributed.weight_transfer.nccl_common import decode_nccl_unique_id
from vllm.logger import init_logger
from vllm.utils.nccl import unpinned_nccl_env

if TYPE_CHECKING:
    from vllm.distributed.utils import StatelessProcessGroup

logger = init_logger(__name__)

# Placement code for a mesh axis that replicates. Any other (non-negative) code
# is the tensor dim that axis shards. Ints rather than `nccl.m2n` objects so
# layouts survive the JSON init handshake without a custom encoder.
REPLICATE = -1

# `NCCL_RESHARD_MAX_TENSOR_DIMS`; 4-D and higher are rejected by the library.
MAX_TENSOR_DIMS = 3

MESH_NDIMS = 2

# Destination mesh convention. Keeping these roles named avoids scattering
# positional axis assumptions through the worker-side planner.
DESTINATION_REPLICA_AXIS = 0
DESTINATION_SHARD_AXIS = 1

# Conservative capacities accepted by the supported nccl_m2n algorithms. They
# are not exposed by its Python API yet; remove these mirrors once nccl_m2n
# provides communication-free plan validation.
MAX_SOURCE_SHARDS = 16
MAX_DEST_SHARDS = 64

# Increment when the JSON init payload changes incompatibly.
M2N_WIRE_SCHEMA_VERSION = 2

SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset(
    dtype
    for dtype in (
        getattr(torch, name, None)
        for name in (
            "int8",
            "uint8",
            "float8_e4m3fn",
            "float8_e5m2",
            "float16",
            "bfloat16",
            "int32",
            "uint32",
            "float32",
            "int64",
            "uint64",
            "float64",
        )
    )
    if dtype is not None
)

_IMPORT_HINT = (
    "The nccl_m2n weight transfer backend requires the `nccl-extensions` "
    "package (and its `nccl4py` dependency), which is not installed by vLLM. "
    "See https://github.com/NVIDIA/nccl-extensions for build "
    "and install instructions. It needs NCCL 2.30.5 or newer."
)


def prepare_m2n_local_runtime(m2n: Any, max_cta: int | None) -> tuple[Any, int, Any]:
    """Prepare rank-local resources before any rank enters NCCL init."""
    import nccl.core as nccl

    device = torch.accelerator.current_device_index()
    if not isinstance(device, int) or isinstance(device, bool) or device < 0:
        raise RuntimeError(f"invalid current accelerator device index: {device!r}")
    handle = m2n.Handle.create(m2n.Config(max_cta=max_cta))
    return nccl, device, handle


def create_m2n_communicator(
    nccl: Any,
    group: "StatelessProcessGroup",
    unique_id_bytes: bytes | None,
) -> Any:
    """Create the M2N data-plane communicator through NCCL4Py."""
    if unique_id_bytes is None:
        envelope: dict[str, bytes | str | None] | None = None
        if group.rank == 0:
            try:
                envelope = {
                    "unique_id": bytes(nccl.get_unique_id().as_bytes),
                    "error": None,
                }
            except Exception as exc:
                envelope = {
                    "unique_id": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        envelope = group.broadcast_obj(envelope, src=0)
        if not isinstance(envelope, Mapping) or set(envelope) != {"unique_id", "error"}:
            raise RuntimeError("invalid NCCL4Py unique-id envelope")
        error = envelope["error"]
        received_unique_id = envelope["unique_id"]
        if error is not None:
            if not isinstance(error, str) or not error:
                raise RuntimeError("invalid NCCL4Py unique-id error")
            raise RuntimeError(f"NCCL4Py unique-id generation failed: {error}")
        if not isinstance(received_unique_id, bytes):
            raise RuntimeError("NCCL4Py unique-id broadcast did not return bytes")
        unique_id_bytes = received_unique_id

    unique_id = nccl.UniqueId.from_bytes(unique_id_bytes)
    with unpinned_nccl_env():
        return nccl.Communicator.init(
            nranks=group.world_size,
            rank=group.rank,
            unique_id=unique_id,
        )


def import_m2n() -> Any:
    """Import `nccl.m2n` lazily, with an actionable error when it is missing.

    Deferred so that importing vLLM — or any other weight transfer backend —
    never requires the m2n runtime to be present.
    """
    try:
        import nccl.m2n as m2n
    except ImportError as e:
        raise ImportError(f"{_IMPORT_HINT} (import failed: {e})") from e
    return m2n


@dataclass(frozen=True, slots=True)
class M2NMesh:
    """One side's rank topology — pure topology, no tensor placement.

    Mirrors `ncclMesh_t`: a 2-axis mesh owning the contiguous rank interval
    `[start_rank, start_rank + dims[0] * dims[1])`. There is no 1-D mesh; a
    single-axis topology is spelled with a second axis of size 1.

    A model may use a different factorization for each tensor. For example,
    dense weights can use a DP x TP mesh while expert weights use EDP x EP.
    """

    dims: tuple[int, int]
    start_rank: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", tuple(self.dims))
        if len(self.dims) != MESH_NDIMS or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in self.dims
        ):
            raise ValueError(f"mesh dims must be {MESH_NDIMS} positive ints")
        if (
            not isinstance(self.start_rank, int)
            or isinstance(self.start_rank, bool)
            or self.start_rank < 0
        ):
            raise ValueError(
                f"mesh start_rank must be non-negative, got {self.start_rank}"
            )

    @property
    def size(self) -> int:
        return self.dims[0] * self.dims[1]


# A tensor's placement over its side's mesh: one code per mesh axis
# (`REPLICATE`, or the tensor dim that axis shards), or `REPLICATED` for a
# tensor every rank holds in full.
Placements = tuple[int, int]
REPLICATED: Placements | None = None


@dataclass(frozen=True, slots=True)
class M2NLayout:
    """A tensor's mesh and placement on one side of a transfer."""

    mesh: M2NMesh
    placements: Placements | None

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, M2NMesh):
            raise TypeError("M2N layout mesh must be an M2NMesh")
        if self.placements is not None:
            object.__setattr__(self, "placements", tuple(self.placements))


def check_placements(placements: Placements, context: str = "placements") -> None:
    """Reject placement pairs nccl_m2n cannot express."""
    if len(placements) != MESH_NDIMS:
        raise ValueError(f"{context} must have {MESH_NDIMS} entries")
    if any(not isinstance(code, int) or isinstance(code, bool) for code in placements):
        raise TypeError(f"{context} must contain integer placement codes")
    invalid = [code for code in placements if code < REPLICATE]
    if invalid:
        raise ValueError(
            f"{context} contains invalid placement code {invalid[0]}; valid "
            f"codes are {REPLICATE} (Replicate) and non-negative tensor dimensions"
        )
    num_sharded = sum(code != REPLICATE for code in placements)
    if num_sharded == 0:
        raise ValueError(
            f"{context}: a fully replicated tensor is carried as REPLICATED, "
            "not as two REPLICATE axes"
        )
    if num_sharded == MESH_NDIMS:
        raise ValueError(
            f"{context}: nccl_m2n needs one REPLICATE mesh axis, but "
            f"{placements} shards both"
        )


def resolve_layout(
    mesh: M2NMesh,
    placements: Placements | None,
    context: str = "placements",
) -> tuple[M2NMesh, Placements]:
    """Pair one tensor's placement with the mesh m2n should see for it.

    A replicated tensor needs a size-1 mesh axis to carry a no-op shard, since
    m2n has no `{REPLICATE, REPLICATE}`. It is therefore described over the
    *same rank interval* re-factored as `(size, 1)`. That re-factoring is sound
    precisely because replication is order-independent — every rank holds the
    whole tensor, so it does not matter that `(a, b)` and `(size, 1)` walk the
    interval in a different order. A sharded tensor keeps its side's own
    factorization, where rank order decides who owns which shard.
    """
    if placements is None:
        return M2NMesh((mesh.size, 1), mesh.start_rank), (REPLICATE, 0)
    check_placements(placements, context)
    return mesh, placements


def validate_layout(
    mesh: M2NMesh,
    placements: Placements,
    shape: Sequence[int],
    context: str,
) -> tuple[int, ...]:
    """Validate one resolved layout and return its required local shape."""
    local_shape = list(shape)
    for axis, tensor_dim in enumerate(placements):
        if tensor_dim == REPLICATE:
            continue
        if tensor_dim >= len(shape):
            raise ValueError(
                f"{context} shards tensor dim {tensor_dim}, but parameter has "
                f"rank {len(shape)}"
            )
        factor = mesh.dims[axis]
        if shape[tensor_dim] % factor:
            raise ValueError(
                f"{context} shards dim {tensor_dim} (size {shape[tensor_dim]}) "
                f"over {factor} ranks, which does not divide evenly"
            )
        local_shape[tensor_dim] //= factor
    return tuple(local_shape)


def expected_local_shape(meta: "M2NParamMeta") -> tuple[int, ...]:
    """Validate a parameter's source layout and return its local shape."""
    mesh, placements = resolve_layout(
        meta.source_layout.mesh,
        meta.source_layout.placements,
        f"parameter '{meta.name}' source placements",
    )
    return validate_layout(
        mesh, placements, meta.shape, f"parameter '{meta.name}' source layout"
    )


def check_transferable(name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:
    """Reject parameters nccl_m2n cannot transfer."""
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"parameter '{name}' has dtype {dtype}, which nccl_m2n does not "
            f"support. Supported: {sorted(str(d) for d in SUPPORTED_DTYPES)}"
        )
    if not 1 <= len(shape) <= MAX_TENSOR_DIMS:
        raise ValueError(
            f"parameter '{name}' has rank {len(shape)}; nccl_m2n supports "
            f"rank 1..{MAX_TENSOR_DIMS}"
        )


def _shard_spec(mesh: M2NMesh, placements: Placements) -> tuple[int, int]:
    return next(
        (code, mesh.dims[axis])
        for axis, code in enumerate(placements)
        if code != REPLICATE
    )


def _max_partition_overlaps(
    extent: int,
    partitions: int,
    other_partitions: int,
) -> int:
    """Return the maximum other partitions intersected by any one partition."""
    if extent == 0:
        return 0
    width = extent // partitions
    other_width = extent // other_partitions
    max_overlaps = 0
    for index in range(partitions):
        start = index * width
        end = start + width
        first_other = start // other_width
        last_other = (end - 1) // other_width
        max_overlaps = max(max_overlaps, last_other - first_other + 1)
    return max_overlaps


def check_plan_limits(
    src: tuple[M2NMesh, Placements],
    dst: tuple[M2NMesh, Placements],
    name: str,
    shape: Sequence[int],
) -> None:
    """Reject plans whose per-shard fan-in or fan-out exceeds M2N capacity."""
    src_dim, src_shards = _shard_spec(*src)
    dst_dim, dst_shards = _shard_spec(*dst)
    if any(extent == 0 for extent in shape):
        max_sources = max_targets = 0
    elif src_dim != dst_dim:
        max_sources = src_shards
        max_targets = dst_shards
    else:
        extent = shape[src_dim]
        max_sources = _max_partition_overlaps(extent, dst_shards, src_shards)
        max_targets = _max_partition_overlaps(extent, src_shards, dst_shards)

    if max_sources > MAX_SOURCE_SHARDS:
        raise ValueError(
            f"parameter '{name}' would feed up to {max_sources} source shards "
            f"into one destination shard, over nccl_m2n MAX_SOURCES="
            f"{MAX_SOURCE_SHARDS}"
        )
    if max_targets > MAX_DEST_SHARDS:
        raise ValueError(
            f"parameter '{name}' would fan one source shard out to up to "
            f"{max_targets} destination shards, over nccl_m2n "
            f"MAX_TARGETS={MAX_DEST_SHARDS}"
        )


def validate_transfer_plan(
    metas: Sequence["M2NParamMeta"],
    dst_mesh: M2NMesh,
    dst_placements: Sequence[Placements | None],
) -> None:
    """Validate the complete ordered source-to-destination transfer plan."""
    if len(dst_placements) != len(metas):
        raise ValueError(
            f"destination plan has {len(dst_placements)} parameters, expected "
            f"{len(metas)}"
        )
    for meta, placement in zip(metas, dst_placements):
        check_transferable(meta.name, meta.dtype, meta.shape)
        src = resolve_layout(
            meta.source_layout.mesh,
            meta.source_layout.placements,
            f"parameter '{meta.name}' source placements",
        )
        validate_layout(*src, meta.shape, f"parameter '{meta.name}' source layout")
        dst = resolve_layout(
            dst_mesh,
            placement,
            f"parameter '{meta.name}' destination placements",
        )
        validate_layout(*dst, meta.shape, f"parameter '{meta.name}' destination layout")
        check_plan_limits(src, dst, meta.name, meta.shape)


@dataclass(frozen=True)
class M2NParamMeta(ParamMeta):
    """`ParamMeta` with its m2n source layout spelled out, as the engines plan
    reshards from it. Built from a source's `ParamMeta` by `m2n_param_meta`.
    """

    source_layout: M2NLayout

    def __post_init__(self) -> None:
        if not isinstance(self.source_layout, M2NLayout):
            raise TypeError("source_layout must be an M2NLayout")


def m2n_param_meta(meta: ParamMeta, num_trainer_ranks: int) -> M2NParamMeta:
    """The m2n source layout of `meta`, which must be owned by every trainer
    rank in rank order (or by default).

    A 1-D mesh gains a size-1 second axis; a replicated parameter keeps
    `REPLICATED`. Placements m2n cannot express (both axes sharded, more than
    two axes) are left for `validate_transfer_plan` to reject.
    """
    everyone = tuple(range(num_trainer_ranks))
    if meta.owners is not None and meta.owners != everyone:
        raise ValueError(
            f"parameter '{meta.name}' is owned by trainer ranks {meta.owners}, "
            f"but nccl_m2n needs every parameter owned by ranks {everyone}, in "
            "order"
        )
    mesh_shape = meta.mesh_shape or (num_trainer_ranks,)
    if len(mesh_shape) > MESH_NDIMS:
        raise ValueError(
            f"parameter '{meta.name}' has a {len(mesh_shape)}-D source mesh; "
            f"nccl_m2n supports at most {MESH_NDIMS} axes"
        )
    dims = (mesh_shape[0], mesh_shape[1] if len(mesh_shape) == 2 else 1)
    placements: Placements | None = REPLICATED
    if meta.placements is not None:
        codes = [p.dim if isinstance(p, Shard) else REPLICATE for p in meta.placements]
        codes += [REPLICATE] * (MESH_NDIMS - len(codes))
        placements = cast(Placements, tuple(codes))
    return M2NParamMeta(
        meta.name,
        meta.dtype,
        meta.shape,
        M2NLayout(M2NMesh(dims, 0), placements),
    )


@dataclass(frozen=True, slots=True)
class M2NWireParam:
    """JSON-safe source metadata for one parameter in the init handshake."""

    name: str
    dtype_name: str
    shape: tuple[int, ...]
    src_mesh_dims: tuple[int, int]
    src_placements: Placements | None

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        dims = tuple(self.src_mesh_dims)
        placements = None if self.src_placements is None else tuple(self.src_placements)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "src_mesh_dims", dims)
        object.__setattr__(self, "src_placements", placements)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("M2N wire parameter name must not be empty")
        if not isinstance(self.dtype_name, str) or not self.dtype_name:
            raise ValueError(f"parameter '{self.name}' has an empty wire dtype name")
        if not shape or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
            for dim in shape
        ):
            raise ValueError(
                f"parameter '{self.name}' shape must contain positive integers"
            )
        M2NMesh(cast(tuple[int, int], dims), 0)
        if placements is not None:
            check_placements(
                cast(Placements, placements),
                f"parameter '{self.name}' source placements",
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "M2NWireParam":
        """Parse one strict wire record from its JSON-decoded mapping."""
        fields = {
            "name",
            "dtype_name",
            "shape",
            "src_mesh_dims",
            "src_placements",
        }
        missing = fields - value.keys()
        extra = value.keys() - fields
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if extra:
                details.append(f"unexpected {sorted(extra)}")
            raise ValueError(f"invalid M2N wire parameter: {', '.join(details)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize this record using only JSON-compatible values."""
        return {
            "name": self.name,
            "dtype_name": self.dtype_name,
            "shape": list(self.shape),
            "src_mesh_dims": list(self.src_mesh_dims),
            "src_placements": (
                None if self.src_placements is None else list(self.src_placements)
            ),
        }


def source_plan_digest(params: Sequence[M2NWireParam]) -> str:
    """Return a rank-independent digest of the ordered source wire plan."""
    payload = json.dumps(
        {
            "schema_version": M2N_WIRE_SCHEMA_VERSION,
            "params": [param.to_dict() for param in params],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def complete_plan_digest(
    source_digest: str,
    dst_mesh: M2NMesh,
    dst_placements: Sequence[Placements | None],
) -> str:
    """Return a digest of the agreed source and destination plan."""
    if not isinstance(source_digest, str) or not source_digest:
        raise ValueError("source digest must be a non-empty string")
    payload = json.dumps(
        {
            "schema_version": M2N_WIRE_SCHEMA_VERSION,
            "source_digest": source_digest,
            "destination_mesh": {
                "dims": list(dst_mesh.dims),
                "start_rank": dst_mesh.start_rank,
            },
            "destination_placements": [
                None if placement is None else list(placement)
                for placement in dst_placements
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _preflight_agree(
    group: "StatelessProcessGroup",
    *,
    phase: str,
    label: str,
    error_prefix: str,
    local_error: str | None,
    default_payload: Mapping[str, Any] | None = None,
    build_payload: Callable[[], Mapping[str, Any]] | None = None,
    extract_value: Callable[[Mapping[str, Any]], Any] | None = None,
    validate_values: Callable[[list[tuple[int, Any]]], list[str]] | None = None,
    disagreement_message: str = "",
) -> Any:
    """Gather a per-rank preflight envelope and require unanimous agreement."""
    payload = {} if default_payload is None else dict(default_payload)
    prepared_error = local_error
    if prepared_error is None and build_payload is not None:
        try:
            payload.update(build_payload())
        except Exception as exc:
            prepared_error = f"{type(exc).__name__}: {exc}"
    elif prepared_error is not None and (
        not isinstance(prepared_error, str) or not prepared_error
    ):
        prepared_error = f"TypeError: local {label} error must be a non-empty string"

    envelope = {"phase": phase, "error": prepared_error, **payload}
    gathered = group.all_gather_obj(envelope)
    expected_fields = set(envelope)
    errors: list[str] = []
    if len(gathered) != group.world_size:
        errors.append(
            "control-plane all-gather returned "
            f"{len(gathered)} records for world size {group.world_size}"
        )

    values: list[tuple[int, Any]] = []
    for rank, item in enumerate(gathered):
        if (
            not isinstance(item, Mapping)
            or set(item) != expected_fields
            or item.get("phase") != phase
        ):
            errors.append(f"rank {rank}: invalid {label} envelope")
            continue
        error = item["error"]
        if error is not None:
            if not isinstance(error, str) or not error:
                errors.append(f"rank {rank}: invalid {label} error")
            else:
                errors.append(f"rank {rank}: {error}")
            continue
        if extract_value is not None:
            try:
                values.append((rank, extract_value(item)))
            except ValueError as exc:
                errors.append(f"rank {rank}: {exc}")

    if not errors and values:
        if validate_values is not None:
            errors.extend(validate_values(values))
        else:
            canonical = values[0][1]
            for rank, value in values[1:]:
                if value != canonical:
                    errors.append(f"rank {rank}: {disagreement_message}")
    if errors:
        raise RuntimeError(f"{error_prefix}: " + "; ".join(errors))
    return values[0][1] if values else None


def check_source_plan_agreement(
    group: "StatelessProcessGroup",
    params: Sequence[M2NWireParam],
    expected_digest: str | None = None,
    local_error: str | None = None,
) -> str:
    """Verify every participant entered with the same source plan."""

    def build() -> Mapping[str, Any]:
        digest = source_plan_digest(params)
        declared = digest if expected_digest is None else expected_digest
        if not isinstance(declared, str) or not declared:
            raise ValueError("declared source digest must be a non-empty string")
        return {"digest": digest, "declared_digest": declared}

    def extract(item: Mapping[str, Any]) -> tuple[str, str]:
        computed, claimed = item["digest"], item["declared_digest"]
        if not (
            isinstance(computed, str)
            and computed
            and isinstance(claimed, str)
            and claimed
        ):
            raise ValueError("invalid source digest")
        return computed, claimed

    def validate(values: list[tuple[int, tuple[str, str]]]) -> list[str]:
        canonical = values[0][1][0]
        errs: list[str] = []
        for rank, (computed, claimed) in values:
            if computed != claimed:
                errs.append(
                    f"rank {rank}: computed digest {computed} does not match "
                    f"declared digest {claimed}"
                )
            if computed != canonical:
                errs.append(f"rank {rank}: source digest disagrees with rank 0")
        return errs

    computed, _ = cast(
        tuple[str, str],
        _preflight_agree(
            group,
            phase="source",
            label="source preflight",
            error_prefix="nccl_m2n source preflight rejected before NCCL init",
            local_error=local_error,
            default_payload={"digest": None, "declared_digest": expected_digest},
            build_payload=build,
            extract_value=extract,
            validate_values=validate,
        ),
    )
    return computed


def check_complete_plan_agreement(
    group: "StatelessProcessGroup",
    source_digest: str,
    dst_mesh: M2NMesh,
    dst_placements: Sequence[Placements | None],
    local_error: str | None = None,
) -> str:
    """Require every rank to accept the same complete transfer plan."""

    def extract(item: Mapping[str, Any]) -> str:
        rank_digest = item["digest"]
        if not isinstance(rank_digest, str) or not rank_digest:
            raise ValueError("invalid complete-plan digest")
        return rank_digest

    return cast(
        str,
        _preflight_agree(
            group,
            phase="complete_plan",
            label="complete-plan",
            error_prefix=(
                "nccl_m2n complete-plan preflight rejected before first reshard"
            ),
            local_error=local_error,
            default_payload={"digest": None},
            build_payload=lambda: {
                "digest": complete_plan_digest(source_digest, dst_mesh, dst_placements)
            },
            extract_value=extract,
            disagreement_message="complete-plan digest disagrees with rank 0",
        ),
    )


def check_runtime_ready_agreement(
    group: "StatelessProcessGroup", local_error: str | None
) -> None:
    """Require every rank to prepare local resources before NCCL init."""
    _preflight_agree(
        group,
        phase="runtime_ready",
        label="runtime readiness",
        error_prefix="nccl_m2n local runtime preflight failed before NCCL init",
        local_error=local_error,
    )


def check_data_plane_agreement(
    group: "StatelessProcessGroup",
    unique_id_bytes: bytes | None,
    local_error: str | None = None,
) -> None:
    """Agree on the NCCL rendezvous identity before NCCL init."""

    def build() -> Mapping[str, Any]:
        if unique_id_bytes is None:
            return {"uid_digest": None}
        if not isinstance(unique_id_bytes, bytes):
            raise TypeError("NCCL unique id must decode to bytes")
        return {"uid_digest": hashlib.sha256(unique_id_bytes).hexdigest()}

    def extract(item: Mapping[str, Any]) -> tuple[str, str | None]:
        mode, uid = item["mode"], item["uid_digest"]
        if mode not in {"tcp", "uid"} or (mode == "uid") != isinstance(uid, str):
            raise ValueError("invalid data-plane identity")
        return mode, uid

    _preflight_agree(
        group,
        phase="data_plane",
        label="data-plane preflight",
        error_prefix="nccl_m2n data-plane preflight rejected before NCCL init",
        local_error=local_error,
        default_payload={
            "mode": "uid" if unique_id_bytes is not None else "tcp",
            "uid_digest": None,
        },
        build_payload=build,
        extract_value=extract,
        disagreement_message="NCCL data-plane identity disagrees",
    )


def validate_m2n_rendezvous(
    *,
    master_address: str,
    master_port: int,
    num_trainer_ranks: int,
    world_size: int,
    dst_mesh_dims: Sequence[int] | None,
    split_field: str,
) -> None:
    """Validate shared metadata rendezvous, rank split, and destination mesh."""
    if not isinstance(master_address, str) or not master_address:
        raise ValueError("`master_address` must name the metadata rendezvous")
    if (
        not isinstance(master_port, int)
        or isinstance(master_port, bool)
        or not 0 < master_port < 65536
    ):
        raise ValueError("`master_port` must be a valid metadata port")
    if (
        not isinstance(num_trainer_ranks, int)
        or isinstance(num_trainer_ranks, bool)
        or not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or not 0 < num_trainer_ranks < world_size
    ):
        raise ValueError(
            f"`{split_field}` ({num_trainer_ranks}) must leave at least one "
            f"trainer rank and one worker in world_size {world_size}"
        )
    num_workers = world_size - num_trainer_ranks
    if dst_mesh_dims is not None:
        dims = tuple(dst_mesh_dims)
        try:
            mesh = M2NMesh(cast(tuple[int, int], dims), num_trainer_ranks)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid `dst_mesh_dims` {dims}: {exc}") from exc
        if mesh.size != num_workers:
            raise ValueError(
                f"`dst_mesh_dims` {dims} must be {MESH_NDIMS} dims covering the "
                f"{num_workers} inference workers"
            )


def decode_m2n_unique_id(nccl_unique_id_b64: str | None, ctx: str) -> bytes | None:
    """Decode an optional base64 NCCL unique ID for the M2N data plane."""
    if nccl_unique_id_b64 is None:
        return None
    return decode_nccl_unique_id(
        master_address=None,
        master_port=None,
        nccl_unique_id_b64=nccl_unique_id_b64,
        ctx=ctx,
    )


@contextmanager
def init_m2n_data_plane(
    m2n: Any,
    metadata_group: "StatelessProcessGroup",
    get_unique_id_bytes: Callable[[], bytes | None],
    max_cta: int | None,
    *,
    pre_runtime_error: str | None = None,
) -> Iterator[tuple[Any, Any, int]]:
    """Agree on data-plane/runtime readiness, construct handle & communicator,
    and abort/destroy them if any subsequent init step fails."""
    unique_id_bytes: bytes | None = None
    data_plane_error: str | None = None
    try:
        unique_id_bytes = get_unique_id_bytes()
    except Exception as exc:
        data_plane_error = f"{type(exc).__name__}: {exc}"
    check_data_plane_agreement(metadata_group, unique_id_bytes, data_plane_error)

    handle = None
    group = None
    try:
        runtime_error = pre_runtime_error
        nccl = None
        device = None
        if runtime_error is None:
            try:
                nccl, device, handle = prepare_m2n_local_runtime(m2n, max_cta)
            except Exception as exc:
                runtime_error = f"{type(exc).__name__}: {exc}"
        check_runtime_ready_agreement(metadata_group, runtime_error)
        assert nccl is not None and device is not None and handle is not None
        group = create_m2n_communicator(nccl, metadata_group, unique_id_bytes)
        yield handle, group, device
    except BaseException:
        if group is not None:
            try:
                # Transport cleanup only; reload validity belongs to the
                # model-update lifecycle, not this backend.
                group.abort()
            except BaseException:
                logger.exception(
                    "failed to abort partially initialized nccl_m2n communicator"
                )
        if handle is not None:
            try:
                handle.destroy()
            except BaseException:
                logger.exception(
                    "failed to destroy partially initialized nccl_m2n handle"
                )
        raise


def reshard_m2n_tensor(
    m2n: Any,
    handle: Any,
    comm: Any,
    stream: torch.cuda.Stream,
    *,
    src_tensor: torch.Tensor | None,
    dst_tensor: torch.Tensor | None,
    src_meta: M2NParamMeta,
    dst_mesh: M2NMesh,
    dst_placements: Placements | None,
) -> None:
    """Reshard one parameter between trainer and worker meshes."""
    src_mesh, resolved_src = resolve_layout(
        src_meta.source_layout.mesh, src_meta.source_layout.placements
    )
    resolved_dst_mesh, resolved_dst = resolve_layout(dst_mesh, dst_placements)
    m2n.reshard(
        src_tensor,
        dst_tensor,
        comm,
        stream,
        src_mesh=to_mesh(m2n, src_mesh),
        src_placements=to_placements(m2n, resolved_src),
        dst_mesh=to_mesh(m2n, resolved_dst_mesh),
        dst_placements=to_placements(m2n, resolved_dst),
        handle=handle,
    )


def shutdown_m2n_resources(handle: Any, group: Any) -> None:
    """Finish pending GPU work and release M2N handle and communicator."""
    first_error: BaseException | None = None

    def attempt(action: Any, message: str) -> None:
        nonlocal first_error
        try:
            action()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            logger.exception(message)

    if handle is not None or group is not None:
        # M2N does not synchronize caller streams on finalize.
        attempt(torch.accelerator.synchronize, "failed to synchronize nccl_m2n")
    if handle is not None:
        attempt(handle.destroy, "failed to destroy nccl_m2n handle")
    if group is not None:
        attempt(group.abort, "failed to abort nccl_m2n communicator")
    if first_error is not None:
        raise first_error


def to_mesh(m2n: Any, mesh: M2NMesh) -> Any:
    return m2n.Mesh(mesh.dims, start_rank=mesh.start_rank)


def to_placements(m2n: Any, placements: Placements) -> list[Any]:
    return [
        m2n.Replicate() if code == REPLICATE else m2n.Shard(code) for code in placements
    ]


def publish_destination_placements(
    comm: Any,
    device: torch.device,
    stream: torch.cuda.Stream,
    first_worker_rank: int,
    placements: "Sequence[Placements | None] | None",
    num_parameters: int,
) -> list[Placements | None]:
    """Share the worker-side destination plan with every rank in the group.

    The trainer must issue each reshard with the same destination the workers
    use, but once destinations are per-parameter they depend on the inference
    model, which only the workers can see. The first worker publishes them here,
    over the shared NCCL communicator; trainer ranks pass `None` and receive
    them, and the other workers pass their own so a disagreement is caught
    rather than deadlocking later.

    Only placements travel: the destination *mesh* is still derived from the
    rank split, identically on both sides. The plan is published on the data
    plane before any M2N reshard collective.
    """
    replicated_sentinel = REPLICATE - 1
    encoded = torch.full(
        (num_parameters, MESH_NDIMS),
        replicated_sentinel,
        dtype=torch.int8,
        device=device,
    )
    if comm.rank == first_worker_rank:
        if placements is None or len(placements) != num_parameters:
            raise ValueError(
                f"publishing rank needs {num_parameters} destination placements"
            )
        rows = [
            [replicated_sentinel] * MESH_NDIMS if placement is REPLICATED else placement
            for placement in placements
        ]
        if rows:
            encoded.copy_(torch.tensor(rows, dtype=torch.int8, device=device))

    comm.broadcast(encoded, encoded, root=first_worker_rank, stream=stream)
    return [
        REPLICATED
        if all(code == replicated_sentinel for code in row)
        else cast(Placements, tuple(row))
        for row in encoded.tolist()
    ]
