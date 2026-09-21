# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tensors a module reaches through objects it owns rather than through the
module tree.

``named_parameters`` and ``named_buffers`` stop at the module tree, but a
layer's kernels and quant methods are plain objects, and a captured CUDA graph
reads the tensors they hold exactly as it reads a buffer. MXFP4 keeps its
SwiGLU constants on ``quant_method.moe_kernel.fused_experts`` and its
permutation indices in a dict on the quant method; ``process_weights_after_loading``
rebuilds the kernel, which allocates fresh constants and leaves the graph
reading freed storage. The layerwise path restores the parameters it captured,
and no path restores these.

`snapshot` records where each such tensor lives, and `restore` puts the values a
post-load pass produced back into that storage.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import torch

__all__ = [
    "OwnedTensor",
    "REFRESH",
    "snapshot_module_owned_tensors",
    "snapshot_owned_tensors",
    "restore_owned_tensors",
    "refresh_owned_state",
]

# An object a module owns implements this to rebuild whatever it derived from
# the weights. Restoring storage is not enough for state that is never rebuilt:
# `MoERunner` fuses the router and shared-expert gate weights on the first
# forward and guards on whether it has ever done so, so after an update it goes
# on routing with the previous checkpoint's gate. The refresh must write through
# the tensor it already holds, for the same reason everything else here must.
REFRESH = "refresh_after_weight_reload"

# How far to follow a module's own objects. A quant method reaches its constants
# in three hops (`quant_method.moe_kernel.fused_experts.gemm1_alpha`); nothing
# in tree needs more, and a bound keeps an unexpectedly large graph from being
# walked.
_MAX_DEPTH = 4

# `vars(module)` is where `torch.nn.Module` keeps its own bookkeeping, so
# walking it blindly re-finds every parameter and buffer under a second name and
# would have `restore` write into the registry rather than through it.
_MODULE_INTERNALS = frozenset(
    {"_parameters", "_buffers", "_modules", "_non_persistent_buffers_set"}
)


@dataclass(frozen=True)
class OwnedTensor:
    """One tensor, and the slot it is reachable through."""

    owner: object
    key: str | int | object
    tensor: torch.Tensor

    def get(self) -> torch.Tensor | None:
        """The tensor currently in the slot, or None if it is gone."""
        if isinstance(self.owner, (dict, list)):
            try:
                current = self.owner[self.key]  # type: ignore[index]
            except (KeyError, IndexError):
                return None
        else:
            current = getattr(self.owner, str(self.key), None)
        return current if isinstance(current, torch.Tensor) else None

    @property
    def writable(self) -> bool:
        """A tuple's slot cannot be rewritten, so a rebuild there is unrepairable."""
        return not isinstance(self.owner, tuple)

    def set(self, tensor: torch.Tensor) -> None:
        if isinstance(self.owner, (dict, list)):
            self.owner[self.key] = tensor  # type: ignore[index]
        else:
            setattr(self.owner, str(self.key), tensor)


def snapshot_module_owned_tensors(
    module: torch.nn.Module, prefix: str = ""
) -> dict[str, OwnedTensor]:
    """Off-tree tensors ``module`` itself owns, not its submodules'.

    Each module keeps its own snapshot, so the layerwise path can restore one
    layer as it finishes without walking the rest of the model.
    """
    found: dict[str, OwnedTensor] = {}
    for attr, value in vars(module).items():
        if attr in _MODULE_INTERNALS:
            continue
        name = f"{prefix}.{attr}" if prefix else attr
        for owned_name, owned in _walk(value, module, attr, name, set(), 0):
            found[owned_name] = owned
    return found


def snapshot_owned_tensors(root: torch.nn.Module) -> dict[str, OwnedTensor]:
    """Every off-tree tensor reachable from ``root``'s modules, by name."""
    found: dict[str, OwnedTensor] = {}
    for prefix, module in root.named_modules():
        found.update(snapshot_module_owned_tensors(module, prefix))
    return found


def restore_owned_tensors(
    before: dict[str, OwnedTensor], after: dict[str, OwnedTensor]
) -> list[str]:
    """Write the values in ``after`` into the storage ``before`` recorded.

    This is what `_copy_and_restore_kernel_tensors` does for the parameters the
    module tree holds, applied to the tensors it cannot see. A post-load pass is
    free to rebuild its kernel; what it may not do is leave a graph reading the
    storage that kernel no longer uses.

    Matching is by name rather than by slot, because rebuilding replaces the
    objects along the path: the new constants live on a new ``fused_experts``,
    and the old one still holds the old values. So the new value is copied into
    the old storage and that storage is put into the *new* slot.

    Returns:
        Names that could not be restored, because the name is gone or now holds
        a tensor of a different shape, stride or dtype. Those are not a
        relocation the caller can repair, and leave the model undefined.

    """
    unrestorable = []
    for name, old in before.items():
        new = after.get(name)
        if new is None:
            unrestorable.append(name)
            continue
        if new.tensor is old.tensor:
            continue
        if (
            not new.writable
            or new.tensor.shape != old.tensor.shape
            or new.tensor.stride() != old.tensor.stride()
            or new.tensor.dtype != old.tensor.dtype
        ):
            unrestorable.append(name)
            continue
        old.tensor.copy_(new.tensor)
        new.set(old.tensor)
    return unrestorable


def refresh_owned_state(module: torch.nn.Module) -> None:
    """Have every object ``module`` owns rebuild what it derived from weights.

    Runs before the storage snapshot is restored, so a refresh that allocates
    instead of writing in place is caught by `restore_owned_tensors` rather than
    left for a captured graph to read.
    """
    for owned in _owned_objects(module):
        getattr(owned, REFRESH)()


def _owned_objects(module: torch.nn.Module) -> Iterator[object]:
    seen: set[int] = set()
    for attr, value in vars(module).items():
        if attr not in _MODULE_INTERNALS:
            yield from _walk_objects(value, seen, 0)


def _walk_objects(value: object, seen: set[int], depth: int) -> Iterator[object]:
    if depth >= _MAX_DEPTH or isinstance(value, torch.Tensor) or id(value) in seen:
        return
    seen.add(id(value))
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_objects(item, seen, depth + 1)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from _walk_objects(item, seen, depth + 1)
        return
    if isinstance(value, torch.nn.Module) or not type(value).__module__.startswith(
        "vllm."
    ):
        return
    if callable(getattr(value, REFRESH, None)):
        yield value
    for item in vars(value).values():
        yield from _walk_objects(item, seen, depth + 1)


def _walk(
    value: object,
    owner: object,
    key: str | int | object,
    prefix: str,
    seen: set[int],
    depth: int,
) -> Iterator[tuple[str, OwnedTensor]]:
    if isinstance(value, torch.Tensor):
        # Not marked seen: every name a graph may read through is wanted, so a
        # tensor reachable twice is recorded under both names.
        yield prefix, OwnedTensor(owner, key, value)
        return
    if depth >= _MAX_DEPTH or id(value) in seen:
        return
    seen.add(id(value))

    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk(item, value, index, f"{prefix}[{index}]", seen, depth + 1)
        return
    if isinstance(value, dict):
        for dict_key, item in value.items():
            yield from _walk(
                item, value, dict_key, f"{prefix}[{dict_key!r}]", seen, depth + 1
            )
        return
    # A Module's own tensors are already named by the module tree. Only vLLM's
    # own classes are followed past that: a third-party object may hold
    # anything, and its identity is not ours to police.
    if isinstance(value, torch.nn.Module) or not type(value).__module__.startswith(
        "vllm."
    ):
        return
    for attr, item in vars(value).items():
        yield from _walk(item, value, attr, f"{prefix}.{attr}", seen, depth + 1)
