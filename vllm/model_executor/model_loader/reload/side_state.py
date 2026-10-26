# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep a layer's derived tensors at their startup storage across a reload.

`process_weights_after_loading` derives tensors from the checkpoint and
registers them (kernel constants, derived scales) or stores them on objects
hanging off the layer: `quant_method.moe_quant_config`,
`quant_method.moe_kernel`, experts objects, plain layer attributes. A reload
reruns it, so the layer ends up holding freshly allocated tensors, while CUDA
graphs captured at startup keep reading the startup storage.

One rule: before the rerun, `snapshot_side_tensors` records every derived
device tensor of the layer (registered without a checkpoint layout, or
reachable from its attributes) by path, as an alias of its storage; after the
rerun, `pin_side_tensors` copies the tensor found at the same path into that
storage and rebinds that tensor object to it (`t.data = startup`), so every
holder of the object reads the startup storage.

A walk that stops at its depth limit refuses the reload. Out of scope (not
walked, so a tensor held only there is not pinned): `functools.partial`
arguments, closure cells, and `__slots__` of a base class whose subclass
declares its own.
"""

import functools
from typing import Any

import torch

__all__ = ["SideWalk", "pin_side_tensors", "snapshot_side_tensors"]

Path = tuple[tuple[str, Any], ...]

_MAX_DEPTH = 16
# packages whose own classes hold no tensors derived from the weights
_LIBRARIES = {"transformers", "tokenizers", "torch", "threading", "logging"}
_SKIP_LAYER_ATTRS = {"_parameters", "_buffers", "_modules", "training"}


_OPAQUE: dict[type, bool] = {}


@functools.cache
def _tensor_free_bases() -> frozenset[type]:
    """Tokenizer, processor and configuration bases. Their subclasses, vLLM's
    included, hold no model tensors: the cached tokenizer a multimodal model
    root keeps subclasses one, with ~250k BPE merges."""
    import transformers

    from vllm.tokenizers.protocol import TokenizerLike

    names = (
        "PreTrainedConfig",
        "PreTrainedTokenizerBase",
        "ProcessorMixin",
        "FeatureExtractionMixin",
        "ImageProcessingMixin",
    )
    return frozenset(
        [TokenizerLike]
        + [getattr(transformers, n) for n in names if hasattr(transformers, n)]
    )


def _opaque(cls: type) -> bool:
    """Objects that hold no derived device tensors: classes of the libraries
    above (by top-level package, so `transformers_modules`, where
    trust_remote_code classes live, is walked), vLLM's configuration classes,
    and subclasses of tokenizer, processor and configuration bases."""
    opaque = _OPAQUE.get(cls)
    if opaque is None:
        module = cls.__module__ or ""
        opaque = _OPAQUE[cls] = (
            module.partition(".")[0] in _LIBRARIES
            or module.startswith("vllm.config")
            or not _tensor_free_bases().isdisjoint(cls.__mro__)
        )
    return opaque


def _children(o: Any) -> list[tuple[tuple[str, Any], Any]]:
    if isinstance(o, (list, tuple)):
        return [(("i", i), v) for i, v in enumerate(o)]
    if isinstance(o, dict):
        return [(("k", k), v) for k, v in o.items()]
    if _opaque(type(o)):
        return []
    d = getattr(o, "__dict__", None)
    out = [(("a", k), v) for k, v in d.items()] if isinstance(d, dict) else []
    for k in getattr(type(o), "__slots__", ()):
        if isinstance(k, str) and hasattr(o, k):
            out.append((("a", k), getattr(o, k)))
    return out


class SideWalk:
    """State shared by the walks of one pass over a model's modules; it must
    not outlive the pass.

    ``clean`` holds the objects proven tensor-free (by id, with the object to
    keep the id valid), so objects every layer references, e.g. a
    quantization config with a dict per checkpoint layer, are walked once per
    pass instead of once per layer. A proof covers the object's whole
    reachable graph, so it holds at any depth. ``truncated`` holds the
    objects reached only past the depth limit, never walked in the pass:
    never proven, never pinned."""

    def __init__(self) -> None:
        self.clean: dict[int, Any] = {}
        self.walked: dict[int, Any] = {}
        self.truncated: dict[int, Any] = {}


def collect_side_tensors(
    layer: torch.nn.Module, side_walk: SideWalk | None = None
) -> dict[Path, torch.Tensor]:
    """The layer's derived tensors, by path: registered tensors without a
    checkpoint layout (the address check covers the checkpoint ones), and
    every tensor reachable from its non-module attributes."""
    found: dict[Path, torch.Tensor] = {}
    seen: dict[int, int] = {}  # id -> the shallowest depth it was walked at
    state = side_walk if side_walk is not None else SideWalk()
    clean = state.clean

    def walk(o: Any, path: Path, depth: int) -> bool:
        """Record the tensors under ``o``; True iff it is proven tensor-free."""
        if isinstance(o, torch.Tensor):
            found[path] = o
            return False
        if o is None or isinstance(
            o, (str, bytes, int, float, bool, torch.nn.Module, type)
        ):
            return True
        if id(o) in clean:
            return True
        if depth > _MAX_DEPTH:
            if id(o) not in state.walked:
                state.truncated[id(o)] = o
            return False
        if seen.get(id(o), depth + 1) <= depth:
            return False  # walked on a path at most as deep: not proven here
        seen[id(o)] = depth  # first reached, or shallower than before: walk
        state.walked[id(o)] = o
        state.truncated.pop(id(o), None)
        tensor_free = True
        for key, child in _children(o):
            tensor_free = walk(child, path + (key,), depth + 1) and tensor_free
        if tensor_free:
            clean[id(o)] = o
        return tensor_free

    for registered in (layer._parameters, layer._buffers):
        for name, t in registered.items():
            if t is not None and not hasattr(t, "_ckpt_layout"):
                found[(("a", name),)] = t
    for name, value in vars(layer).items():
        if (
            name in _SKIP_LAYER_ATTRS
            or name.startswith(("_forward", "_direct"))  # _direct: reload records
            or "hooks" in name
        ):
            continue
        walk(value, (("a", name),), 1)
    return found


def _layout(t: torch.Tensor) -> tuple:
    return (tuple(t.shape), t.stride(), t.dtype, t.device)


def snapshot_side_tensors(
    layer: torch.nn.Module, side_walk: SideWalk | None = None
) -> dict[Path, torch.Tensor]:
    """The layer's derived device tensors, by path. Aliases, not the objects:
    `x.data = y` later leaves them on the storage graphs captured."""
    return {
        path: t.detach()
        for path, t in collect_side_tensors(layer, side_walk).items()
        if t.device.type not in ("cpu", "meta")
    }


@torch.no_grad()
def pin_side_tensors(
    layer: torch.nn.Module,
    before: dict[Path, torch.Tensor],
    side_walk: SideWalk | None = None,
) -> list[str]:
    """Put the tensor at every snapshotted path back on its startup storage,
    holding the rerun's values. Returns the paths that cannot be pinned: gone,
    or now a tensor of a different shape, stride, dtype or device. Pinning
    changes no object but tensors, so objects the pass proved tensor-free
    stay so."""
    after = collect_side_tensors(layer, side_walk)
    failed = []
    for path, old in before.items():
        t = after.get(path)
        if t is None or _layout(t) != _layout(old):
            failed.append(_fmt(path))
            continue
        if t.data_ptr() == old.data_ptr():
            continue  # the same view of the startup storage
        old.copy_(t)
        t.data = old  # every holder of `t`, seen by the walk or not
    return failed


def _fmt(path: Path) -> str:
    return "".join(f".{k}" if kind == "a" else f"[{k!r}]" for kind, k in path)
