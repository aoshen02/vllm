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
"""

from typing import Any

import torch

__all__ = ["pin_side_tensors", "snapshot_side_tensors"]

Path = tuple[tuple[str, Any], ...]

_MAX_DEPTH = 6
_SKIP_MODULE_PREFIXES = (
    "vllm.config",
    "transformers",
    "tokenizers",
    "torch",
    "threading",
    "logging",
)
_SKIP_LAYER_ATTRS = {"_parameters", "_buffers", "_modules", "training"}


_OPAQUE: dict[type, bool] = {}


def _opaque(cls: type) -> bool:
    """Library objects that hold no derived device tensors (configs,
    tokenizers, locks, loggers), also when vLLM subclasses them: the cached
    tokenizer a multimodal model root keeps is a vLLM subclass of a
    transformers tokenizer, with ~250k BPE merges."""
    opaque = _OPAQUE.get(cls)
    if opaque is None:
        opaque = _OPAQUE[cls] = any(
            (c.__module__ or "").startswith(_SKIP_MODULE_PREFIXES) for c in cls.__mro__
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


Clean = dict[int, tuple[int, Any]]


def collect_side_tensors(
    layer: torch.nn.Module, clean: Clean | None = None
) -> dict[Path, torch.Tensor]:
    """The layer's derived tensors, by path: registered tensors without a
    checkpoint layout (the address check covers the checkpoint ones), and
    every tensor reachable from its non-module attributes.

    ``clean`` is shared by the calls of one pass over a model's modules and
    must not outlive it. It records objects proven tensor-free (by id, with
    the depth they were proven at, and the object to keep the id valid), so
    objects every layer references, e.g. a quantization config with a dict
    per checkpoint layer, are walked once per pass instead of once per layer.
    An object proven tensor-free at depth d holds no tensor within the depth
    limit at any depth >= d, so skipping it finds the same tensors."""
    found: dict[Path, torch.Tensor] = {}
    seen: set[int] = set()
    if clean is None:
        clean = {}

    def walk(o: Any, path: Path, depth: int) -> bool:
        """Record the tensors under ``o``; True iff it is proven tensor-free."""
        if isinstance(o, torch.Tensor):
            found[path] = o
            return False
        if (
            depth > _MAX_DEPTH
            or o is None
            or isinstance(o, (str, bytes, int, float, bool, torch.nn.Module, type))
        ):
            return True
        known = clean.get(id(o))
        if known is not None and known[0] <= depth:
            return True
        if id(o) in seen:
            return False  # walked on another path of this layer: not proven here
        seen.add(id(o))
        tensor_free = True
        for key, child in _children(o):
            tensor_free = walk(child, path + (key,), depth + 1) and tensor_free
        if tensor_free:
            clean[id(o)] = (depth, o)
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
    layer: torch.nn.Module, clean: Clean | None = None
) -> dict[Path, torch.Tensor]:
    """The layer's derived device tensors, by path. Aliases, not the objects:
    `x.data = y` later leaves them on the storage graphs captured.
    ``clean``: see `collect_side_tensors`."""
    return {
        path: t.detach()
        for path, t in collect_side_tensors(layer, clean).items()
        if t.device.type not in ("cpu", "meta")
    }


@torch.no_grad()
def pin_side_tensors(
    layer: torch.nn.Module,
    before: dict[Path, torch.Tensor],
    clean: Clean | None = None,
) -> list[str]:
    """Put the tensor at every snapshotted path back on its startup storage,
    holding the rerun's values. Returns the paths that cannot be pinned: gone,
    or now a tensor of a different shape, stride, dtype or device.
    ``clean``: see `collect_side_tensors`; pinning changes no object but
    tensors, so the objects it proved tensor-free stay so."""
    after = collect_side_tensors(layer, clean)
    failed = []
    for path, old in before.items():
        t = after.get(path)
        if t is not None and t.untyped_storage().data_ptr() == (
            old.untyped_storage().data_ptr()
        ):
            continue
        if t is None or _layout(t) != _layout(old):
            failed.append(_fmt(path))
            continue
        old.copy_(t)
        t.data = old  # every holder of `t`, seen by the walk or not
    return failed


def _fmt(path: Path) -> str:
    return "".join(f".{k}" if kind == "a" else f"[{k!r}]" for kind, k in path)
