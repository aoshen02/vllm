# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep tensors held outside a layer's parameters and buffers at their
startup storage across a reload.

`process_weights_after_loading` often stores tensors on objects hanging off
the layer rather than registering them: `quant_method.moe_quant_config`,
`quant_method.moe_kernel`, experts objects, plain layer attributes. A reload
reruns it, so those objects end up pointing at freshly allocated tensors,
while CUDA graphs captured at startup keep reading the startup storage.

One rule: before the rerun, `snapshot_side_tensors` records every device
tensor reachable from the layer's attributes, by attribute path, as an alias
of its storage; after the rerun, `pin_side_tensors` copies the tensor found at
the same path into that storage and puts the alias back at the path.
"""

from typing import Any

import torch

__all__ = ["pin_side_tensors", "snapshot_side_tensors"]

Path = tuple[tuple[str, Any], ...]

_MAX_DEPTH = 6
_SKIP_MODULE_PREFIXES = ("vllm.config", "transformers", "torch", "threading", "logging")
_SKIP_LAYER_ATTRS = {"_parameters", "_buffers", "_modules", "training"}


def _children(o: Any) -> list[tuple[tuple[str, Any], Any]]:
    if isinstance(o, (list, tuple)):
        return [(("i", i), v) for i, v in enumerate(o)]
    if isinstance(o, dict):
        return [(("k", k), v) for k, v in o.items()]
    if (type(o).__module__ or "").startswith(_SKIP_MODULE_PREFIXES):
        return []
    d = getattr(o, "__dict__", None)
    out = [(("a", k), v) for k, v in d.items()] if isinstance(d, dict) else []
    for k in getattr(type(o), "__slots__", ()):
        if isinstance(k, str) and hasattr(o, k):
            out.append((("a", k), getattr(o, k)))
    return out


def collect_side_tensors(layer: torch.nn.Module) -> dict[Path, torch.Tensor]:
    """Every tensor reachable from the layer's non-module attributes, by path."""
    found: dict[Path, torch.Tensor] = {}
    seen: set[int] = set()

    def walk(o: Any, path: Path, depth: int) -> None:
        if isinstance(o, torch.Tensor):
            found[path] = o
            return
        if (
            depth > _MAX_DEPTH
            or o is None
            or isinstance(o, (str, bytes, int, float, bool, torch.nn.Module, type))
            or id(o) in seen
        ):
            return
        seen.add(id(o))
        for key, child in _children(o):
            walk(child, path + (key,), depth + 1)

    for name, value in vars(layer).items():
        if (
            name in _SKIP_LAYER_ATTRS
            or name.startswith(("_forward", "_direct"))  # _direct: reload records
            or "hooks" in name
        ):
            continue
        walk(value, (("a", name),), 1)
    return found


def _get(o: Any, key: tuple[str, Any]) -> Any:
    kind, k = key
    return getattr(o, k) if kind == "a" else o[k]


def _with(o: Any, key: tuple[str, Any], value: Any) -> Any:
    """Set `key` on `o`; returns the container to keep (a new one for tuples)."""
    kind, k = key
    if kind == "a":
        object.__setattr__(o, k, value)
        return o
    if isinstance(o, tuple):
        items = list(o)
        items[k] = value
        return type(o)(*items) if hasattr(o, "_fields") else type(o)(items)
    o[k] = value
    return o


def _set_path(root: Any, path: Path, value: Any) -> Any:
    if len(path) == 1:
        return _with(root, path[0], value)
    child = _get(root, path[0])
    new_child = _set_path(child, path[1:], value)
    return root if new_child is child else _with(root, path[0], new_child)


def _layout(t: torch.Tensor) -> tuple:
    return (tuple(t.shape), t.stride(), t.dtype, t.device)


def snapshot_side_tensors(layer: torch.nn.Module) -> dict[Path, torch.Tensor]:
    """Device tensors reachable from the layer's non-module attributes, by
    path. Aliases, not the objects: `x.data = y` later leaves them on the
    storage graphs captured."""
    return {
        path: t.detach()
        for path, t in collect_side_tensors(layer).items()
        if t.device.type not in ("cpu", "meta")
    }


@torch.no_grad()
def pin_side_tensors(
    layer: torch.nn.Module, before: dict[Path, torch.Tensor]
) -> list[str]:
    """Put every snapshotted path back on its startup storage, holding the
    rerun's values. Returns the paths that cannot be pinned: gone, or now a
    tensor of a different shape, stride, dtype or device."""
    after = collect_side_tensors(layer)
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
        _set_path(layer, path, old)
    return failed


def _fmt(path: Path) -> str:
    return "".join(f".{k}" if kind == "a" else f"[{k!r}]" for kind, k in path)
