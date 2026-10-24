# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep tensors held outside a layer's parameters and buffers at their
startup storage across a reload.

`process_weights_after_loading` often stores tensors on objects hanging off
the layer rather than registering them: `quant_method.moe_quant_config`,
`quant_method.moe_kernel`, experts objects, plain layer attributes.  A reload
reruns it on freshly materialized tensors, so those objects end up pointing
at the reload's temporaries (and at freshly computed derived tensors), while
CUDA graphs captured at startup keep reading the startup storage.

`pin_side_tensors` walks the same attribute paths before and after the rerun
and puts every tensor back on startup storage:
- a reference to one of the layer's processed temporaries (or a view of one)
  is redirected to the matching kernel tensor, which already holds the new
  values;
- a freshly computed tensor is copied into the startup tensor found at the
  same path (same shape, dtype and device) and that tensor is kept.
"""

from collections.abc import Callable
from typing import Any

import torch

__all__ = ["collect_side_tensors", "pin_side_tensors"]

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
        if name in _SKIP_LAYER_ATTRS or name.startswith("_forward") or "hooks" in name:
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


def _view_on(
    t: torch.Tensor, temp: torch.Tensor, kernel: torch.Tensor
) -> torch.Tensor | None:
    """`t` is a view of `temp`; return the same view of `kernel`."""
    if kernel.shape != temp.shape or kernel.stride() != temp.stride():
        return None
    base = kernel
    if t.dtype != base.dtype:
        if t.element_size() != base.element_size():
            return None
        base = base.view(t.dtype)
    offset = base.storage_offset() + t.storage_offset() - temp.storage_offset()
    return base.as_strided(t.size(), t.stride(), offset)


@torch.no_grad()
def pin_side_tensors(
    layer: torch.nn.Module,
    before: dict[Path, torch.Tensor],
    temps: dict[str, torch.Tensor],
    kernels: dict[str, torch.Tensor],
    log: Callable[[str], None] | None = None,
) -> None:
    """Redirect side references to startup storage (see module docstring).

    Args:
        layer: the reloaded layer, its kernel tensors already restored
        before: `collect_side_tensors(layer)` taken before the rerun
        temps: the layer's processed tensors by name, before the copy-back
        kernels: the startup kernel tensors by name
        log: called with a message for each reference that could not be pinned

    """
    by_id = {id(t): kernels[n] for n, t in temps.items() if n in kernels}
    by_storage = {
        t.untyped_storage().data_ptr(): (t, kernels[n])
        for n, t in temps.items()
        if n in kernels and t.device.type != "meta"
    }
    kernel_ids = {id(t) for t in kernels.values()}

    for path, t in collect_side_tensors(layer).items():
        repl: torch.Tensor | None = None
        if id(t) in by_id:
            repl = by_id[id(t)]
        elif t.device.type != "meta" and t.untyped_storage().data_ptr() in by_storage:
            temp, kernel = by_storage[t.untyped_storage().data_ptr()]
            repl = _view_on(t, temp, kernel)
        else:
            old = before.get(path)
            if (
                old is not None
                and old is not t
                and id(old) not in kernel_ids
                and old.shape == t.shape
                and old.dtype == t.dtype
                and old.device == t.device
                and old.data_ptr() != t.data_ptr()
            ):
                old.copy_(t)
                repl = old
        if repl is not None and repl is not t:
            _set_path(layer, path, repl)
        elif (
            repl is None
            and log is not None
            and path in before
            and before[path] is not t
        ):
            log(f"{type(layer).__name__}: cannot pin {_fmt(path)} to startup storage")


def _fmt(path: Path) -> str:
    return "".join(f".{k}" if kind == "a" else f"[{k!r}]" for kind, k in path)
