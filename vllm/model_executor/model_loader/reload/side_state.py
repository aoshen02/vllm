# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Find the derived tensors a reload would move.

`process_weights_after_loading` derives tensors from the checkpoint and keeps
some outside the modules' parameters and buffers: on
`quant_method.moe_quant_config`, on kernel and experts objects, as plain layer
attributes. A reload reruns it while CUDA graphs captured at startup keep
reading the startup storage, so such a tensor must be refilled in place
(`replace_parameter`, `copy_`).

`side_tensors` finds each of them, by path, so a direct reload raises when
the rerun moved one. Not walked: runtime state not
derived from weights (KV and state caches, workspaces, communicators),
library and configuration objects, objects nested deeper than 16 attributes,
`functools.partial` arguments and closure cells.
"""

import functools
from collections import deque
from typing import Any

import regex as re
import torch

__all__ = ["side_tensors"]

_MAX_DEPTH = 16
# packages whose own classes hold no tensors derived from the weights
_LIBRARIES = {"transformers", "tokenizers", "torch", "threading", "logging"}
_SKIP_LAYER_ATTRS = {"_parameters", "_buffers", "_modules", "training"}
# runtime state a reload does not touch (attention, mamba and indexer caches)
_RUNTIME_ATTRS = re.compile(r"kv_cache|workspace")


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
    process groups and communicators (`vllm.distributed`), and subclasses of
    tokenizer, processor and configuration bases."""
    opaque = _OPAQUE.get(cls)
    if opaque is None:
        module = cls.__module__ or ""
        opaque = _OPAQUE[cls] = (
            module.partition(".")[0] in _LIBRARIES
            or module.startswith(("vllm.config", "vllm.distributed"))
            or not _tensor_free_bases().isdisjoint(cls.__mro__)
        )
    return opaque


def _children(o: Any) -> list[tuple[str, Any]]:
    if isinstance(o, (list, tuple)):
        return [(f"[{i}]", v) for i, v in enumerate(o)]
    if isinstance(o, dict):
        return [(f"[{k!r}]", v) for k, v in o.items()]
    if _opaque(type(o)):
        return []
    d = getattr(o, "__dict__", None)
    attrs = list(d.items()) if isinstance(d, dict) else []
    for k in getattr(type(o), "__slots__", ()):
        if isinstance(k, str) and hasattr(o, k):
            attrs.append((k, getattr(o, k)))
    return [(f".{k}", v) for k, v in attrs if not _RUNTIME_ATTRS.search(k)]


def side_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Every device tensor reachable from the modules' non-module attributes,
    by the shallowest path that reaches it, as an alias: holding it keeps its
    storage from being reused, so a moved tensor cannot look unmoved."""
    out: dict[str, torch.Tensor] = {}
    queue: deque[tuple[Any, str, int]] = deque()
    for name, module in model.named_modules():
        for attr, value in vars(module).items():
            if (
                attr in _SKIP_LAYER_ATTRS
                or attr.startswith(("_forward", "_direct"))  # _direct: reload's
                or "hooks" in attr
                or _RUNTIME_ATTRS.search(attr)
            ):
                continue
            queue.append((value, f"{name}.{attr}" if name else attr, 1))
    seen: set[int] = set()
    while queue:  # breadth first: each object once, at its shallowest depth
        o, path, depth = queue.popleft()
        if id(o) in seen or depth > _MAX_DEPTH:
            continue
        if isinstance(o, torch.Tensor):
            seen.add(id(o))
            if o.device.type not in ("cpu", "meta") and o.numel():
                out[path] = o.detach()
        elif not isinstance(o, (str, bytes, int, float, bool, torch.nn.Module, type)):
            seen.add(id(o))
            queue.extend((v, path + k, depth + 1) for k, v in _children(o))
    return out
