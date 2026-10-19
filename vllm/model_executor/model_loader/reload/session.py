# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One reload session per model; its strategy decides how weights are applied."""

from abc import ABC, abstractmethod
from weakref import WeakKeyDictionary

import torch

from vllm.config import ModelConfig

from .layerwise import finalize_layerwise_reload, initialize_layerwise_reload


class ReloadStrategy(ABC):
    @abstractmethod
    def start(self, model: torch.nn.Module) -> None: ...

    @abstractmethod
    def finish(self, model: torch.nn.Module, model_config: ModelConfig) -> None: ...


class LayerwiseReloadStrategy(ReloadStrategy):
    def start(self, model: torch.nn.Module) -> None:
        initialize_layerwise_reload(model)

    def finish(self, model: torch.nn.Module, model_config: ModelConfig) -> None:
        finalize_layerwise_reload(model, model_config)


# Strategy of each model's active session, latched at start.
_SESSIONS: WeakKeyDictionary[torch.nn.Module, ReloadStrategy] = WeakKeyDictionary()


def select_reload_strategy(model: torch.nn.Module) -> ReloadStrategy:
    return LayerwiseReloadStrategy()


def start_reload(model: torch.nn.Module) -> None:
    strategy = _SESSIONS.get(model) or select_reload_strategy(model)
    strategy.start(model)
    _SESSIONS[model] = strategy


def finish_reload(model: torch.nn.Module, model_config: ModelConfig) -> None:
    strategy = _SESSIONS.get(model) or select_reload_strategy(model)
    strategy.finish(model, model_config)
    _SESSIONS.pop(model, None)
