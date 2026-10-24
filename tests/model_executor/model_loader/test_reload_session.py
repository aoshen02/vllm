# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

import vllm.model_executor.model_loader.reload.session as session
from vllm.model_executor.model_loader.reload import (
    finish_reload,
    record_metadata_for_reloading,
    start_reload,
)


class RecordingStrategy(session.ReloadStrategy):
    def __init__(self, calls: list, name: str):
        self.calls, self.name, self.fail_finish = calls, name, False

    def start(self, model):
        self.calls.append((self.name, "start"))

    def finish(self, model, model_config):
        self.calls.append((self.name, "finish"))
        if self.fail_finish:
            self.fail_finish = False
            raise RuntimeError


@pytest.fixture
def calls(monkeypatch):
    calls: list = []
    names = iter("abc")
    monkeypatch.setattr(
        session,
        "select_reload_strategy",
        lambda model: RecordingStrategy(calls, next(names)),
    )
    return calls


def test_strategy_latched_per_model(calls):
    target, draft = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    start_reload(target)
    start_reload(draft)
    start_reload(target)
    finish_reload(target, None)
    finish_reload(draft, None)
    finish_reload(draft, None)
    assert calls == [
        ("a", "start"),
        ("b", "start"),
        ("a", "start"),
        ("a", "finish"),
        ("b", "finish"),
        ("c", "finish"),
    ]
    assert not session._SESSIONS


def test_failed_finish_keeps_session(calls):
    model = torch.nn.Linear(2, 2)
    start_reload(model)
    session._SESSIONS[model].fail_finish = True
    with pytest.raises(RuntimeError):
        finish_reload(model, None)
    finish_reload(model, None)
    assert calls == [("a", "start"), ("a", "finish"), ("a", "finish")]
    assert model not in session._SESSIONS


def test_layerwise_reload_keeps_storage():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4))
    record_metadata_for_reloading(model)
    weight, ptr = model[0].weight, model[0].weight.data_ptr()
    new = {k: torch.randn_like(v) for k, v in model.state_dict().items()}

    start_reload(model)
    for name, param in list(model.named_parameters()):
        param.weight_loader(param, new[name])
    finish_reload(model, None)

    assert model[0].weight is weight and weight.data_ptr() == ptr
    torch.testing.assert_close(model.state_dict(), new)
