"""Stage4 performance composition without host selection or step observers."""

import hashlib
import json
import os
from pathlib import Path

import humming_stage4_candidate as candidate
from fi_w4a16_bench_worker import BenchWorker
from humming_stage4_bench_contract import validate_bench_inventory


def parameter_identity(model):
    return {
        name: (
            id(parameter),
            parameter.data_ptr(),
            tuple(parameter.shape),
            tuple(parameter.stride()),
        )
        for name, parameter in model.named_parameters()
    }


class HummingStage4BenchWorker(BenchWorker):
    def load_model(self, *, load_dummy_weights=False):
        if load_dummy_weights:
            raise ValueError("Require actual checkpoint weights")
        super().load_model(load_dummy_weights=load_dummy_weights)
        model = self.model_runner.get_model()
        self.stage4_parameter_identity = parameter_identity(model)
        self.humming_stage4 = candidate.install(self.model_runner, 23)
        assert parameter_identity(model) == self.stage4_parameter_identity

    def audit_bench(self):
        record = super().audit_bench()
        assert (
            parameter_identity(self.model_runner.get_model())
            == self.stage4_parameter_identity
        )
        actual = candidate.audit(self.model_runner, self.humming_stage4, 23)
        for item in self.model_runner._humming_stage4_state:
            experts = item["experts"]
            method = experts.prepare_humming_moe_kwargs
            assert getattr(method, "__self__", None) is experts
            assert (
                getattr(method, "__func__", None)
                is type(experts).prepare_humming_moe_kwargs
            )
            assert not item["row"]["observed_selections"]
        actual["worker_sha256"] = hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest()
        actual["parameter_identity_unchanged"] = True
        actual["host_selection_observer_installed"] = False
        validate_bench_inventory(actual, Path(__file__).parent)
        record["humming_stage4"] = actual
        target = Path(os.environ["FI_BENCH_OUT"]) / f"rank{record['rank']}.json"
        target.write_text(json.dumps(record, indent=2))
        return record
