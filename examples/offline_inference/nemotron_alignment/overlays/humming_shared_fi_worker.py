"""Isolated shared-FI candidate on the existing Nemotron BI execution path."""

import hashlib
import json
import os
from pathlib import Path

import torch

from fi_w4a16_shared_overlay import construction_overlay
from humming_stage4_bench_worker import HummingStage4BenchWorker
from vllm.v1.worker.gpu_worker import Worker


class SharedFISelection:
    def load_model(self, *, load_dummy_weights=False):
        if load_dummy_weights:
            raise ValueError("Shared-FI candidate requires checkpoint weights")
        with construction_overlay() as selected:
            super().load_model(load_dummy_weights=False)
        expected = 2 * int(os.environ["FI_EXPECTED_MOE"])
        if len(selected) != expected:
            raise ValueError(f"Selected {len(selected)} shared linears, expected {expected}")
        self.shared_fi_construction = selected

    def audit_shared_fi(self):
        from vllm.distributed import get_ep_group

        layers = {}
        for name, layer in self.model_runner.get_model().named_modules():
            marker = getattr(layer, "_fi_shared_candidate", None)
            if marker is None:
                continue
            if (
                ".shared_experts." not in name
                or not name.endswith((".up_proj", ".down_proj"))
                or type(layer.quant_method.kernel).__name__
                != "NemotronSharedHummingScaleFI"
                or not layer.is_w4a16_nvfp4
            ):
                raise ValueError(f"Unexpected shared-FI selection: {name}")
            layers[name] = {
                "kernel": type(layer.quant_method.kernel).__name__,
                "weight_shape": list(layer.weight.shape),
                "alpha_bits": layer.weight_global_scale.detach()
                .reshape(-1)
                .view(torch.int32)
                .cpu()
                .tolist(),
            }
        if len(layers) != len(self.shared_fi_construction):
            raise ValueError("Shared-FI layer inventory changed after load")
        return {
            "rank": get_ep_group().rank_in_group,
            "selected": layers,
            "construction": self.shared_fi_construction,
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }


class SharedFIProxyWorker(SharedFISelection, Worker):
    """Use only for fixed-token proxy admission, not timed measurements."""

    def load_model(self, *, load_dummy_weights=False):
        from humming_stage4_candidate import install

        super().load_model(load_dummy_weights=load_dummy_weights)
        self.humming_stage4 = install(
            self.model_runner, expected_moe_layers=int(os.environ["FI_EXPECTED_MOE"])
        )

    def audit_shared_fi(self):
        from humming_stage4_candidate import audit

        record = super().audit_shared_fi()
        record["humming_stage4"] = audit(
            self.model_runner,
            self.humming_stage4,
            expected_moe_layers=int(os.environ["FI_EXPECTED_MOE"]),
        )
        return record


class SharedFIBenchWorker(SharedFISelection, HummingStage4BenchWorker):
    """Compose shared-FI selection with the r4 routed-MoE tuning."""

    def audit_bench(self):
        record = super().audit_bench()
        record["shared_fi"] = self.audit_shared_fi()
        target = Path(os.environ["FI_BENCH_OUT"]) / f"rank{record['rank']}.json"
        target.write_text(json.dumps(record, indent=2))
        return record
