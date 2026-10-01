"""Isolated shared-FI worker with pre-capture Humming hybrid tuning."""

import hashlib
import json
import os
from pathlib import Path

import humming_hybrid_candidate as candidate
from fi_w4a16_bench_worker import BenchWorker
from humming_shared_fi_worker import SharedFISelection
from humming_stage4_bench_worker import parameter_identity
from vllm.distributed import get_ep_group
from vllm.v1.worker.gpu_worker import Worker


class HybridSelection(SharedFISelection):
    def load_model(self, *, load_dummy_weights=False):
        super().load_model(load_dummy_weights=load_dummy_weights)
        self.hybrid_parameter_identity = parameter_identity(
            self.model_runner.get_model()
        )
        self.humming_hybrid = candidate.install(
            self.model_runner, expected_moe_layers=int(os.environ["FI_EXPECTED_MOE"])
        )
        assert (
            parameter_identity(self.model_runner.get_model())
            == self.hybrid_parameter_identity
        )

    def audit_hybrid(self):
        assert (
            parameter_identity(self.model_runner.get_model())
            == self.hybrid_parameter_identity
        )
        expected = int(os.environ["FI_EXPECTED_MOE"])
        inventory = candidate.audit(self.model_runner, self.humming_hybrid, expected)
        assert inventory["recipe"] == candidate.RECIPE
        assert inventory["transform_sha256"] == candidate.source_sha(
            "humming_hybrid_contract.py"
        )
        assert all(not row["observed_selections"] for row in inventory["layers"])
        record = {
            "rank": get_ep_group().rank_in_group,
            "humming_hybrid": inventory,
            "shared_fi": self.audit_shared_fi(),
            "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parameter_identity_unchanged": True,
        }
        target = Path(os.environ["FI_BENCH_OUT"]) / f"hybrid-rank{record['rank']}.json"
        with target.open("x") as output:
            json.dump(record, output, indent=2)
        return {"rank": record["rank"], "layers": expected}


class HybridProxyWorker(HybridSelection, Worker):
    """Proxy numerical admission only; no throughput claims."""


class HybridBenchWorker(HybridSelection, BenchWorker):
    """Full-model performance screen after proxy numerical admission."""

    def audit_bench(self):
        record = super().audit_bench()
        record["hybrid"] = self.audit_hybrid()
        target = Path(os.environ["FI_BENCH_OUT"]) / f"rank{record['rank']}.json"
        target.write_text(json.dumps(record, indent=2))
        return record
