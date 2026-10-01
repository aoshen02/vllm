"""Read-only actual backend/EP ownership admission for real server scheduling."""

import json
import os
from pathlib import Path

from fi_w4a16_clean_worker import CleanFIWorker


class BenchWorker(CleanFIWorker):
    def audit_bench(self):
        from vllm.distributed import get_ep_group, get_tp_group

        ep = get_ep_group()
        config = self.vllm_config
        assert ep.world_size == 4 and get_tp_group().world_size == 1
        assert not config.parallel_config.enable_eplb
        assert config.cache_config.cache_dtype == "fp8_e4m3"
        assert config.scheduler_config.max_num_batched_tokens == 16384
        expected = (
            "FlashInferCuteDSLW4A16Experts"
            if os.environ["FI_BENCH_COMPUTE"] == "fi"
            else "HummingIndexedExperts"
        )
        layers = {}
        for name, layer in self.model_runner.get_model().named_modules():
            if type(layer).__name__ != "RoutedExperts":
                continue
            method = layer.quant_method
            kernel = method.moe_kernel
            assert type(kernel.fused_experts).__name__ == expected
            assert method.use_a16
            assert layer.local_num_experts == 32 and layer.global_num_experts == 128
            assert type(kernel.prepare_finalize).__name__ == (
                "FlashInferNVLinkOneSidedPrepareAndFinalize"
            )
            mapping = layer.expert_map.cpu().tolist()
            owned = [index for index, local in enumerate(mapping) if local >= 0]
            assert owned == list(
                range(ep.rank_in_group * 32, (ep.rank_in_group + 1) * 32)
            )
            layers[name] = {"class": expected, "owned": owned, "map": mapping}
        assert len(layers) == 23
        record = {
            "pid": os.getpid(),
            "rank": ep.rank_in_group,
            "provenance": self.fi_worker_provenance,
            "layers": layers,
            "config": str(config),
            "prefix_cache": config.cache_config.enable_prefix_caching,
        }
        target = Path(os.environ["FI_BENCH_OUT"]) / f"rank{ep.rank_in_group}.json"
        with target.open("x") as output:
            json.dump(record, output, indent=2)
        return {"rank": ep.rank_in_group, "pid": os.getpid(), "layers": 23}
