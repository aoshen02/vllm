"""Worker-local provenance/loader audit for the isolated pristine-package gate."""

import hashlib
import json
import os
from collections import Counter
from functools import wraps
from pathlib import Path

from vllm.v1.worker.gpu_worker import Worker


def arm_shutdown_trace(worker):
    import faulthandler
    import time

    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(2, repeat=False)
    marker = {"pid": os.getpid(), "time_ns": time.time_ns()}
    print(json.dumps({"FI_SHUTDOWN_ARMED": marker}), flush=True)
    return marker


class CleanFIWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        assert not load_dummy_weights
        import vllm
        import vllm.model_executor.layers.fused_moe.oracle.nvfp4 as oracle
        from vllm.model_executor.layers.fused_moe.experts import (
            flashinfer_cutedsl_w4a16_moe as adapter,
        )
        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
        from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM
        from vllm.model_executor.models.utils import AutoWeightsLoader

        root = Path(os.environ["FI_ISOLATED_PACKAGE_ROOT"]).resolve()
        manifest = json.loads(Path(os.environ["FI_PACKAGE_MANIFEST"]).read_text())
        assert Path(vllm.__file__).resolve().parent == root
        paths = {}
        for module in (oracle, adapter):
            path = Path(module.__file__).resolve()
            relative = str(path.relative_to(root))
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == manifest["isolated"][relative]
            paths[module.__name__] = {"path": str(path), "sha256": digest}
        assert adapter.FlashInferCuteDSLW4A16Experts.__module__ == adapter.__name__
        self.fi_worker_provenance = {"pid": os.getpid(), "sources": paths}
        if os.environ.get("FI_NATIVE_DLPACK_METADATA") == "1":
            from fi_w4a16_native_dlpack_overlay import install

            self.fi_worker_provenance["native_dlpack_metadata"] = install()
        print(json.dumps({"FI_WORKER_PRELOAD": self.fi_worker_provenance}), flush=True)
        counts = Counter()
        originals = []
        for cls in (DefaultModelLoader, NemotronHForCausalLM, AutoWeightsLoader):
            original = cls.load_weights
            originals.append((cls, original))

            def wrapper(self, *args, _original=original, _label=cls.__name__, **kwargs):
                counts[_label] += 1
                return _original(self, *args, **kwargs)

            cls.load_weights = wraps(original)(wrapper)
        try:
            super().load_model(load_dummy_weights=False)
            assert all(counts[cls.__name__] > 0 for cls, _ in originals)
            routed = [
                layer
                for layer in self.model_runner.get_model().modules()
                if type(layer).__name__ == "RoutedExperts"
            ]
            expected_moe = int(os.environ.get("FI_EXPECTED_MOE", "2"))
            assert len(routed) == expected_moe
            expected_compute = os.environ.get("FI_BENCH_COMPUTE", "fi")
            assert expected_compute in ("fi", "humming")
            expected_class = (
                "FlashInferCuteDSLW4A16Experts"
                if expected_compute == "fi"
                else "HummingIndexedExperts"
            )
            assert all(
                type(layer.quant_method.moe_kernel.fused_experts).__name__
                == expected_class
                for layer in routed
            )
            self.fi_loader_calls = dict(counts)
            print(
                json.dumps(
                    {
                        "FI_WORKER_LOADED": {
                            **self.fi_worker_provenance,
                            "loader_calls": self.fi_loader_calls,
                            "actual_expert_class": expected_class,
                        }
                    }
                ),
                flush=True,
            )
        finally:
            for cls, original in originals:
                cls.load_weights = original
