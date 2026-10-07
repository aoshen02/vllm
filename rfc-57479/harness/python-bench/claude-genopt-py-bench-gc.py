"""r20 (claude-python-subagent): run scripts/claude-genopt-py-bench.py UNCHANGED (runpy),
with garbage-collector instrumentation per bench window (startup / consume / abort / build /
render): number and duration of collections per generation (gc.callbacks), gc counters and
thresholds, live tracked objects and loaded modules after imports and at the abort.

Extra options (removed before the bench parses its own):
  --gc-out PATH          where to write the GC record (JSON)
  --gc-off-build         gc.disable() for the build window only (control)
  --profile-build PATH   cProfile the build window only, pstats dump to PATH
"""

import gc
import json
import runpy
import sys
import time
from pathlib import Path

BENCH = Path(__file__).resolve().parent / "claude-genopt-py-bench.py"


def pop_opt(args, name, has_value=True):
    if name not in args:
        return None
    i = args.index(name)
    value = args[i + 1] if has_value else True
    del args[i : i + (2 if has_value else 1)]
    return value


args = sys.argv[1:]
gc_out = Path(pop_opt(args, "--gc-out"))
gc_off_build = bool(pop_opt(args, "--gc-off-build", has_value=False))
profile_build = pop_opt(args, "--profile-build")

record = {"threshold": gc.get_threshold(), "gc_off_build": gc_off_build,
          "python": sys.version.split()[0]}
window = {"name": "startup"}
acc: dict = {}
started = {}


def callback(phase, info):
    if phase == "start":
        started["t"] = time.perf_counter()
        return
    key = f"{window['name']}:gen{info['generation']}"
    entry = acc.setdefault(key, {"collections": 0, "seconds": 0.0, "collected": 0})
    entry["collections"] += 1
    entry["seconds"] += time.perf_counter() - started.get("t", time.perf_counter())
    entry["collected"] += info["collected"]


gc.callbacks.append(callback)

# The modules the bench imports (same objects; patched before the bench runs).
import vllm  # noqa: E402,F401
import vllm.entrypoints.scale_out.token_in_token_out.serving as serving_mod  # noqa: E402
import vllm.v1.engine.output_processor as op_mod  # noqa: E402

record["after_import"] = {"tracked_objects": len(gc.get_objects()),
                          "modules": len(sys.modules), "gc_count": gc.get_count(),
                          "vllm_file": vllm.__file__}

orig_process = op_mod.OutputProcessor.process_outputs
orig_abort = op_mod.OutputProcessor.abort_requests
orig_full = serving_mod.ServingTokens.serve_tokens_full_generator


def process_outputs(self, *a, **kw):
    if window["name"] == "startup":
        window["name"] = "consume"
    return orig_process(self, *a, **kw)


def abort_requests(self, *a, **kw):
    record["at_abort"] = {"tracked_objects": len(gc.get_objects()),
                          "gen2_objects": len(gc.get_objects(2)),
                          "modules": len(sys.modules), "gc_count": gc.get_count()}
    window["name"] = "abort"
    return orig_abort(self, *a, **kw)


async def serve_tokens_full_generator(self, *a, **kw):
    window["name"] = "build"
    record["build_start_gc_count"] = gc.get_count()
    profiler = None
    if profile_build:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()
    if gc_off_build:
        gc.disable()
    try:
        return await orig_full(self, *a, **kw)
    finally:
        if gc_off_build:
            gc.enable()
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(profile_build)
        window["name"] = "render"


op_mod.OutputProcessor.process_outputs = process_outputs
op_mod.OutputProcessor.abort_requests = abort_requests
serving_mod.ServingTokens.serve_tokens_full_generator = serve_tokens_full_generator

sys.argv = [str(BENCH), *args]
try:
    runpy.run_path(str(BENCH), run_name="__main__")
finally:
    gc.callbacks.remove(callback)
    record["gc"] = {k: {**v, "seconds": round(v["seconds"], 3)} for k, v in sorted(acc.items())}
    gc_out.write_text(json.dumps(record, indent=1) + "\n")
