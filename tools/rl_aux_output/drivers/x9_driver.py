# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""X5-long: continuous multi-turn agent sessions whose contexts grow to 128K and 512K
tokens.

Same engine settings as x2_driver.py / x4_driver.py (BI env from the launcher,
FLASHMLA_SPARSE_DSV41, nvfp4_ds_mla KV, TP4 + EP, moe_backend OC_MOE=auto),
max_model_len raised to the longest phase.
Loop per phase (x4 style): keep up to C sessions in flight (C from the pool bytes), run
one turn of every in-flight session per engine call, and when a session has done its
turns the "trainer" reads every request of it back, checks it, then deletes the
session's keys (release).

Arms (OC_ARM): off   no Mooncake, R3 inline only (step-time baseline)
               r3    R3 delivered through Mooncake (OC_EXTRA=0)
               full  R3 + indexer top-k + hidden (OC_EXTRA=1)
Checks: R3 delivered == inline (missing / wrong rows); indexer rows present for every
position, no missing objects, values in range; hidden chunks present for every sampled
token and lm_head(dequantized) logprob vs vLLM's logprob.
Metrics: bytes/token per field, objects/s, put latency p50/p99 (writer histogram), step
time per turn, host memory peak (node MemTotal - MemAvailable, 1 s samples).
Stall guard: an engine call longer than OC_GEN_TIMEOUT dumps py-spy stacks of this
process tree and exits with a DEADLOCK summary.
"""

import json
import os
import pathlib
import random
import subprocess
import sys
import threading
import time

import numpy as np

OUT = os.environ["OUT"]
M = os.environ["MODEL"]
ARM = os.environ.get("OC_ARM", "full")
DESIGN = os.environ.get("OC_DESIGN", "ps")
EXTRA = os.environ.get("OC_EXTRA", "0") == "1"
POOL = int(os.environ.get("OC_POOL_BYTES", str(96 << 30)))
LENGTHS = [int(x) for x in os.environ.get("OC_LENGTHS", "131072,524288").split(",")]
SESS = [int(x) for x in os.environ.get("OC_SESS", "8,4").split(",")]
TURNS = int(os.environ.get("OC_T", "4"))
MAXTOK = int(os.environ.get("OC_MAXTOK", "128"))
C_MULT = float(os.environ.get("OC_C_MULT", "1"))
IDX_BPT_EST = float(
    os.environ.get("OC_IDX_BPT_EST", "10240")
)  # 20-bit packed, before zstd
READ_LAST_ONLY = os.environ.get("OC_READ_LAST_ONLY") == "1"
GEN_TIMEOUT = float(os.environ.get("OC_GEN_TIMEOUT", "1800"))
RUN_DIR = os.path.dirname(OUT)
log = open(OUT + ".log", "a")  # noqa: SIM115


def say(*a):
    msg = " ".join(str(x) for x in a)
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()


# ---------------- host memory sampler ----------------
MEM = dict(peak_used_gb=0.0, base_used_gb=None, phase_peak={})
PHASE = ["load"]


def meminfo():
    d = {}
    for line in pathlib.Path("/proc/meminfo").read_text().splitlines(keepends=True):
        k, v = line.split(":", 1)
        d[k] = int(v.split()[0]) * 1024
    return d


def mem_loop():
    while True:
        d = meminfo()
        used = (d["MemTotal"] - d["MemAvailable"]) / 2**30
        if MEM["base_used_gb"] is None:
            MEM["base_used_gb"] = round(used, 1)
            MEM["total_gb"] = round(d["MemTotal"] / 2**30, 1)
        MEM["peak_used_gb"] = max(MEM["peak_used_gb"], round(used, 1))
        p = PHASE[0]
        MEM["phase_peak"][p] = max(MEM["phase_peak"].get(p, 0.0), round(used, 1))
        time.sleep(1)


threading.Thread(target=mem_loop, daemon=True).start()


# ---------------- stall guard ----------------
def dump_stacks(tag):
    d = os.path.join(RUN_DIR, f"stall-{ARM}-{int(time.time())}")
    os.makedirs(d, exist_ok=True)
    pids = subprocess.run(
        [
            "bash",
            "-c",
            f"pgrep -g $(ps -o pgid= {os.getpid()}) || pgrep -P {os.getpid()}",
        ],
        capture_output=True,
        text=True,
    ).stdout.split()
    for pid in [str(os.getpid())] + [p for p in pids if p != str(os.getpid())]:
        r = subprocess.run(
            ["py-spy", "dump", "--native", "--pid", pid], capture_output=True, text=True
        )
        pathlib.Path(os.path.join(d, f"pyspy-{pid}.txt")).write_text(
            r.stdout + r.stderr
        )
    for cmd, name in [
        ("nvidia-smi", "nvidia-smi.txt"),
        ("free -g; cat /proc/pressure/memory", "mem.txt"),
        ("ps -eo pid,ppid,pgid,stat,rss,etime,cmd --sort=-rss | head -40", "ps.txt"),
    ]:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
        pathlib.Path(os.path.join(d, name)).write_text(r.stdout + r.stderr)
    say("STALL_DUMP", tag, d)
    return d


class Guard:
    def __init__(self):
        self.t = None
        self.tag = ""
        threading.Thread(target=self._loop, daemon=True).start()

    def arm(self, tag):
        self.tag, self.t = tag, time.time()

    def disarm(self):
        self.t = None

    def _loop(self):
        while True:
            time.sleep(10)
            if self.t is not None and time.time() - self.t > GEN_TIMEOUT:
                d = dump_stacks(self.tag)
                STATS["deadlock"] = dict(
                    where=self.tag,
                    after_s=round(time.time() - self.t),
                    dump=d,
                    writer=last_writer_stats(),
                )
                finish(deadlock=True)
                os._exit(9)


# ---------------- Mooncake ----------------
procs = []
REP = os.environ.get(
    "OC_REPLICA", ""
)  # replica tag: session names and the end-of-job sweep stay per replica
if ARM != "off" and os.environ.get("OC_EXT_MASTER"):
    os.environ["OC_MC_MASTER"] = os.environ[
        "OC_EXT_MASTER"
    ]  # shared pool on another node
elif ARM != "off":
    port = int(os.environ.get("OC_MC_PORT", "50051"))
    margs = ["mooncake_master", f"--port={port}", "--default_kv_lease_ttl=1000"]
    margs += os.environ.get("OC_MASTER_ARGS", "").split()
    procs.append(
        subprocess.Popen(
            margs,
            stdout=open(OUT + ".master.log", "w"),  # noqa: SIM115
            stderr=subprocess.STDOUT,
        )
    )
    time.sleep(3)
    os.environ["OC_MC_MASTER"] = f"127.0.0.1:{port}"
    procs.append(
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import os,time,sys; sys.path.insert(0, os.environ['OC_DIR']); "
                "import oc_common as c;"
                f" s=c.make_store(segment_bytes={POOL}, buffer_bytes=1<<28); "
                "print('holder up', flush=True); time.sleep(10**7)",
            ],
            stdout=open(OUT + ".holder.log", "w"),  # noqa: SIM115
            stderr=subprocess.STDOUT,
        )
    )
    time.sleep(8)
os.environ["OC_STATS_FILE"] = OUT + ".writer"
sys.path.insert(0, os.environ["OC_DIR"])
import oc_common as oc  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

if int(os.environ.get("OC_INJECT_ERR_EVERY", "0") or 0):
    # Before the engine starts: its worker processes inherit the log the checkers read.
    os.environ.setdefault("OC_INJECT_LOG", os.environ.get("OUT", "oc") + ".inject")
from vllm.inputs import TokensPrompt  # noqa: E402

NOR3 = os.environ.get("OC_NOR3") == "1"
MAXLEN = max(LENGTHS) + MAXTOK + 1024
t0 = time.time()
llm = LLM(
    model=M,
    tensor_parallel_size=4,
    enable_expert_parallel=True,
    enable_return_routed_experts=not NOR3,
    max_model_len=MAXLEN,
    gpu_memory_utilization=float(os.environ.get("OC_GPU_UTIL", "0.65")),
    trust_remote_code=True,
    attention_backend="FLASHMLA_SPARSE_DSV41",
    hf_overrides={"head_dtype": "float32"},
    compilation_config=(
        {"cudagraph_mode": "FULL"}
        if os.environ.get("OC_COMP") == "graph"
        else {"mode": 0}
    ),
    attention_config={"indexer_kv_dtype": "mxfp4", "indexer_sparse_logits": False},
    kv_cache_dtype="nvfp4_ds_mla",
    all2all_backend="allgather_reducescatter",
    kernel_config={
        "enable_flashinfer_autotune": True,
        "moe_backend": os.environ.get("OC_MOE", "auto"),
    },
    engram_config={
        "cpu_offload": True,
        "use_thp": True,
        "embedding_across_dp": True,
        "dp_shared_memory": False,
    },
    language_model_only=True,
    max_num_seqs=64,
    max_num_batched_tokens=8192,
    enable_prefix_caching=True,
    **(
        dict(enable_sleep_mode=True)
        if os.environ.get("OC_SLEEP_EACH_STEP") == "1"
        else {}
    ),
    **(dict(disable_log_stats=False) if os.environ.get("OC_OFFLOAD_GB") else {}),
    **(
        dict(
            kv_transfer_config={
                "kv_connector": "OffloadingConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": {
                    "block_size": int(os.environ.get("OC_OFFLOAD_BLOCK", "128")),
                    "cpu_bytes_to_use": int(
                        float(os.environ["OC_OFFLOAD_GB"]) * (1 << 30)
                    ),
                },
            }
        )
        if os.environ.get("OC_OFFLOAD_GB")
        else {}
    ),
)
STATS = dict(
    arm=ARM,
    design=DESIGN,
    extra=EXTRA,
    pool_bytes=POOL,
    c_mult=C_MULT,
    max_model_len=MAXLEN,
    load_s=round(time.time() - t0, 1),
    phases=[],
)
say("load_s", STATS["load_s"], "arm", ARM, "extra", EXTRA, "max_model_len", MAXLEN)
tok = llm.get_tokenizer()
reader = oc.make_store(segment_bytes=0, buffer_bytes=2 << 30) if ARM != "off" else None
guard = Guard()

W32 = None
if EXTRA:
    from safetensors import safe_open

    widx = json.loads(pathlib.Path(f"{M}/model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    with safe_open(f"{M}/{widx['head.weight']}", "pt") as f:
        W32 = f.get_tensor("head.weight").to("cuda:0").float()

_writer_snap = [None]


def last_writer_stats():
    """Newest line of the worker's periodic writer stats file(s)."""
    best = None
    for fn in os.listdir(RUN_DIR):
        if fn.startswith(os.path.basename(OUT) + ".writer."):
            try:
                lines = (
                    pathlib.Path(os.path.join(RUN_DIR, fn))
                    .read_text()
                    .strip()
                    .splitlines()
                )
                st = json.loads(lines[-1]) if lines else None
            except Exception:
                st = None
            if st and (best is None or st.get("puts", 0) > best.get("puts", 0)):
                best = st
    return best or _writer_snap[0]


WORDS = [
    "agent",
    "repository",
    "tests",
    "failing",
    "assertion",
    "parser",
    "patch",
    "diff",
    "compile",
    "run",
    "trace",
    "stack",
    "frame",
    "module",
    "import",
    "config",
    "cache",
    "request",
    "worker",
    "shell",
    "output",
    "error",
    "retry",
    "build",
    "lint",
    "coverage",
]


_CORPUS = [None]


def _corpus_ids():
    """Real text for prompts and tool outputs: vLLM's own source tree, tokenized once
    and cached."""
    if _CORPUS[0] is None:
        import vllm as _v

        cache = os.path.join(
            os.environ.get("OC_CORPUS_DIR", "/tmp"), "oc_corpus_ids.npy"
        )
        if os.path.exists(cache):
            _CORPUS[0] = np.load(cache)
        else:
            root = os.path.dirname(_v.__file__)
            texts, size = [], 0
            for d, _, fs in sorted(os.walk(root)):
                for f in sorted(fs):
                    if (
                        f.endswith((".py", ".md", ".txt", ".json", ".yaml"))
                        and size < 24 << 20
                    ):
                        try:
                            t = pathlib.Path(os.path.join(d, f)).read_text(
                                encoding="utf-8", errors="replace"
                            )
                        except OSError:
                            continue
                        texts.append(t)
                        size += len(t)
            ids = []
            for i in range(0, len(texts), 64):
                ids += tok.encode(
                    "\n".join(texts[i : i + 64]), add_special_tokens=False
                )
            _CORPUS[0] = np.asarray(ids, dtype=np.int32)
            np.save(cache, _CORPUS[0])
        say("corpus_tokens", len(_CORPUS[0]))
    return _CORPUS[0]


def text_ids(seed, n):
    """~n tokens deterministic in seed: real source text (OC_TEXT=real) or random
    words."""
    if os.environ.get("OC_TEXT") == "real":
        c = _corpus_ids()
        off = random.Random(seed).randrange(0, max(1, len(c) - n))
        out = c[off : off + n].tolist()
        while len(out) < n:
            out += c[: n - len(out)].tolist()
        return out
    rng = random.Random(seed)
    ids = []
    while len(ids) < n:
        chunk = " ".join(rng.choice(WORDS) for _ in range(4096))
        ids += tok.encode(chunk, add_special_tokens=False)
    return ids[:n]


GET_CHUNK = int(os.environ.get("OC_GET_CHUNK", "32"))
READ_STATS = dict(batch_calls=0, single_retries=0, read_fail_keys=0)


def get_many(keys):
    """Reader: get_batch in chunks of GET_CHUNK keys (one batch of a whole 512K request,
    ~4K objects / 300 MB over the TCP transport, outlives the 1 s read lease and fails
    with TRANSFER_FAIL), then single-key retries for any key that came back empty."""
    out = []
    gb = getattr(reader, "get_batch", None)
    for i in range(0, len(keys), GET_CHUNK):
        part = keys[i : i + GET_CHUNK]
        vals = None
        if gb is not None and len(part) > 1:
            try:
                READ_STATS["batch_calls"] += 1
                vals = [bytes(v) if v else None for v in gb(part)]
            except Exception:
                vals = None
        if vals is None:
            vals = [None] * len(part)
        for j, k in enumerate(part):
            tries = 0
            while vals[j] is None and tries < 3:
                tries += 1
                READ_STATS["single_retries"] += 1
                v = reader.get(k)
                vals[j] = bytes(v) if v else None
            if vals[j] is None:
                READ_STATS["read_fail_keys"] += 1
        out += vals
    return out


def get_manifest(session, rid, timeout=300):
    end = time.time() + timeout
    while time.time() < end:
        raw = reader.get(oc.manifest_key(session, rid))
        err = reader.get(oc.err_key(oc.manifest_key(session, rid)))
        if raw:
            # A manifest decides by its listed objects; a marker beside it is stale.
            if err:
                READ_STATS["stale_marker"] = READ_STATS.get("stale_marker", 0) + 1
            return json.loads(bytes(raw))
        if err:
            return None
        time.sleep(0.05)
    return None


FAILED_REQ: set = set()
INJECT_EVERY = int(os.environ.get("OC_INJECT_ERR_EVERY", "0") or 0)


def absent_listed(m):
    """Every object the manifest lists that the store does not hold (one batched call
    if it can)."""
    keys = sorted(oc.manifest_object_keys(m))
    batch = getattr(reader, "batch_is_exist", None)
    got = batch(keys) if batch is not None else [reader.is_exist(k) for k in keys]
    return [k for k, v in zip(keys, got) if not (v is True or v == 1)]


def request_failed(ph, s, rid, why, keys=None):
    """The request failed as a whole: its error marker is up, or an object its manifest
    lists is absent. Counted once per request in failed_requests, or in
    injected_failed_requests when all its failed objects (absent, or named by its
    marker) are keys whose write was injected to fail. Rows of a failed request are not
    counted as missing."""
    if (s["name"], rid) in FAILED_REQ:
        return
    FAILED_REQ.add((s["name"], rid))
    if keys is None:
        mk = reader.get(oc.err_key(oc.manifest_key(s["name"], rid)))
        keys = oc.marker_failed_keys(mk) if mk else []
    injected = INJECT_EVERY > 0 and bool(keys) and set(keys) <= oc.injected_keys()
    k = "injected_failed_requests" if injected else "failed_requests"
    ph[k] = ph.get(k, 0) + 1
    if len(ph.setdefault("failed_detail", [])) < 20:
        ph["failed_detail"].append(
            dict(session=s["name"], rid=rid, why=why, injected=injected)
        )


def check_request(s, rec, ph, full=True):
    """Read one request back; update the phase counters."""
    rid, need, inline, gen_ids, lps, P = rec[:6]
    m = get_manifest(s["name"], rid)
    if m is not None and (WCHANGE_AFTER or WCHANGE_EVERY) and len(rec) >= 8:
        # Requests up to the change step: one segment at version 0. The first request
        # after it: version 0 up to its aux start, version 1 from there.
        aux, step = rec[6], rec[7]
        wv = m.get("weight_versions")
        want = [[0, 0]] if step <= WCHANGE_AFTER else None
        if step == WCHANGE_AFTER + 1:
            want = [[0, 0], [aux, 1]]
        ph.setdefault("wv_checked", 0)
        ph.setdefault("wv_bad", [])
        if WCHANGE_EVERY:
            # A weight update after every step: request k delivers rows of versions
            # 0..k-1, the last segment starting at its own aux start.
            want = None
            # Request k of a session admitted at weight version v0 delivers versions
            # v0..v0+k-1.
            k = next(n for n, r in enumerate(s["reqs"]) if r is rec) + 1
            v0 = s.get("v0", 0)
            ok = (
                isinstance(wv, list)
                and len(wv) == k
                and [x[1] for x in wv] == list(range(v0, v0 + k))
                and wv[0][0] == 0
                and (k == 1 or wv[-1][0] == aux)
            )
            ph["wv_checked"] += 1
            if not ok and len(ph["wv_bad"]) < 10:
                ph["wv_bad"].append(dict(session=s["name"], step=step, aux=aux, got=wv))
        if want is not None:
            ph["wv_checked"] += 1
            if wv != want and len(ph["wv_bad"]) < 10:
                ph["wv_bad"].append(
                    dict(session=s["name"], step=step, aux=aux, got=wv, want=want)
                )
    if m is None and reader.get(oc.err_key(oc.manifest_key(s["name"], rid))):
        request_failed(ph, s, rid, "error marker")
        return
    if m is None:
        ph["missing_manifest"] += 1
        ph["r3_missing_rows"] += need
        return
    if not full:
        # Manifest-only check: the session's last request covers every row, so its full
        # read checks all positions; earlier requests are checked for their manifest,
        # row count and versions.
        ph["manifest_only_checked"] = ph.get("manifest_only_checked", 0) + 1
        if int(m.get("rows", 0)) < need:
            ph["r3_missing_rows"] += need - int(m.get("rows", 0))
        return
    if m.get("writer_stats"):
        _writer_snap[0] = m["writer_stats"]
    absent = absent_listed(m)
    if absent:
        request_failed(ph, s, rid, f"listed object absent: {absent[0]}", absent)
        return
    parts = get_many([oc.block_key(s["name"], h) for h in m["blocks"]])
    tail_raw = get_many([oc.tail_key(s["name"], rid)])[0]
    if any(p is None for p in parts) or (m["tail_nbytes"] and tail_raw is None):
        request_failed(ph, s, rid, "absent r3 object")
        return
    miss = 0
    tail = (tail_raw or b"")[: m["tail_nbytes"]]
    if miss:
        ph["r3_missing_rows"] += need
    else:
        got = np.frombuffer(b"".join(parts) + tail, dtype=m["dtype"]).reshape(
            (-1, *m["shape"])
        )[: m["rows"]]
        if len(got) < need:
            ph["r3_missing_rows"] += need - len(got)
        n = min(len(got), need)
        if "r3_cone" in m:
            vals = get_many([p["key"] for p in m["r3_cone"]["patches"] if "key" in p])
            vals_i = get_many(
                [p["idx_key"] for p in m["r3_cone"]["patches"] if "idx_key" in p]
            )
            if any(v is None for v in vals + vals_i):
                request_failed(ph, s, rid, "absent cone patch")
                return
            ph["cone_bytes"] = ph.get("cone_bytes", 0) + sum(len(v) for v in vals if v)
            ph["cone_idx_bytes"] = ph.get("cone_idx_bytes", 0) + sum(
                len(v) for v in vals_i if v
            )
            got, inline = (
                got[:, : oc.R3_FIRST_TRIMMED],
                inline[:, : oc.R3_FIRST_TRIMMED],
            )
        bad = np.any(got[:n].reshape(n, -1) != inline[:n].reshape(n, -1), axis=1)
        ph["r3_wrong_rows"] += int(bad.sum())
        ph["r3_rows_checked"] += n
        if bad.any() and len(ph["wrong_detail"]) < 20:
            idx = np.nonzero(bad)[0]
            ph["wrong_detail"].append(
                dict(
                    session=s["name"],
                    rid=rid,
                    need=need,
                    first=int(idx[0]),
                    last=int(idx[-1]),
                    n=int(len(idx)),
                )
            )
    if not EXTRA:
        return
    fields = {f["name"]: f for f in m.get("fields", [])}
    fi = fields.get("indexer_topk")
    if fi is None:
        ph["idx_manifest_missing"] += 1
    else:
        keys = [oc.block_key(s["name"], h) + fi["block_suffix"] for h in m["blocks"]]
        if fi["tail_rows"] > 0:
            keys.append(fi["tail_key"])
        vals = get_many(keys)
        if any(not b for b in vals):
            request_failed(ph, s, rid, "absent indexer object")
            return
        rows, miss = [], 0
        for k, b in zip(keys, vals):
            if k == fi.get("tail_key") and len(b) <= 1:
                miss += 1
                continue
            rows.append(b)
        ph["idx_missing_objects"] += miss + fi.get("missing", 0)
        ph["idx_holes"] += fi.get("holes", 0)
        t_dec = time.time()
        if (
            rows
            and not miss
            and need > int(os.environ.get("OC_IDX_DECODE_GPU_MAX", "100000"))
        ):
            r = np.concatenate([oc.idx_decode(b_, fi["shape"]) for b_ in rows])
        else:
            r = oc.idx_decode_gpu(rows, fi["shape"]) if rows and not miss else None
        ph["idx_decode_s"] = ph.get("idx_decode_s", 0.0) + time.time() - t_dec
        if r is None or len(r) < need:
            ph["idx_missing_rows"] += need - (0 if r is None else len(r))
        else:
            r = r[:need]
            # Valid entries are -1 (pad) or an index below the position's compressed
            # length.
            ph["idx_out_of_range"] += int(((r < -1) | (r >= MAXLEN)).sum())
            ph["idx_rows_checked"] += need
    fh = fields.get("hidden")
    G = len(gen_ids)
    if fh is None:
        ph["hid_manifest_missing"] += 1
        return
    vals = get_many([oc.hid_key(s["name"], rid, i) for i in range(fh["chunks"])])
    if any(v is None for v in vals):
        request_failed(ph, s, rid, "absent hidden chunk")
        return
    miss = 0
    if miss or fh["start"] != P - 1 or fh["rows"] < G:
        ph["hid_bad"] += 1
        return
    if oc.hid_check(fh, vals) is not None:
        ph["hid_bad"] += 1
        return
    import torch

    # Raw rows are the lm_head input itself.
    deq = oc.hid_torch(fh, vals, G).float()
    lsm = torch.log_softmax(deq @ W32.T, -1)
    lp = lsm.gather(1, torch.tensor(gen_ids, device="cuda")[:, None]).squeeze(1)
    if len(lps) == G:
        e = (lp - torch.tensor(lps, device="cuda")).abs()
        ph["hid_lp_err"].append(float(e.mean()))
        ph["hid_lp_err_max"] = max(ph["hid_lp_err_max"], float(e.max()))
    ph["hid_rows_checked"] += G


def finish(deadlock=False):
    STATS["host_mem"] = MEM
    STATS["writer_final"] = last_writer_stats()
    STATS["deadlock_detected"] = deadlock
    STATS["reader"] = READ_STATS
    if reader is not None and not deadlock:
        STATS["leaked_keys"] = reader.remove_by_regex("^" + oc.PREFIX + "/" + REP, True)
    say("SUMMARY", json.dumps(STATS, default=str))
    pathlib.Path(OUT).write_text(json.dumps(STATS, indent=1, default=str))
    for p in procs:
        p.terminate()


# ---------------- warmup (untimed) ----------------
PHASE[0] = "warmup"
guard.arm("warmup")
llm.generate(
    [TokensPrompt(prompt_token_ids=text_ids(900 + i, 2000)) for i in range(4)],
    [
        SamplingParams(max_tokens=16, extra_args={"oc_session": f"{DESIGN}-warm"})
        for _ in range(4)
    ],
    use_tqdm=False,
)
guard.disarm()
say("warmup done")


def tick():
    if os.environ.get("OC_NO_TICK") == "1":  # engine-native finish: no extra requests
        return
    for _ in range(2):
        llm.generate(
            [TokensPrompt(prompt_token_ids=[1, 2, 3])],
            SamplingParams(max_tokens=1),
            use_tqdm=False,
        )


WCHANGE_AFTER = int(os.environ.get("OC_WEIGHT_CHANGE_AFTER_STEP", "0"))
WCHANGE_SCALE = float(os.environ.get("OC_WEIGHT_CHANGE_SCALE", "0.05"))


def _perturb_router(self, scale, seed):
    """Worker side: add seeded noise (same on every rank) to every MoE router weight
    in place."""
    import torch

    runner = self.model_runner
    model = getattr(runner, "model", None) or runner.get_model()
    import os as _os

    full = _os.environ.get("OC_PERTURB") == "full"
    small = float(_os.environ.get("OC_PERTURB_SMALL", "0.01"))
    n = 0
    counts = {"router": 0, "norm": 0, "indexer": 0}
    for name, p in model.named_parameters():
        if not (p.dtype.is_floating_point and p.element_size() >= 2):
            continue
        if name.endswith(".gate.weight") and p.dim() == 2 and p.shape[0] >= 64:
            kind, sc = "router", scale
        elif full and name.endswith("norm.weight"):
            kind, sc = (
                "norm",
                small,
            )  # every layer's norms: KV, indexer and hidden all change
        elif full and "indexer" in name:
            kind, sc = "indexer", small
        else:
            continue
        g = torch.Generator().manual_seed(seed + n)
        noise = torch.randn(p.shape, generator=g, dtype=torch.float32)
        ref = (
            p.data.float().abs().mean().cpu()
            if kind == "norm"
            else p.data.float().std().cpu()
        )
        p.data.add_((noise * sc * ref).to(p.device, p.dtype))
        counts[kind] += 1
        n += 1
    from vllm.distributed.aux_output_connector import oc_common as _occ

    _occ.bump_weight_version()
    return counts if full else n


def probe_rows(salt):
    # A salted prompt never hits the prefix cache, so every row is computed with the
    # current weights.
    ids = text_ids(987654, 4096)
    o = llm.generate(
        [TokensPrompt(prompt_token_ids=ids, cache_salt=salt)],
        SamplingParams(max_tokens=1, routed_experts_prompt_start=0),
        use_tqdm=False,
    )[0]
    return np.asarray(o.outputs[0].routed_experts)


WCHANGE_EVERY = os.environ.get("OC_WEIGHT_CHANGE_EVERY") == "1"
GEN_DIST = os.environ.get(
    "OC_GEN_DIST", ""
)  # "lognorm:<mean>:<cap>" or "" (fixed MAXTOK)


def gen_len(s):
    if not GEN_DIST:
        return MAXTOK
    _, mean, cap = GEN_DIST.split(":")
    mean, cap = float(mean), int(cap)
    rng = np.random.default_rng(hash((s["i"], s["turn"], REP)) & 0xFFFFFFFF)
    sigma = 0.8
    v = rng.lognormal(np.log(mean) - sigma * sigma / 2, sigma)
    return int(min(cap, MAXTOK, max(16, v)))


SLEEP_EACH_STEP = os.environ.get("OC_SLEEP_EACH_STEP") == "1"


def probe_session(tag):
    """Write one salted probe request into the pool under its own session and read all
    three outputs back: used before and after the first weight update to show R3,
    indexer and hidden all change."""
    name = f"{REP}probe-{tag}"
    ids = text_ids(424242, 2048)
    o = llm.generate(
        [TokensPrompt(prompt_token_ids=ids, cache_salt=name)],
        SamplingParams(
            temperature=0.0,
            max_tokens=32,
            ignore_eos=True,
            logprobs=0,
            routed_experts_prompt_start=0,
            extra_args={"oc_session": name},
        ),
        use_tqdm=False,
    )[0]
    tick()
    m = get_manifest(name, o.request_id)
    if m is None:
        return None
    parts = get_many([oc.block_key(name, h) for h in m["blocks"]])
    tail = (get_many([oc.tail_key(name, o.request_id)])[0] or b"")[: m["tail_nbytes"]]
    r3 = np.frombuffer(
        b"".join(p for p in parts if p) + tail, dtype=m["dtype"]
    ).reshape((-1, *m["shape"]))
    fields = {f["name"]: f for f in m.get("fields", [])}
    idx = hid = None
    fi = fields.get("indexer_topk")
    if fi:
        keys = [oc.block_key(name, h) + fi["block_suffix"] for h in m["blocks"]] + [
            fi["tail_key"]
        ]
        idx = np.concatenate(
            [oc.idx_decode(b, fi["shape"]) for b in get_many(keys) if b and len(b) > 1]
        )
    fh = fields.get("hidden")
    if fh:
        hid = b"".join(
            v
            for v in get_many(
                [oc.hid_key(name, o.request_id, i) for i in range(fh["chunks"])]
            )
            if v
        )
    reader.remove_by_regex("^" + oc.PREFIX + "/" + name + "/", True)
    return dict(r3=r3, idx=idx, hid=hid)


def _rows_changed(a, b):
    if a is None or b is None:
        return None
    k = min(len(a), len(b))
    return int(np.any(a[:k].reshape(k, -1) != b[:k].reshape(k, -1), axis=1).sum()), k


def stale_step(ph):
    """One weight update between steps: level-1 sleep keeping the offloaded KV tier
    (PR #57810), wake up, perturb the routers, bump the aux weight version."""
    t = time.time()
    if SLEEP_EACH_STEP:
        llm.sleep(level=1, reset_connector=False)
        t_sleep = time.time() - t
        llm.wake_up()
    else:
        t_sleep = 0.0
    first = not ph.get("weight_changes")
    before = (
        probe_session(f"v{len(ph.get('weight_changes', []))}a")
        if first and EXTRA
        else None
    )
    n = llm.collective_rpc(
        _perturb_router, args=(WCHANGE_SCALE, 1234 + len(ph["step_s"]))
    )
    if before is not None:
        after = probe_session(f"v{len(ph.get('weight_changes', []))}b")
        ph["probe_change"] = dict(
            r3_rows_changed=_rows_changed(before["r3"], after["r3"]),
            idx_rows_changed=_rows_changed(before["idx"], after["idx"]),
            hid_bytes_equal=(before["hid"] == after["hid"])
            if before["hid"] is not None
            else None,
        )
        say("PROBE_CHANGE", json.dumps(ph["probe_change"]))
    rec = dict(
        after_step=len(ph["step_s"]),
        layers=n[0] if n else None,
        sleep_s=round(t_sleep, 2),
        total_s=round(time.time() - t, 2),
        host_used_gb=round(
            (meminfo()["MemTotal"] - meminfo()["MemAvailable"]) / 1e9, 1
        ),
    )
    ph.setdefault("weight_changes", []).append(rec)
    say("STALE_STEP", json.dumps(rec))


def weight_change(ph):
    before = probe_rows("probe-before")
    t = time.time()
    n = llm.collective_rpc(_perturb_router, args=(WCHANGE_SCALE, 1234))
    after = probe_rows("probe-after")
    k = min(len(before), len(after))
    diff = int(
        np.any(before[:k].reshape(k, -1) != after[:k].reshape(k, -1), axis=1).sum()
    )
    ph["weight_change"] = dict(
        after_step=len(ph["step_s"]),
        layers=n,
        scale=WCHANGE_SCALE,
        probe_rows=k,
        probe_rows_changed=diff,
        s=round(time.time() - t, 2),
    )
    say("WEIGHT_CHANGE", json.dumps(ph["weight_change"]))


for L, S in zip(LENGTHS, SESS):
    PHASE[0] = f"L{L}"
    P0 = L // TURNS
    D = (L - MAXTOK - 64 - P0) // max(
        1, TURNS - 1
    )  # context added per turn (reply + observation)
    OBS = D - MAXTOK
    # Bytes a session holds in the pool at most: every position's R3 row (+ indexer, +
    # hidden of the sampled tokens), x1.1 for tails/manifests. R3 row bytes from the
    # model config.
    r3_bpt = float(os.environ.get("OC_R3_BPT", "480"))
    bpt = r3_bpt + (IDX_BPT_EST if EXTRA else 0)
    sess_bytes = L * bpt * 1.1 + (TURNS * MAXTOK * 5124 if EXTRA else 0)
    C = S if ARM == "off" else max(1, min(S, int(C_MULT * 0.75 * POOL / sess_bytes)))
    say(
        "phase",
        L,
        "sessions",
        S,
        "turns",
        TURNS,
        "P0",
        P0,
        "obs",
        OBS,
        "session_bytes_est",
        int(sess_bytes),
        "C",
        C,
    )
    ph = dict(
        L=L,
        sessions=S,
        C=C,
        turns=TURNS,
        P0=P0,
        obs=OBS,
        requests=0,
        step_s=[],
        step_ctx=[],
        missing_manifest=0,
        failed_requests=0,
        r3_missing_objects=0,
        r3_missing_rows=0,
        r3_wrong_rows=0,
        r3_rows_checked=0,
        idx_manifest_missing=0,
        idx_missing_objects=0,
        idx_holes=0,
        idx_missing_rows=0,
        idx_out_of_range=0,
        idx_rows_checked=0,
        hid_manifest_missing=0,
        hid_missing_objects=0,
        hid_bad=0,
        hid_rows_checked=0,
        hid_lp_err=[],
        hid_lp_err_max=0.0,
        wrong_detail=[],
        read_s=0.0,
        release_s=[],
        released=0,
        unique_rows=0,
        gen_tokens=0,
        prefill_tokens=0,
    )
    w0 = last_writer_stats() or {}
    t_phase = time.time()
    pending = list(range(S))
    inflight = []
    while pending or inflight:
        while pending and len(inflight) < C:
            i = pending.pop(0)
            inflight.append(
                dict(
                    v0=len(ph.get("weight_changes", [])),
                    name=f"{REP}{DESIGN}-L{L}-{ARM}-{i}",
                    ids=text_ids(L * 100 + i, P0),
                    rows=None,
                    turn=0,
                    reqs=[],
                    i=i,
                )
            )
        prompts, params = [], []
        for s in inflight:
            aux = 0 if s["rows"] is None else min(len(s["rows"]), len(s["ids"]) - 1)
            s["aux"] = aux
            prompts.append(TokensPrompt(prompt_token_ids=list(s["ids"])))
            s["gen_len"] = gen_len(s)
            params.append(
                SamplingParams(
                    temperature=1.0,
                    max_tokens=s["gen_len"],
                    ignore_eos=True,
                    seed=s["i"] * 100 + s["turn"],
                    logprobs=0,
                    **({} if NOR3 else {"routed_experts_prompt_start": aux}),
                    extra_args={"oc_session": s["name"]},
                )
            )
            ph["prefill_tokens"] += len(s["ids"]) - aux
        ctx = max(len(s["ids"]) for s in inflight)
        guard.arm(f"L{L} generate ctx={ctx} inflight={len(inflight)}")
        t = time.time()
        outs = llm.generate(prompts, params, use_tqdm=False)
        dt = time.time() - t
        tick()
        guard.disarm()
        ph["step_s"].append(round(dt, 2))
        ph["step_ctx"].append(ctx)
        done = []
        for s, o in zip(inflight, outs):
            c = o.outputs[0]
            suffix = np.zeros(0, np.uint8) if NOR3 else np.asarray(c.routed_experts)
            rows = (
                suffix
                if s["rows"] is None or s["aux"] == 0
                else np.concatenate([s["rows"][: s["aux"]], suffix])
            )
            gen = list(c.token_ids)
            lps = [float(dd[t_].logprob) for dd, t_ in zip(c.logprobs or [], gen)]
            need = len(s["ids"]) + len(gen) - 1
            s["reqs"].append(
                (
                    o.request_id,
                    need,
                    rows,
                    gen,
                    lps,
                    len(s["ids"]),
                    s["aux"],
                    len(ph["step_s"]),
                )
            )
            s["rows"] = rows
            ph["gen_tokens"] += len(gen)
            obs = (
                text_ids(L * 1000 + s["i"] * 10 + s["turn"], max(16, D - len(gen)))
                if s["turn"] + 1 < TURNS
                else []
            )
            s["ids"] = list(s["ids"]) + gen + obs
            s["turn"] += 1
            if s["turn"] >= TURNS:
                done.append(s)
        say(
            "step",
            len(ph["step_s"]),
            "L",
            L,
            "ctx",
            ctx,
            "inflight",
            len(inflight),
            "gen_s",
            round(dt, 1),
            "pending",
            len(pending),
            "released",
            ph["released"],
        )
        if (
            WCHANGE_AFTER
            and len(ph["step_s"]) == WCHANGE_AFTER
            and "weight_change" not in ph
        ):
            weight_change(ph)
        if WCHANGE_EVERY and inflight and any(x["turn"] < TURNS for x in inflight):
            stale_step(ph)
        for s in done:
            ph["unique_rows"] += s["reqs"][-1][1]
            if ARM != "off":
                guard.arm(f"L{L} trainer read {s['name']}")
                tr = time.time()
                for j, rec in enumerate(s["reqs"]):
                    ph["requests"] += 1
                    check_request(
                        s, rec, ph, full=not READ_LAST_ONLY or j == len(s["reqs"]) - 1
                    )
                ph["read_s"] += time.time() - tr
                tr = time.time()
                reader.remove_by_regex("^" + oc.PREFIX + "/" + s["name"] + "/", True)
                ph["release_s"].append(round(time.time() - tr, 4))
                guard.disarm()
            else:
                ph["requests"] += len(s["reqs"])
            ph["released"] += 1
            inflight.remove(s)
            say(
                "released",
                s["name"],
                "r3_missing_rows",
                ph["r3_missing_rows"],
                "r3_wrong_rows",
                ph["r3_wrong_rows"],
                "idx_missing_rows",
                ph["idx_missing_rows"],
                "hid_bad",
                ph["hid_bad"],
            )
    ph["wall_s"] = round(time.time() - t_phase, 1)
    ph["gen_s_total"] = round(sum(ph["step_s"]), 1)
    time.sleep(7)  # one more periodic writer-stats line after the phase's last put
    w1 = last_writer_stats() or {}
    if w1.get("lat_hist"):
        h = np.array(w1["lat_hist"]) - np.array(
            w0.get("lat_hist") or [0] * len(w1["lat_hist"])
        )
        edges = np.logspace(-6, 3, 181)
        c = np.cumsum(h)
        if c[-1] > 0:
            for q in (50, 90, 99, 99.9):
                i = int(np.searchsorted(c, q / 100 * c[-1]))
                ph[f"put_lat_p{q}_ms"] = round(
                    float(edges[min(i, len(edges) - 1)]) * 1e3, 4
                )
            ph["put_lat_hist_n"] = int(c[-1])
    ph["writer_start"], ph["writer_end"] = w0, w1
    if w1:
        dp = w1.get("puts", 0) - w0.get("puts", 0)
        ph["objects"] = dp
        ph["objects_per_s_gen"] = (
            round(dp / ph["gen_s_total"], 1) if ph["gen_s_total"] else None
        )
        span = w1.get("t_last_put", 0) - max(w0.get("t_last_put", 0), t_phase)
        ph["objects_per_s_span"] = round(dp / span, 1) if span > 0 else None
        for k in ("r3", "idx", "hid", "manifest"):
            ph[f"bytes_{k}"] = w1.get(f"bytes_{k}", 0) - w0.get(f"bytes_{k}", 0)
        ph["r3_bytes_per_token"] = (
            round(ph["bytes_r3"] / ph["unique_rows"], 1) if ph["unique_rows"] else None
        )
        ph["idx_bytes_per_token"] = (
            round(ph["bytes_idx"] / ph["unique_rows"], 1) if ph["unique_rows"] else None
        )
        ph["hid_bytes_per_gen_token"] = (
            round(ph["bytes_hid"] / ph["gen_tokens"], 1) if ph["gen_tokens"] else None
        )
        tot = sum(ph[f"bytes_{k}"] for k in ("r3", "idx", "hid", "manifest"))
        ph["total_bytes_per_token"] = (
            round(tot / ph["unique_rows"], 1) if ph["unique_rows"] else None
        )
        ph["retries"] = w1.get("retries", 0) - w0.get("retries", 0)
        ph["blocked_puts"] = w1.get("blocked_puts", 0) - w0.get("blocked_puts", 0)
        for q in ("p50", "p90", "p99", "p99.9"):
            ph[f"put_lat_{q}_ms_cum"] = w1.get(f"put_lat_{q}_ms")
        ph["put_lat_max_ms_cum"] = w1.get("put_lat_max_ms")
    try:
        ph["engine_metrics"] = {
            m.name: getattr(m, "value", None)
            for m in llm.get_metrics()
            if any(
                k in m.name
                for k in ("prefix_cache", "offload", "external", "kv_transfer")
            )
            and getattr(m, "value", None) is not None
        }
    except Exception as e:  # metrics are diagnostics only
        ph["engine_metrics"] = repr(e)
    lp = ph.pop("hid_lp_err")
    ph["hid_lp_err_mean"] = float(np.mean(lp)) if lp else None
    ph["host_mem_peak_gb"] = MEM["phase_peak"].get(PHASE[0])
    ph["reader"] = dict(READ_STATS)
    STATS["phases"].append(ph)
    say(
        "PHASE_DONE",
        json.dumps(
            {k: v for k, v in ph.items() if k not in ("writer_start", "writer_end")},
            default=str,
        ),
    )

finish()
