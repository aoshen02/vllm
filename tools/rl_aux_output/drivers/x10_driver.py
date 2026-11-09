# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""X10: x9 plus GRPO groups and forced preemption (correctness tests for the
branch-splice and preemption fixes).
OC_GROUP=N        every session holds N trajectories that share the first prompt and the
                  session name (GRPO group); the session is released once all N are
                  read.
OC_TRAJ_TAG=1     send extra_args["oc_traj"] per trajectory (0 = the pre-fix behaviour:
                  branches of one session are indistinguishable to the connector).
OC_KV_BLOCKS=n    num_gpu_blocks_override, small enough that the scheduler preempts.
OC_COMPACT=1      context compaction (agent RL): each session is one rollout. A segment
                  (one trajectory, oc_traj = segment index) runs turns until its context
                  reaches OC_COMPACT_AT tokens, then a summary request on the same
                  trajectory (a summarize instruction, OC_SUMMARY_TOK generated tokens);
                  the next segment, a NEW trajectory of the same session, starts from
                  the fixed system prompt (OC_SYS_TOK tokens, the same for every
                  session) + the summary + the last 2 turns' raw tokens. OC_COMPACT_N
                  compactions, then the last segment runs to the threshold and the
                  session ends. Every segment's requests are checked like any
                  trajectory's; also: no segment is served the previous segment's rows
                  at its own positions, system-prompt blocks are shared across segments
                  (block objects listed vs unique), bytes estimated vs written.

X5-long: continuous multi-turn agent sessions whose contexts grow to 128K and 512K
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
               full  R3 + indexer top-k + hidden (OC_EXTRA=1); OC_EXTRA_TOPK=1 adds the
                     top-k logprobs (OC_TOPK, default 128), checked against the inline
                     logprobs=0
Checks: R3 delivered == inline (missing / wrong rows); indexer rows present for every
position, no missing objects, values in range; hidden chunks present for every sampled
token and lm_head(dequantized) logprob vs vLLM's logprob.
Metrics: bytes/token per field, objects/s, put latency p50/p99 (writer histogram), step
time per turn, host memory peak (node MemTotal - MemAvailable, 1 s samples).
Stall guard: an engine call longer than OC_GEN_TIMEOUT dumps py-spy stacks of this
process tree and exits with a DEADLOCK summary.
"""

import contextlib
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
NOR3 = (
    os.environ.get("OC_NOR3") == "1"
)  # R3 off: the baseline every overhead is measured against
DESIGN = os.environ.get("OC_DESIGN", "ps")
EXTRA = os.environ.get("OC_EXTRA", "0") == "1"
TOPK = EXTRA and os.environ.get("OC_EXTRA_TOPK", "0") == "1"
TOPK_K = int(os.environ.get("OC_TOPK", "128"))
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
GROUP = int(os.environ.get("OC_GROUP", "1"))
TRAJ_TAG = os.environ.get("OC_TRAJ_TAG", "1") == "1"
# OC_FORK=1: in a group, trajectories 1..N-1 fork from trajectory 0 after its first turn
# (its context and rows); OC_FORK_PARENT=1 sends oc_parent on the fork's first request
# (0 = control).
FORK = os.environ.get("OC_FORK") == "1"
FORK_PARENT = os.environ.get("OC_FORK_PARENT", "1") == "1"
FORK_DELAY = int(
    os.environ.get("OC_FORK_DELAY", "0")
)  # parent turns between snapshot and fork
FORK_CUT = float(
    os.environ.get("OC_FORK_CUT", "0")
)  # fork at this fraction of the parent's rows, at the fork turn
GEN_TIMEOUT = float(os.environ.get("OC_GEN_TIMEOUT", "1800"))
COMPACT = os.environ.get("OC_COMPACT") == "1"
COMPACT_AT = int(os.environ.get("OC_COMPACT_AT", "8192"))
COMPACT_N = int(os.environ.get("OC_COMPACT_N", "2"))
SUMMARY_TOK = int(os.environ.get("OC_SUMMARY_TOK", "256"))
SYS_TOK = int(os.environ.get("OC_SYS_TOK", "1024"))
COMPACT_TASK_TOK = int(os.environ.get("OC_COMPACT_TASK_TOK", "1024"))
COMPACT_OBS = int(os.environ.get("OC_COMPACT_OBS", "1024"))
COMPACT_MAX_TURNS = int(os.environ.get("OC_COMPACT_MAX_TURNS", "64"))
if COMPACT:
    assert GROUP == 1 and not FORK, (
        "OC_COMPACT: one trajectory at a time per session (no groups, no forks)"
    )
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


FINISHING = threading.Event()
FIN = dict(lock=threading.Lock(), t=None, written=threading.Event(), complete=False)
FINISH_WAIT_S = float(os.environ.get("OC_FINISH_WAIT_S", "120"))
FINISH_DEADLINE_S = float(os.environ.get("OC_FINISH_DEADLINE_S", "600"))
STORE_CALL_S = float(os.environ.get("OC_STORE_CALL_S", "120"))


SUMMARY_LOCK = threading.Lock()


HARD_LOCK_S = float(os.environ.get("OC_HARD_LOCK_S", "10"))
HARD_WRITE_S = float(os.environ.get("OC_HARD_WRITE_S", "5"))


def _replace_out(data, path):
    tmp = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}"
    with open(tmp, "w") as f:
        f.write(data)
    os.replace(tmp, path)


def write_summary(data, complete, force=False):
    """Every summary write: under one lock, to a temp file moved over OUT, so a reader
    (or an exit in between) only ever sees a whole summary. A minimal one never replaces
    a complete one. Once a hard exit is under way, OUT is its alone: this write goes to
    OUT.finish.json."""
    with SUMMARY_LOCK:
        if not complete and not force and FIN.get("complete"):
            return
        _replace_out(data, OUT + ".finish.json" if FIN.get("hard") else OUT)
        if complete:
            FIN["complete"] = True


def _hard_marked(st):
    """A summary dict as a hard exit leaves it: never ok, with the exit code and
    reason."""
    hard = FIN.get("hard")
    if hard:
        st = dict(st)
        st.update(ok=False, exit_code=hard[0], exit_reason=hard[1])
        st["fail_reasons"] = list(st.get("fail_reasons", [])) + [
            f"exit {hard[0]}: {hard[1]}"
        ]
    return st


def write_min_summary(code, reason):
    """Last resort before an exit: whatever STATS holds, or a stub."""
    try:
        st = globals().get("STATS")
        st = dict(st) if isinstance(st, dict) else {}
        st.update(ok=False, exit_code=code, exit_reason=reason)
        data = json.dumps(st, indent=1, default=str)
    except BaseException:  # noqa: BLE001
        data = json.dumps(dict(ok=False, exit_code=code, exit_reason=reason))
    write_summary(data, complete=False)


def hard_exit(code, reason):
    """Every os._exit path. The hard flag goes up first (a finish completing meanwhile
    marks its own summary ok=False with this exit); a running finish gets a bounded
    wait; then this exit writes the last summary itself. If the summary lock cannot be
    had in OC_HARD_LOCK_S, it writes OUT.hard.json instead. Then the process ends."""
    try:
        FIN.setdefault("hard", (code, reason))
        if FIN["lock"].locked() and not FIN["written"].is_set():
            FIN["written"].wait(FINISH_WAIT_S)
        try:
            st = globals().get("STATS")
            data = json.dumps(
                _hard_marked(dict(st) if isinstance(st, dict) else {}),
                indent=1,
                default=str,
            )
        except BaseException:  # noqa: BLE001
            data = json.dumps(_hard_marked({}))
        locked = SUMMARY_LOCK.acquire(timeout=HARD_LOCK_S)

        def write():
            # Without the lock a normal finish may be mid-write: OUT still gets this
            # summary (the finish's own write goes to OUT.finish.json once it sees the
            # flag), plus OUT.hard.json.
            try:
                _replace_out(data, OUT)
                if not locked:
                    _replace_out(data, OUT + ".hard.json")
            finally:
                if locked:
                    SUMMARY_LOCK.release()

        th = threading.Thread(target=write, daemon=True)
        th.start()
        th.join(HARD_WRITE_S)
    except BaseException:  # noqa: BLE001
        pass
    finally:
        os._exit(code)


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
            if FINISHING.is_set():
                if FIN["t"] is not None and time.time() - FIN["t"] > FINISH_DEADLINE_S:
                    hard_exit(11, f"finish exceeded {FINISH_DEADLINE_S:.0f} s")
                continue
            for name in (
                "check_helpers",
                "check_writer_fatal",
            ):  # defined further down the script
                poll = globals().get(name)
                if poll is not None:
                    poll()
            t_armed, tag = self.t, self.tag
            if t_armed is not None and time.time() - t_armed > GEN_TIMEOUT:
                try:
                    try:
                        d = dump_stacks(tag)
                    except BaseException as e:  # noqa: BLE001
                        d = f"dump failed: {e!r}"
                    STATS["deadlock"] = dict(
                        where=tag, after_s=round(time.time() - t_armed), dump=d
                    )
                    finish(deadlock=True)
                finally:
                    hard_exit(9, "deadlock")


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
    if os.environ.get("OC_MC_SSD_PATH"):
        try:
            sys.path.insert(0, os.environ["OC_DIR"])
            import oc_common as oc  # noqa: E402

            oc.check_ssd_pin()
            os.environ["OC_MC_SSD_TOKEN"] = oc.ssd_prepare_dir(
                os.path.expandvars(os.environ["OC_MC_SSD_PATH"])
            )
        except BaseException as e:  # noqa: BLE001
            write_min_summary(1, f"SSD setup rejected: {e!r}")
            raise
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
import glob as _glob  # noqa: E402
import uuid as _uuid  # noqa: E402

DRIVER_T0 = time.time()
# Session names carry a per-run nonce: a release marker of an earlier run (or phase)
# never matches.
RUN_NONCE = _uuid.uuid4().hex[:8]
for _f in _glob.glob(_glob.escape(OUT + ".writer") + ".*"):
    # A previous run's writer reports on the same OUT are not this run's writers.
    with contextlib.suppress(OSError):
        os.remove(_f)
RELEASE_OWNER = ".oc_release_owner"
if os.environ.get("OC_RELEASE_DIR"):
    _rd = os.path.realpath(os.environ["OC_RELEASE_DIR"])
    if not os.path.isdir(_rd):
        os.makedirs(_rd)
        open(os.path.join(_rd, RELEASE_OWNER), "w").close()
    elif os.path.exists(os.path.join(_rd, RELEASE_OWNER)):
        # A release dir a driver created (its marker): its old release markers are stale
        # state.
        for _n in os.listdir(_rd):
            if _n != RELEASE_OWNER:
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(_rd, _n))
import oc_common as oc  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402

if int(os.environ.get("OC_INJECT_ERR_EVERY", "0") or 0):
    # Before the engine starts: its worker processes inherit the log the checkers read.
    os.environ.setdefault("OC_INJECT_LOG", os.environ.get("OUT", "oc") + ".inject")
from vllm.inputs import TokensPrompt  # noqa: E402

MAXLEN = max(
    max(LENGTHS) + MAXTOK + 1024,
    COMPACT_AT + COMPACT_OBS + MAXTOK + SUMMARY_TOK + 1024 if COMPACT else 0,
)
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
    max_num_seqs=int(os.environ.get("OC_MAX_SEQS", "64")),
    max_num_batched_tokens=int(os.environ.get("OC_MBT", "8192")),
    enable_prefix_caching=True,
    **(
        dict(swa_bounded_replay=False) if os.environ.get("OC_SWA_REPLAY") == "0" else {}
    ),
    **(
        dict(max_logprobs=int(os.environ["OC_LOGPROBS"]))
        if int(os.environ.get("OC_LOGPROBS", "0")) > 20
        else {}
    ),
    **(
        dict(
            speculative_config={
                "method": "dspark",
                "num_speculative_tokens": int(os.environ["OC_SPEC"]),
            }
        )
        if os.environ.get("OC_SPEC")
        else {}
    ),
    **(
        dict(num_gpu_blocks_override=int(os.environ["OC_KV_BLOCKS"]))
        if os.environ.get("OC_KV_BLOCKS")
        else {}
    ),
    **(
        dict(enable_sleep_mode=True)
        if os.environ.get("OC_SLEEP_EACH_STEP") == "1"
        else {}
    ),
    **(
        dict(disable_log_stats=False)
        if os.environ.get("OC_OFFLOAD_GB") or os.environ.get("OC_STATS") == "1"
        else {}
    ),
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
        if os.environ.get("OC_OFFLOAD_GB") and not os.environ.get("OC_OFFLOAD_SPEC")
        else {}
    ),
    **(
        dict(
            kv_transfer_config={
                "kv_connector": "OffloadingConnector",
                "kv_role": "kv_both",
                "kv_connector_extra_config": json.loads(
                    os.path.expandvars(os.environ["OC_OFFLOAD_SPEC"])
                ),
            }
        )
        if os.environ.get("OC_OFFLOAD_SPEC")
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
try:
    _cc = llm.llm_engine.vllm_config.cache_config
    STATS["kv"] = dict(num_gpu_blocks=_cc.num_gpu_blocks, block_size=_cc.block_size)
    say("KV_CAPACITY", json.dumps(STATS["kv"]))
except Exception as e:  # diagnostics only
    say("KV_CAPACITY_ERR", repr(e))
tok = llm.get_tokenizer()
reader = oc.make_store(segment_bytes=0, buffer_bytes=2 << 30) if ARM != "off" else None
guard = Guard()


def _uncaught(tp, e, tb):
    # A rejection or crash in the main flow still goes through cleanup and writes the
    # summary.
    STATS["error"] = repr(e)
    STATS["ok"] = False
    try:
        finish()
    finally:
        sys.__excepthook__(tp, e, tb)


sys.excepthook = _uncaught

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
SSD = bool(os.environ.get("OC_MC_SSD_PATH")) and bool(procs)
RESERVED: dict = {}
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
        if not SSD:
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
    if SSD:
        # One spaced round (1 s in all) for every key that read empty, whatever their
        # number.
        READ_STATS["single_retries"] += oc.refetch_missing(reader.get, keys, out)
        READ_STATS["read_fail_keys"] += sum(v is None for v in out)
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
            READ_STATS["write_errors"] = READ_STATS.get("write_errors", 0) + 1
            say("WRITE_ERROR", session, rid, bytes(err)[:300])
            return None
        time.sleep(0.05)
    return None


_REF = {}
_REF_ACC = {}
_REF_SEEN = set()
_REF_PIDS: list = []


def ref_rows(track):
    """Rows this trajectory computed itself (first computation per position), from the
    worker's per-step dumps: an independent reference, unlike inline rows, which read
    hit positions back from the base store. New dump files are folded in on every call.
    """
    import glob

    import torch

    commit = os.environ.get("OC_DUMP_COMMIT") == "1"
    pat = "c*.pt" if commit else "s*.pt"
    files = sorted(glob.glob(os.path.join(os.environ["OC_DUMP"], pat)))
    if commit:
        # One file series per worker process (TP rank): the reference uses the first
        # rank only.
        pids = sorted({os.path.basename(f)[1:].split("_", 1)[0] for f in files})
        if pids:
            _REF_PIDS[:] = pids
            files = [
                f for f in files if os.path.basename(f)[1:].split("_", 1)[0] == pids[0]
            ]
    new = [f for f in files if f not in _REF_SEEN]
    for f in new:
        _REF_SEEN.add(f)
        d = torch.load(f, weights_only=False)
        if commit:
            for tr, start, rows in d["commit"]:
                if tr is not None:
                    _REF_ACC.setdefault(tr, []).append(
                        (np.arange(len(rows)) + start, rows)
                    )
                    _REF.pop(tr, None)
            continue
        for i, tr in enumerate(d["sess"]):
            if tr is None:
                continue
            a, b = int(d["qsl"][i]), int(d["qsl"][i + 1])
            _REF_ACC.setdefault(tr, []).append(
                (np.arange(b - a) + int(d["tstart"][i]), d["r3"][a:b])
            )
            _REF.pop(tr, None)
    if track not in _REF and track in _REF_ACC:
        parts = _REF_ACC[track]
        pos = np.concatenate([p for p, _ in parts])
        rows = np.concatenate([r for _, r in parts])
        u, first = np.unique(pos, return_index=True)
        _REF[track] = (u, rows[first])
    return _REF.get(track)


def check_cones(s, rec, m, ph, got):
    """Deep cells of the delivered rows, the way a reader assembles them; returns
    them overlaid."""
    deep = oc.R3_FIRST_TRIMMED
    layers = list(range(deep, got.shape[1]))
    no = np.iinfo(got.dtype).max
    cone = m.get("r3_cone")
    if cone is None:
        full = got
    else:
        rp = [p for p in cone["patches"] if "key" in p]
        vals = get_many([p["key"] for p in rp])
        if any(v is None for v in vals):
            request_failed(ph, s, rec[0], "absent cone patch")
        ph["cone_bytes"] = ph.get("cone_bytes", 0) + sum(len(v) for v in vals if v)
        ph["cone_reported_uncovered"] = (
            ph.get("cone_reported_uncovered", 0)
            + int(cone.get("uncovered", 0))
            + int(cone.get("uncovered_idx", 0))
        )
        full, _, ver = oc.overlay_cones(
            got, deep, no, [(v, p["end"]) for v, p in zip(vals, rp) if v]
        )
        if len(s["reqs"]) and (ver[ver >= 0].size):
            ph["cone_versions_seen"] = max(
                ph.get("cone_versions_seen", 0), int(len(np.unique(ver[ver >= 0])))
            )
    ends = s.get("parent_ends", []) + [
        (r[5], r[1])
        for r in s["reqs"][: next(i for i, r in enumerate(s["reqs"]) if r is rec) + 1]
    ]
    mask = oc.cone_mask(len(full), ends, layers)
    cells = (full[:, deep:] == no).all(-1)
    ph["cone_cells"] = ph.get("cone_cells", 0) + int(mask.sum())
    ph["cone_missing_cells"] = ph.get("cone_missing_cells", 0) + int(
        (cells & mask).sum()
    )
    if (cells & mask).any() and len(ph.setdefault("cone_detail", [])) < 20:
        q = np.argwhere(cells & mask)
        ph["cone_detail"].append(
            dict(
                session=s["name"],
                rid=rec[0],
                n=len(q),
                first=q[0].tolist(),
                last=q[-1].tolist(),
                ends=ends[-3:],
                patches=[
                    (p["lo"], p["end"], p["prompt_end"])
                    for p in (cone or {}).get("patches", [])
                ][-3:],
            )
        )
    if os.environ.get("OC_R3_SAVE"):
        import hashlib

        kept = np.where(mask[:, :, None], full[:, deep:], no)
        ph.setdefault("cone_digests", {})[str(s["i"])] = [
            hashlib.md5(np.ascontiguousarray(kept).tobytes()).hexdigest(),
            int(mask.sum()),
            hashlib.md5(np.ascontiguousarray(full[:, :deep]).tobytes()).hexdigest(),
        ]
        if s["i"] < int(os.environ["OC_R3_SAVE"]):
            s["deliver"] = dict(
                rows=full,
                turns=np.array([(r[5], len(r[3])) for r in s["reqs"]], np.int64),
                lp=np.concatenate([np.asarray(r[4], np.float32) for r in s["reqs"]]),
            )
    return full


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
    if s.get("parent_rid0") and "prev_rows" not in s:
        # A fork's previous request is its parent's request at the fork point.
        pm = get_manifest(s["name"], s["parent_rid0"])
        if pm is not None:
            s["prev_rows"] = int(pm.get("rows", 0))
            if not FORK_CUT:
                s["parent_rows0"] = int(pm.get("rows", 0))
            s["parent_need0"] = s["reqs"][0][6] if len(s["reqs"][0]) > 6 else None
    m = get_manifest(s["name"], rid)
    prev_missing = s.get("prev_missing", False)
    s["prev_missing"] = m is None
    if m is None:
        # The request failed (error marker) or its manifest is missing. Its manifest
        # body, when it exists, still tells the checker how many rows the request
        # computed (truncated-draft window of later requests); the request itself is
        # counted as failed below.
        raw_m = reader.get(oc.manifest_key(s["name"], rid))
        if raw_m:
            rows_m = int(json.loads(bytes(raw_m)).get("rows", 0))
            s.setdefault("mrows", {})[rid] = rows_m
            s["prev_rows"] = rows_m
            s["prev_missing"] = False
    if prev_missing and m is not None:
        # The previous request's manifest is missing (reported): its row count is
        # unknown, so where this request's own rows begin (aux or aux + 1) and its
        # truncated-draft window are unknown too. Its version segments are not checked;
        # see the ref window below.
        ph["wv_unchecked"] = ph.get("wv_unchecked", 0) + 1
    if PAUSE_PERTURB and m is not None:
        # Versions also change inside a step: the per-step expectation does not hold;
        # the reference (first computation) and OC_ALT check these runs instead.
        ph["wv_unchecked"] = ph.get("wv_unchecked", 0) + 1
    if (
        m is not None
        and not prev_missing
        and not PAUSE_PERTURB
        and (WCHANGE_AFTER or WCHANGE_EVERY)
        and len(rec) >= 8
    ):
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
            k = (
                next(n for n, r in enumerate(s["reqs"]) if r is rec)
                + 1
                + s.get("k0", 0)
            )  # a fork starts at turn 1
            v0 = s.get("v0", 0)
            # The new version starts where this request first computed: its aux start,
            # or later when the previous request already computed past it (speculative
            # decoding verifies the last sampled position in the previous request's
            # final step). Positions past aux in the previous request's rows belong to
            # drafts the scheduler truncated at max_tokens, not to this prompt's tokens.
            start_new = (
                (aux + 1 if s.get("prev_rows", 0) > aux else aux) if k > 1 else 0
            )
            ok = (
                isinstance(wv, list)
                and len(wv) == k
                and [x[1] for x in wv] == list(range(v0, v0 + k))
                and wv[0][0] == 0
                and (k == 1 or wv[-1][0] == start_new)
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
    s["prev_rows_before"] = s.get("prev_rows", 0)
    if m is not None:
        s["prev_rows"] = int(m.get("rows", 0))
        s.setdefault("mrows", {})[rid] = int(m.get("rows", 0))
        if m.get("invalid_from") is not None:
            # Rows computed on invalid KV (engine rewind): reported, never delivered as
            # valid.
            ph["invalid_rows"] = ph.get("invalid_rows", 0) + max(
                0, need - int(m["invalid_from"])
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
            if len(ph.setdefault("missing_detail", [])) < 20:
                ph["missing_detail"].append(
                    dict(
                        session=s["name"],
                        rid=rid,
                        need=need,
                        got=len(got),
                        rows=m["rows"],
                        tail_start=m["tail_start"],
                        tail_nbytes=m["tail_nbytes"],
                        nblocks=len(m["blocks"]),
                    )
                )
        n = min(len(got), need)
        partial = "r3_cone" in m
        full_ = check_cones(s, rec, m, ph, got[:n])
        if (s["name"], rid) in FAILED_REQ:
            return
        if os.environ.get("OC_READER_CHECK") == "1":
            import oc_reader

            smp = oc_reader.read_sample(
                lambda k: (lambda v: bytes(v) if v else None)(reader.get(k)),
                oc.manifest_key(s["name"], rid),
                n,
                want=("r3",),
                manifest=m,
                exists=getattr(reader, "batch_is_exist", None),
                store=reader,
            )
            ph["reader_n"] = ph.get("reader_n", 0) + 1
            ph["reader_not_ok"] = ph.get("reader_not_ok", 0) + int(not smp.ok)
            if smp.r3 is not None and len(smp.r3) == n:
                rm = smp.r3_replay_mask
                diff = (smp.r3 != full_).any(-1) & rm
                ph["reader_r3_diff_cells"] = ph.get("reader_r3_diff_cells", 0) + int(
                    diff.sum()
                )
                ph["reader_r3_replay_cells"] = ph.get(
                    "reader_r3_replay_cells", 0
                ) + int(rm.sum())
            nm = getattr(smp, "r3_needed_missing", None)
            if nm is not None:
                ph["reader_needed_missing"] = ph.get("reader_needed_missing", 0) + int(
                    nm.sum()
                )
            if not smp.ok and len(ph.setdefault("reader_detail", [])) < 10:
                ph["reader_detail"].append(
                    dict(
                        session=s["name"],
                        rid=rid,
                        error=smp.error,
                        counts=dict(smp.missing.counts),
                    )
                )
        shallow_rows = got[:n, : oc.R3_FIRST_TRIMMED] if partial else got[:n]
        inl = inline[:n, : oc.R3_FIRST_TRIMMED] if partial else inline[:n]
        bad = np.any(shallow_rows.reshape(n, -1) != inl.reshape(n, -1), axis=1)
        aux_ = rec[6] if len(rec) > 6 else 0
        if aux_ > 0 and (s.get("prev_rows_before", 0) > aux_ or prev_missing):
            # With speculative decoding the previous request computed position aux
            # first; the inline rows of this and every later request carry this
            # request's recompute there, so these positions are checked by the reference
            # instead.
            s.setdefault("skip_pos", set()).add(aux_)
        for q in s.get("skip_pos", ()):
            if q < n:
                bad[q] = False
        ph["r3_wrong_rows"] += int(bad.sum())
        ph["r3_rows_checked"] += n
        if os.environ.get("OC_DUMP"):
            track = track_of(s)
            ref = ref_rows(track)
            if ref is not None and s.get("k0"):
                # A fork's positions below its first aux start are its parent's (first
                # computation wins); the reference takes them from the parent
                # trajectory.
                ptrack = track.rsplit("", 1)[0] + "0"
                pref = ref_rows(ptrack)
                a0 = s["reqs"][0][6]
                if pref is not None:
                    # The parent computed a0 itself when its rows reach past it
                    # (speculative decoding): that row came first as well.
                    a1 = a0 + 1 if (pref[0] == a0).any() else a0
                    keep = ref[0] >= a1
                    pk = pref[0] < a1
                    order = np.argsort(
                        np.concatenate([pref[0][pk], ref[0][keep]]), kind="stable"
                    )
                    ref = (
                        np.concatenate([pref[0][pk], ref[0][keep]])[order],
                        np.concatenate([pref[1][pk], ref[1][keep]])[order],
                    )
            if ref is not None:
                pos, rr = ref
                sel = pos < n
                spec = int(os.environ.get("OC_SPEC", "0") or 0)
                if spec:
                    # Rows earlier requests captured past their last real token belong
                    # to drafts the scheduler truncated (with async scheduling, several
                    # steps' worth): positions need_j < p < rows_j of every earlier
                    # request j.
                    for r_ in s["reqs"]:
                        rows_j = s.get("mrows", {}).get(r_[0])
                        if rows_j is not None and rows_j > r_[1] + 1:
                            sel &= ~((pos > r_[1]) & (pos < rows_j))
                    if prev_missing:
                        a_ = rec[6] if len(rec) > 6 else 0
                        sel &= ~((pos >= a_) & (pos < a_ + 4 * spec))
                    pr = s.get("parent_rows0")
                    if pr is not None and s.get("parent_need0") is not None:
                        sel &= ~((pos > s["parent_need0"]) & (pos < pr))
                k = int(sel.sum())
                if partial:
                    rb = np.any(
                        got[pos[sel], : oc.R3_FIRST_TRIMMED].reshape(k, -1)
                        != rr[sel][:, : oc.R3_FIRST_TRIMMED].reshape(k, -1),
                        axis=1,
                    )
                else:
                    rb = np.any(
                        got[pos[sel]].reshape(k, -1) != rr[sel].reshape(k, -1), axis=1
                    )
                ph["ref_checked"] = ph.get("ref_checked", 0) + k
                ph["ref_wrong"] = ph.get("ref_wrong", 0) + int(rb.sum())
                ph["ref_unchecked"] = ph.get("ref_unchecked", 0) + n - k
                if rb.any() and not ph.get("ref_probe_saved"):
                    ph["ref_probe_saved"] = True
                    np.savez_compressed(
                        os.path.join(
                            os.environ.get("OC_PROBE_DIR", os.path.dirname(OUT)),
                            f"probe_{rid}.npz",
                        ),
                        got=got[:n],
                        inline=inline[:n],
                        ref_pos=pos,
                        ref_rows=rr,
                        blocks=np.array(m["blocks"]),
                        tail_start=m["tail_start"],
                        aux=rec[6] if len(rec) > 6 else -1,
                    )
                    say(
                        "REF_PROBE",
                        s["name"],
                        s.get("traj"),
                        rid,
                        "blocks",
                        m["blocks"][:60],
                    )
                if rb.any() and int(rb.sum()) <= 4 and ph.get("ref_trace_n", 0) < 6:
                    # Every computation of these positions in the dumps, in step order.
                    ph["ref_trace_n"] = ph.get("ref_trace_n", 0) + 1
                    for q in pos[sel][rb][:2]:
                        hist = []
                        for k_, (pp, rr_) in enumerate(_REF_ACC.get(track, [])):
                            hit = np.nonzero(pp == q)[0]
                            for h_ in hit:
                                hist.append((k_, hash(rr_[h_].tobytes()) % 100000))
                        say(
                            "REF_TRACE",
                            s["name"],
                            s.get("traj"),
                            rid,
                            "pos",
                            int(q),
                            "delivered",
                            hash(got[q].tobytes()) % 100000,
                            "computations",
                            hist[:8],
                        )
                if rb.any() and len(ph.setdefault("ref_detail", [])) < 20:
                    q = pos[sel][rb]
                    ph["ref_detail"].append(
                        dict(
                            session=s["name"],
                            traj=s.get("traj"),
                            rid=rid,
                            first=int(q[0]),
                            last=int(q[-1]),
                            n=int(len(q)),
                        )
                    )
        if (
            bad.any()
            and not ph.get("wrong_probe_saved")
            and os.environ.get("OC_PROBE_DIR")
        ):
            ph["wrong_probe_saved"] = True
            np.savez_compressed(
                os.path.join(
                    os.environ["OC_PROBE_DIR"], f"wrong_{s['name']}_{rid}.npz"
                ),
                got=got[:n],
                inline=inline[:n],
                blocks=np.array(m["blocks"]),
                tail_start=m["tail_start"],
                rows=m["rows"],
                aux=rec[6] if len(rec) > 6 else -1,
            )
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
    if TOPK:
        check_topk(ph, s, rid, fields.get("topk"), gen_ids, lps, P)
        if (s["name"], rid) in FAILED_REQ:
            return
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
        if (miss or fi.get("missing", 0)) and len(
            ph.setdefault("idx_missing_detail", [])
        ) < 30:
            absent = [
                i
                for i, (k, b) in enumerate(zip(keys, vals))
                if not b or (k == fi.get("tail_key") and len(b) <= 1)
            ]
            ph["idx_missing_detail"].append(
                dict(
                    session=s["name"],
                    traj=s.get("traj"),
                    rid=rid,
                    need=need,
                    aux=rec[6] if len(rec) > 6 else None,
                    reported=fi.get("missing", 0),
                    absent_idx=absent[:12],
                    nblocks=len(m["blocks"]),
                    tail_rows=fi["tail_rows"],
                    first_rid=s["reqs"][0][0],
                )
            )
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
            cone_ = m.get("r3_cone")
            if os.environ.get("OC_R3_SAVE") or cone_ is not None:
                import hashlib

                rn = np.asarray(r.cpu()) if hasattr(r, "cpu") else np.asarray(r)
                slots = [
                    j
                    for j, lyr in enumerate(fi["layers"])
                    if lyr >= oc.R3_FIRST_TRIMMED
                ]
                s0 = slots[0]
                if cone_ is not None:
                    plist = [p for p in cone_["patches"] if "idx_key" in p]
                    vals_ = get_many([p["idx_key"] for p in plist])
                    if any(v is None for v in vals_):
                        request_failed(ph, s, rid, "absent indexer cone patch")
                        return
                    per = (len(slots), fi["shape"][1])
                    rn, _, _ = oc.overlay_cones(
                        rn,
                        s0,
                        oc.IDX_NO_REPLAY,
                        [(v, p["end"]) for v, p in zip(vals_, plist) if v],
                        unpack=lambda d: d
                        if d.dtype != np.uint8
                        else oc.unpack_idx_rows(d.tobytes(), per, fi.get("codec")),
                    )
                ends_ = s.get("parent_ends", []) + [(r_[5], r_[1]) for r_ in s["reqs"]]
                mask_ = oc.cone_mask(len(rn), ends_, [fi["layers"][j] for j in slots])
                gone = (rn[:, s0:] == oc.IDX_NO_REPLAY).all(-1)
                ph["cone_idx_cells"] = ph.get("cone_idx_cells", 0) + int(mask_.sum())
                ph["cone_idx_missing_cells"] = ph.get(
                    "cone_idx_missing_cells", 0
                ) + int((gone & mask_).sum())
                kept_ = np.where(mask_[:, :, None], rn[:, s0:], oc.IDX_NO_REPLAY)
                ph.setdefault("idx_digests", {})[str(s["i"])] = [
                    need,
                    hashlib.md5(np.ascontiguousarray(kept_).tobytes()).hexdigest(),
                    hashlib.md5(np.ascontiguousarray(rn[:, :s0]).tobytes()).hexdigest(),
                ]
                if s["i"] < int(os.environ.get("OC_R3_SAVE", "0")) and "deliver" in s:
                    np.savez(
                        os.environ["OUT"] + f".deliver{s['i']}.npz",
                        idx=rn,
                        idx_layers=np.array(fi["layers"]),
                        **s.pop("deliver"),
                    )
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
    why = oc.hid_check(fh, vals)
    if why == "encoding":
        ph["hid_bad"] += 1
        return
    # The manifest's per-chunk row counts must describe the stored chunks exactly.
    ph["hid_meta_checked"] = ph.get("hid_meta_checked", 0) + 1
    if why == "meta":
        ph["hid_meta_bad"] = ph.get("hid_meta_bad", 0) + 1
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


def check_topk(ph, s, rid, ft, gen_ids, lps, P):
    """Stored top-k rows of the generated tokens (positions P - 1 ...) against the
    sampled tokens and their inline logprobs (logprobs=0): distinct ids, non-increasing
    logprobs, the sampled token's stored logprob equal to the inline one when it is in
    the top-k."""
    if ft is None:
        ph["topk_manifest_missing"] = ph.get("topk_manifest_missing", 0) + 1
        return
    vals = get_many([oc.topk_key(s["name"], rid, i) for i in range(ft["chunks"])])
    if any(v is None for v in vals):
        request_failed(ph, s, rid, "absent topk chunk")
        return
    r = oc.topk_check(
        ft, vals, gen_ids, lps, P - 1, float(os.environ.get("OC_TOPK_LP_TOL", "1e-5"))
    )
    ph["topk_lp_err_max"] = max(ph.get("topk_lp_err_max", 0.0), r.pop("lp_err_max"))
    for k_, v_ in r.items():
        ph["topk_" + k_] = ph.get("topk_" + k_, 0) + v_
    ph["topk_bytes"] = ph.get("topk_bytes", 0) + sum(len(v) for v in vals)
    if (r["meta_bad"] or r["bad_rows"] or r["lp_mismatch"]) and len(
        ph.setdefault("topk_detail", [])
    ) < 10:
        ph["topk_detail"].append(
            dict(
                session=s["name"],
                rid=rid,
                start=ft["start"],
                rows=ft["rows"],
                P=P,
                G=len(gen_ids),
                **r,
            )
        )


FAIL_KEYS = (
    "missing_manifest",
    "r3_missing_objects",
    "r3_missing_rows",
    "r3_wrong_rows",
    "idx_manifest_missing",
    "idx_missing_objects",
    "idx_missing_rows",
    "idx_out_of_range",
    "hid_manifest_missing",
    "hid_missing_objects",
    "hid_bad",
    "hid_meta_bad",
    "ref_wrong",
    "failed_requests",
    "topk_manifest_missing",
    "topk_meta_bad",
    "topk_bad_rows",
    "topk_lp_mismatch",
)


def _step(name, fn):
    try:
        fn()
    except BaseException as e:  # noqa: BLE001
        STATS.setdefault("cleanup_errors", []).append(f"{name}: {e!r}")


def _fail(reason):
    STATS["ok"] = False
    STATS.setdefault("fail_reasons", []).append(reason)


def finish(deadlock=False):
    """Once per run: the first thread in owns it and writes the summary; a later caller
    waits for that summary (bounded). Every step is guarded; store calls are bounded
    (OC_STORE_CALL_S)."""
    if not FIN["lock"].acquire(blocking=False):
        FIN["written"].wait(FINISH_WAIT_S)
        return
    FIN["t"] = time.time()
    FINISHING.set()
    deadline = threading.Timer(
        FINISH_DEADLINE_S, hard_exit, (11, f"finish exceeded {FINISH_DEADLINE_S:.0f} s")
    )
    deadline.daemon = True
    deadline.start()
    try:
        STATS["host_mem"] = MEM
        STATS["deadlock_detected"] = deadlock
        STATS["reader"] = READ_STATS
        STATS.setdefault("ok", True)
        _step(
            "writer_stats",
            lambda: STATS.__setitem__("writer_final", last_writer_stats()),
        )
        if reader is not None and not deadlock:
            if FENCE_ON:
                _step(
                    "fences",
                    lambda: settle_fences(
                        STATS, wait_s=float(os.environ.get("OC_FENCE_WAIT_S", "30"))
                    ),
                )
                if STATS.get("fences_open"):
                    _fail(f"fences_open={STATS['fences_open']}")
                if len(WRITERS) < WRITERS_EXPECTED[0]:
                    _fail(f"writers_registered={len(WRITERS)}/{WRITERS_EXPECTED[0]}")

            _step(
                "leak_probe", lambda: STATS.__setitem__("leaked_sample", leak_probe())
            )

            def sweep():
                how, n = oc.call_bounded(
                    lambda: reader.remove_by_regex("^" + oc.PREFIX + "/" + REP, True),
                    STORE_CALL_S,
                )
                if how != "ok":
                    raise RuntimeError(f"final sweep {how}: {n!r}")
                STATS["leaked_keys"] = n
                if n:
                    # Objects left after every session was released: a release missed
                    # them.
                    _fail(f"leaked_keys={n}")

            _step("sweep", sweep)
        if deadlock:
            _fail("deadlock")
        if STATS.get("release_failed"):
            _fail(f"release_failed={len(STATS['release_failed'])}")
        if STATS.get("writer_fatal"):
            _fail("put_stall")
        if STATS.get("helper_died"):
            _fail("helper_died")
        for k in FAIL_KEYS:
            n = sum(int(ph.get(k, 0) or 0) for ph in STATS.get("phases", []))
            if n:
                _fail(f"{k}={n}")
        if READ_STATS.get("read_fail_keys"):
            _fail(f"read_fail_keys={READ_STATS['read_fail_keys']}")
        _step("rank_dump", _rank_dump_compare)
        for p in procs:
            _step("terminate", p.terminate)
        for p in procs:

            def wait(p=p):
                try:
                    p.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait(timeout=30)

            _step("wait", wait)
        if procs and all(p.poll() is not None for p in procs):
            # Our holder is dead: the offload dir this process created goes with it.
            _step(
                "ssd_dir",
                lambda: STATS.__setitem__(
                    "ssd_dir_removed",
                    oc.ssd_remove_owned(os.environ.get("OC_MC_SSD_PATH")),
                ),
            )
        if STATS.get("cleanup_errors"):
            STATS["ok"] = False
    finally:
        try:
            say("SUMMARY", json.dumps(_hard_marked(STATS), default=str))
            write_summary(
                json.dumps(_hard_marked(STATS), indent=1, default=str),
                complete=True,
                force=True,
            )
        except BaseException as e:  # noqa: BLE001
            write_min_summary(1, f"summary failed: {e!r}")
        finally:
            deadline.cancel()
            FIN["written"].set()


def _rank_dump_compare():
    if os.environ.get("OC_DUMP_COMMIT") == "1" and len(_REF_PIDS) > 1:
        # Do the TP ranks agree on every captured row? Compare each rank's commit dumps.
        import glob as _g

        import torch as _t

        per = {}
        for pid in _REF_PIDS:
            acc = {}
            for f in sorted(
                _g.glob(os.path.join(os.environ["OC_DUMP"], f"c{pid}_*.pt"))
            ):
                for k_, (tr, st, rows) in enumerate(
                    _t.load(f, weights_only=False)["commit"]
                ):
                    acc[(os.path.basename(f).split("_", 1)[1], k_)] = (tr, st, rows)
            per[pid] = acc
        p0 = _REF_PIDS[0]
        diff = same = 0
        for pid in _REF_PIDS[1:]:
            for key, (tr, st, rows) in per[pid].items():
                ref_ = per[p0].get(key)
                if (
                    ref_ is None
                    or ref_[0] != tr
                    or ref_[1] != st
                    or len(ref_[2]) != len(rows)
                ):
                    diff += 1
                    continue
                d = int(
                    np.any(
                        ref_[2].reshape(len(rows), -1) != rows.reshape(len(rows), -1),
                        axis=1,
                    ).sum()
                )
                diff += int(d > 0)
                same += int(d == 0)
        STATS["rank_dump_same"], STATS["rank_dump_diff"] = same, diff
        say("RANK_DUMP_COMPARE", "same", same, "diff", diff, "ranks", len(_REF_PIDS))


FENCES: dict[str, float] = {}
# Fences need the release marker (writers drop late puts) or SSD offload; without both
# the release is v57c's plain removal.
FENCE_ON = SSD or bool(os.environ.get("OC_RELEASE_DIR"))
WRITERS: list[int] = []


WRITERS_EXPECTED = [int(os.environ.get("OC_TP", "4"))]
REPORT_READ_S = float(os.environ.get("OC_REPORT_READ_S", "30"))


def read_reports(timeout_s=None):
    """This run's writer reports, read under a time bound (a stuck filesystem must not
    hang the driver); None when the read did not finish in time."""
    how, r = oc.call_bounded(
        lambda: oc.writer_reports(os.environ["OC_STATS_FILE"], DRIVER_T0),
        REPORT_READ_S if timeout_s is None else timeout_s,
    )
    if how != "ok":
        STATS["report_read_failed"] = STATS.get("report_read_failed", 0) + 1
        return None
    return r


def refresh_writers(reports):
    """Add writers that started reporting since: a late rank joins the set instead of
    being missed. `reports` None (a read that did not finish) adds nothing."""
    if reports is not None:
        WRITERS[:] = sorted(set(WRITERS) | set(reports))
    return len(WRITERS) >= WRITERS_EXPECTED[0]


def register_writers(n_expected, wait_s=None):
    """Wait for exactly the expected writers (one per TP rank). Fences stay closed until
    all of them reported; a set still incomplete at finish fails the run. Every read is
    bounded by the time left."""
    WRITERS_EXPECTED[0] = n_expected
    end = time.time() + (
        float(os.environ.get("OC_WRITER_REGISTER_S", "30"))
        if wait_s is None
        else wait_s
    )
    while True:
        left = end - time.time()
        if (
            refresh_writers(read_reports(max(0.01, min(REPORT_READ_S, left))))
            or left <= 0
        ):
            break
        time.sleep(min(1.0, max(0.0, end - time.time())))
    return list(WRITERS)


SESSION_RIDS: dict[str, list] = {}
KEY_HINTS: dict[str, list] = {}


def leak_probe(limit=30):
    """Before the job-end sweep: the first `limit` keys of known sessions (per-request
    objects of every request, plus the block keys of the warmup and probe manifests)
    that still exist. Bounded by OC_LEAK_PROBE_S. Probes the warmup and probes first."""
    exists = getattr(reader, "is_exist", None)
    found, checked = [], [0]

    def run():
        order = sorted(
            SESSION_RIDS, key=lambda n: ("-warm" not in n and "probe-" not in n, n)
        )
        for name in order:
            keys = list(KEY_HINTS.get(name, []))
            for rid in SESSION_RIDS[name]:
                mk = oc.manifest_key(name, rid)
                keys += [
                    mk,
                    oc.err_key(mk),
                    oc.tail_key(name, rid),
                    oc.idx_tail_key(name, rid),
                    oc.cone_key(name, rid, "r3"),
                    oc.cone_key(name, rid, "idx"),
                ]
                keys += [oc.hid_key(name, rid, i) for i in range(4)]
            for k in keys:
                checked[0] += 1
                hit = exists(k) == 1 if exists is not None else bool(reader.get(k))
                if hit:
                    found.append(k)
                    if len(found) >= limit:
                        return

    how, _ = oc.call_bounded(run, float(os.environ.get("OC_LEAK_PROBE_S", "60")))
    return dict(sample=list(found), checked=checked[0], complete=how == "ok")


def release(name):
    """Release marker, then remove. With fences on, the session stays fenced until every
    expected writer reported nothing pending for it after the marker (settle_fences)."""
    how, n = oc.call_bounded(
        lambda: oc.release_session(reader, name, os.environ.get("OC_RELEASE_DIR")),
        STORE_CALL_S,
    )
    if how != "ok":
        # The marker is down (or not): the fence's second sweep retries the removal.
        STATS.setdefault("release_failed", {})[name] = f"{how}: {n!r}"
        n = None
    if FENCE_ON:
        FENCES[name] = time.time()
    return n


def settle_fences(acc, wait_s=0.0):
    """Second sweep (bounded) of every released session all expected writers
    acknowledged; objects found then were recreated by a late put
    (acc['late_recreated']). Writers still owing an acknowledgement are listed in
    acc['fence_waiting']."""
    if not FENCE_ON:
        return
    end = time.time() + wait_s
    while FENCES:
        reports = read_reports()
        complete = reports is not None and refresh_writers(reports)
        reports = reports or {}
        waiting = {}
        for name, t_rel in list(FENCES.items()):
            w = (
                oc.fence_waiting(reports, WRITERS, name, t_rel)
                if complete
                else [f"writers registered {len(WRITERS)}/{WRITERS_EXPECTED[0]}"]
            )
            if w:
                waiting[name] = w
                continue
            how, late = oc.call_bounded(
                lambda name=name: reader.remove_by_regex(
                    "^" + oc.PREFIX + "/" + name + "/", True
                ),
                STORE_CALL_S,
            )
            if how != "ok":
                acc["fence_sweep_failed"] = acc.get("fence_sweep_failed", 0) + 1
                waiting[name] = [f"sweep {how}"]
                continue
            acc["late_recreated"] = acc.get("late_recreated", 0) + int(late or 0)
            del FENCES[name]
        acc["fence_waiting"] = {k: v for k, v in list(waiting.items())[:20]}
        if not FENCES or time.time() >= end:
            break
        time.sleep(1)
    acc["fences_open"] = len(FENCES)


# ---------------- warmup (untimed) ----------------
PHASE[0] = "warmup"
guard.arm("warmup")
WARM = f"{DESIGN}-{RUN_NONCE}-warm"
_warm = llm.generate(
    [TokensPrompt(prompt_token_ids=text_ids(900 + i, 2000)) for i in range(4)],
    [SamplingParams(max_tokens=16, extra_args={"oc_session": WARM}) for _ in range(4)],
    use_tqdm=False,
)
if reader is not None:
    # Teardown (tail, manifest, ...) of a finished request runs at the next engine step
    # (tick()'s steps, defined further down).
    if os.environ.get("OC_NO_TICK") != "1":
        for _ in range(2):
            llm.generate(
                [TokensPrompt(prompt_token_ids=[1, 2, 3])],
                SamplingParams(max_tokens=1),
                use_tqdm=False,
            )
    STATS["warmup_manifests"] = sum(
        get_manifest(WARM, o.request_id, timeout=120) is not None for o in _warm
    )
    SESSION_RIDS[WARM] = [o.request_id for o in _warm]
guard.disarm()
if reader is not None and FENCE_ON:
    guard.arm("writer registration")
    STATS["writers"] = register_writers(int(os.environ.get("OC_TP", "4")))
    guard.disarm()
if reader is not None:
    guard.arm("warmup release")
    STATS["warmup_removed"] = release(WARM)
    guard.disarm()
say("warmup done")


def _start_pause_resets(ph):
    """OC_PAUSE_RESET_EVERY_S: while generate runs, a thread repeats the in-flight
    weight-update path without changing weights: pause_scheduler("keep",
    clear_cache=True) (the engine drains in-flight batches, then
    reset_prefix_cache(reset_running_requests=True) preempts every running request and
    starts a new aux-output generation), then resume_scheduler."""
    period = float(os.environ.get("OC_PAUSE_RESET_EVERY_S", "0") or 0)
    if period <= 0:
        return lambda: None
    import threading

    stop = threading.Event()
    core = llm.llm_engine.engine_core

    once = os.environ.get("OC_PAUSE_RESET_ONCE") == "1"

    def loop():
        # OC_PAUSE_RESET_ONCE=1: one reset per generate call, `period` seconds in (a
        # reset every few seconds preempts the running requests before they make
        # progress).
        fired = False
        while not stop.wait(period) and not (once and fired):
            fired = True
            try:
                t = time.time()
                core.call_utility("pause_scheduler", "keep", True)
                if PAUSE_PERTURB:
                    # In-flight weight update: new weights while paused, so requests
                    # preempted by the reset resume at the next version (a preemption
                    # across a version change).
                    core.collective_rpc(
                        _perturb_router,
                        None,
                        (WCHANGE_SCALE, 7777 + ph.get("pause_resets", 0)),
                        {},
                    )
                    ph["inflight_weight_changes"] = (
                        ph.get("inflight_weight_changes", 0) + 1
                    )
                core.call_utility("resume_scheduler")
                ph["pause_resets"] = ph.get("pause_resets", 0) + 1
                ph.setdefault("pause_reset_s", []).append(round(time.time() - t, 3))
            except Exception as e:
                ph["pause_reset_errors"] = ph.get("pause_reset_errors", 0) + 1
                say("PAUSE_RESET_ERROR", repr(e)[:300])

    th = threading.Thread(target=loop, daemon=True)
    th.start()

    def finish():
        stop.set()
        th.join(timeout=600)

    return finish


def check_writer_fatal():
    """A writer that stopped retrying (OC_PUT_STALL_S without a successful put: disk or
    pool failure) reports `fatal`: abort the run with that reason instead of waiting on
    it."""
    if FINISHING.is_set():
        return
    reports = read_reports(10)
    fatal = {pid: r["fatal"] for pid, r in (reports or {}).items() if r.get("fatal")}
    if fatal:
        STATS["writer_fatal"] = fatal
        try:
            say("WRITER_FATAL", json.dumps(fatal))
            finish(deadlock=True)
        finally:
            hard_exit(12, "put_stall")


def check_helpers():
    """Abort with context when the master or the holder died: a dead holder would
    otherwise go unnoticed for the client lease (600 s) and take the pool's objects with
    it."""
    if FINISHING.is_set():
        return
    dead = oc.dead_procs(procs)
    if dead and FINISHING.is_set():  # our own finish is stopping them
        return
    if dead:
        for d in dead:
            log = OUT + (".master.log" if d["i"] == 0 else ".holder.log")
            with contextlib.suppress(OSError):
                d["log_tail"] = pathlib.Path(log).read_text()[-2000:]
        STATS["helper_died"] = dead
        try:
            say("HELPER_DIED", json.dumps(dead, default=str))
            finish(deadlock=True)
        finally:
            hard_exit(10, "helper died")


def tick():
    check_helpers()
    if os.environ.get("OC_NO_TICK") == "1":  # engine-native finish: no extra requests
        return
    for _ in range(2):
        llm.generate(
            [TokensPrompt(prompt_token_ids=[1, 2, 3])],
            SamplingParams(max_tokens=1),
            use_tqdm=False,
        )


WCHANGE_AFTER = int(os.environ.get("OC_WEIGHT_CHANGE_AFTER_STEP", "0"))
PAUSE_PERTURB = os.environ.get("OC_PAUSE_PERTURB") == "1"
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


def track_of(s):
    """The aux trajectory a request of `s` is captured under (session, or session +
    trajectory id)."""
    tagged = (TRAJ_TAG and GROUP > 1) or s.get("compact")
    return s["name"] + ("" + str(s["traj"]) if tagged else "")


def compact_prompt(sys_ids, summary_ids, chunks):
    """A continuation segment's prompt: the fixed system prompt, the summary, the
    last 2 turns."""
    tail = [t for c in chunks[-2:] for t in c]
    return list(sys_ids) + list(summary_ids) + tail


def compact_sess_bytes(threshold, n_compactions, bpt, extra_hidden, maxtok, obs):
    """Pool bytes of one compaction rollout: every segment holds up to `threshold`
    positions (summary included), rows of shared system-prompt blocks counted per
    segment (an upper bound); hidden rows of each segment's requests."""
    segments = n_compactions + 1
    est = segments * threshold * bpt * 1.1
    if extra_hidden:
        est += segments * (threshold // max(1, maxtok + obs) + 2) * maxtok * 5124
    return est


def compact_init(s, sys_ids, task_ids):
    s.update(
        compact=True,
        ids=list(sys_ids) + list(task_ids),
        traj=0,
        mode="turns",
        chunks=[],
        segments=[],
        compactions=0,
        summary=None,
    )


def compact_advance(s, gen, obs, summarize_ids):
    """After one output of a compaction session (its turn counter not yet advanced).
    Returns 'turn' (next turn of the segment), 'summary' (the next request is the
    summary), 'new_segment' (the summary came back) or 'done' (the last segment reached
    the threshold)."""
    if s["mode"] == "summary":
        s["summary"] = list(gen)
        return "new_segment"
    s["chunks"].append(list(gen) + list(obs))
    s["ids"] = list(s["ids"]) + list(gen) + list(obs)
    if len(s["ids"]) >= COMPACT_AT or s["turn"] + 1 >= COMPACT_MAX_TURNS:
        if s["compactions"] < COMPACT_N:
            s["mode"] = "summary"
            s["ids"] = s["ids"] + list(summarize_ids)
            return "summary"
        return "done"
    return "turn"


def compact_archive(s):
    """The finished segment, as a trajectory record the checker reads like any other."""
    s["segments"].append(
        dict(
            name=s["name"],
            traj=s["traj"],
            reqs=s["reqs"],
            compact=True,
            i=s["i"],
            g=s["g"],
            v0=s.get("v0", 0),
            ids=list(s["ids"]),
        )
    )


def compact_new_segment(s, sys_ids):
    compact_archive(s)
    s["ids"] = compact_prompt(sys_ids, s["summary"], s["chunks"])
    s.update(
        traj=s["traj"] + 1,
        reqs=[],
        rows=None,
        chunks=[],
        mode="turns",
        turn=0,
        compactions=s["compactions"] + 1,
        summary=None,
    )
    for k in ("prev_rows", "prev_rows_before", "prev_missing", "mrows", "skip_pos"):
        s.pop(k, None)


def gen_len(s):
    if s.get("mode") == "summary":
        return SUMMARY_TOK
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
    name = f"{REP}probe-{RUN_NONCE}-p{len(STATS['phases'])}-{tag}"
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
    SESSION_RIDS[name] = [o.request_id]
    if m is not None:
        KEY_HINTS[name] = [oc.block_key(name, h) for h in m.get("blocks", [])]
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
    STATS.setdefault("probe_removed", {})[name] = release(name)
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


def read_rows(name, rid):
    """(manifest, delivered R3 rows) of one request, or (manifest or None, None)."""
    m = get_manifest(name, rid, timeout=60)
    if m is None:
        return None, None
    parts = get_many([oc.block_key(name, h) for h in m["blocks"]])
    if any(p is None for p in parts):
        return m, None
    tail = (get_many([oc.tail_key(name, rid)])[0] or b"")[: m["tail_nbytes"]]
    got = np.frombuffer(b"".join(parts) + tail, dtype=m["dtype"]).reshape(
        (-1, *m["shape"])
    )[: m["rows"]]
    return m, got


def compact_shared_blocks(manifests):
    """Block objects a rollout's segments list (each segment's last manifest: no
    sharing) vs the unique objects they name (written)."""
    listed = sum(len(m["blocks"]) for m in manifests)
    unique = len({b for m in manifests for b in m["blocks"]})
    return listed, unique


def compact_prev_served(got, own, prev, lo):
    """Positions >= lo where a segment was delivered the previous segment's row although
    its own computation there differs: rows served at a shifted position from the wrong
    trajectory."""
    n = min(len(got), len(own), len(prev))
    if n <= lo:
        return 0
    g, o, p_ = (x[lo:n].reshape(n - lo, -1) for x in (got, own, prev))
    return int(((g == p_).all(1) & (o != p_).any(1)).sum())


def compact_check(s, ph):
    """Per rollout: sharing of the system-prompt blocks across segments, no
    previous-segment rows served at a continuation's positions."""
    last = []
    prev = None
    for seg in s["segments"]:
        rec = seg["reqs"][-1]
        m, got = read_rows(s["name"], rec[0])
        if m is None:
            continue
        last.append(m)
        bs = int(m.get("block_size", 1))
        if prev is not None and got is not None:
            # The continuation's own rows are its inline rows; the previous segment's
            # from its last request.
            lo = (len(SYS_IDS) // bs) * bs
            n_ = compact_prev_served(got, rec[2], prev, lo)
            ph["compact_prev_served"] = ph.get("compact_prev_served", 0) + n_
            ph["compact_cont_rows_checked"] = ph.get(
                "compact_cont_rows_checked", 0
            ) + max(0, min(len(got), len(prev)) - lo)
        prev = rec[2]
    if last:
        listed, unique = compact_shared_blocks(last)
        bs = int(last[0].get("block_size", 1))
        ph["compact_blocks_listed"] = ph.get("compact_blocks_listed", 0) + listed
        ph["compact_blocks_unique"] = ph.get("compact_blocks_unique", 0) + unique
        ph["compact_blocks_shared"] = (
            ph.get("compact_blocks_shared", 0) + listed - unique
        )
        ph["compact_sys_blocks_expected_shared"] = ph.get(
            "compact_sys_blocks_expected_shared", 0
        ) + (len(last) - 1) * (len(SYS_IDS) // bs)
    ph["compact_segments"] = ph.get("compact_segments", 0) + len(s["segments"])
    ph["compact_rollouts"] = ph.get("compact_rollouts", 0) + 1


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
    sess_bytes = L * bpt * 1.1 + (
        TURNS * MAXTOK * (5124 + (8 * TOPK_K if TOPK else 0)) if EXTRA else 0
    )
    if COMPACT:
        # One rollout: segments x threshold positions (see compact_sess_bytes).
        sess_bytes = compact_sess_bytes(
            COMPACT_AT, COMPACT_N, bpt, EXTRA, MAXTOK, COMPACT_OBS
        )
        SYS_IDS = text_ids(424243, SYS_TOK)
        SUMMARIZE_IDS = tok.encode(
            "\n\nContext limit reached. Summarize the task, what was tried, what was "
            "learned and what remains, for a fresh context:\n",
            add_special_tokens=False,
        )
    C = (
        S
        if ARM == "off"
        else oc.admission_c(
            S,
            sess_bytes,
            GROUP,
            POOL,
            C_MULT,
            os.environ["OC_MC_SSD_PATH"] if SSD else None,
        )
    )
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
        group=GROUP,
        traj_tag=TRAJ_TAG,
        kv_blocks=os.environ.get("OC_KV_BLOCKS"),
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
    group_left = {}
    inflight = []
    while pending or inflight:
        if reader is not None:
            settle_fences(ph)
        while pending and len(inflight) < C * GROUP:
            if (
                SSD
                and oc.ssd_headroom(os.environ["OC_MC_SSD_PATH"], POOL)
                - sum(RESERVED.values())
                < sess_bytes * GROUP
            ):
                # Free offload space below one group plus the reserve: start no new
                # session.
                ph["admission_paused"] = ph.get("admission_paused", 0) + 1
                if not inflight:
                    raise RuntimeError(
                        "SSD offload headroom below one session group with nothing "
                        "in flight"
                    )
                break
            g = pending.pop(0)
            if SSD:
                # Bytes this group will write, held until its release.
                RESERVED[(L, g)] = sess_bytes * GROUP
            ids0 = text_ids(L * 100 + g, P0)
            group_left[g] = GROUP
            for j in range(GROUP):
                inflight.append(
                    dict(
                        v0=len(ph.get("weight_changes", [])),
                        name=f"{REP}{DESIGN}-{RUN_NONCE}-p{len(STATS['phases'])}-L{L}-{ARM}-{g}",
                        ids=list(ids0),
                        rows=None,
                        turn=0,
                        reqs=[],
                        i=g * GROUP + j,
                        g=g,
                        traj=j,
                        fork_wait=FORK and j > 0,
                    )
                )
                if COMPACT:
                    compact_init(
                        inflight[-1], SYS_IDS, text_ids(L * 100 + g, COMPACT_TASK_TOK)
                    )
        prompts, params = [], []
        active = [s for s in inflight if not s.get("fork_wait")]
        for s in active:
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
                    logprobs=int(os.environ.get("OC_LOGPROBS", "0")),
                    **({} if NOR3 else {"routed_experts_prompt_start": aux}),
                    extra_args={
                        "oc_session": s["name"],
                        **(
                            {"oc_traj": str(s["traj"])}
                            if (TRAJ_TAG and GROUP > 1) or COMPACT
                            else {}
                        ),
                        **({"oc_parent": s.pop("parent")} if "parent" in s else {}),
                    },
                )
            )
            ph["prefill_tokens"] += len(s["ids"]) - aux
        ctx = max(len(s["ids"]) for s in active)
        guard.arm(f"L{L} generate ctx={ctx} inflight={len(active)}")
        t = time.time()
        reset_stop = _start_pause_resets(ph)
        outs = llm.generate(prompts, params, use_tqdm=False)
        reset_stop()
        dt = time.time() - t
        tick()
        guard.disarm()
        ph["step_s"].append(round(dt, 2))
        ph["step_ctx"].append(ctx)
        done = []
        fork_inline = {}
        for s, o in zip(active, outs, strict=True):
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
            SESSION_RIDS.setdefault(s["name"], []).append(o.request_id)
            if FORK and s.get("k0", 0) + len(s["reqs"]) == 2:
                # Trajectory 0's second request and each fork's first send the same
                # prompt in the same step: compare what each computed for the new prompt
                # positions.
                key = (s["g"], tuple(s["ids"][: len(s["ids"])]))
                a0, n0 = s["aux"], len(s["ids"]) - 1
                mine = rows[a0:n0]
                ref = fork_inline.setdefault(key, (s["traj"], mine))
                if ref[0] != s["traj"] and not FORK_CUT:
                    k = min(len(ref[1]), len(mine))
                    d = int(
                        np.any(
                            ref[1][:k].reshape(k, -1) != mine[:k].reshape(k, -1), axis=1
                        ).sum()
                    )
                    ph["fork_same_prompt_rows"] = ph.get("fork_same_prompt_rows", 0) + k
                    ph["fork_same_prompt_diff"] = ph.get("fork_same_prompt_diff", 0) + d
                    if d:
                        say(
                            "FORK_DIFF", s["g"], ref[0], s["traj"], "rows", k, "diff", d
                        )
            s["rows"] = rows
            if not NOR3 and len(suffix):
                ph["r3_inline_sentinel_rows"] = ph.get(
                    "r3_inline_sentinel_rows", 0
                ) + int(
                    (suffix.reshape(len(suffix), -1) == np.iinfo(suffix.dtype).max)
                    .any(axis=1)
                    .sum()
                )
            if os.environ.get("OC_R3_SAVE") and s["turn"] + 1 >= TURNS:
                import hashlib

                dg = ph.setdefault("digests", {})
                dg[str(s["i"])] = [
                    hashlib.md5(np.ascontiguousarray(rows).tobytes()).hexdigest(),
                    hashlib.md5(
                        np.asarray(list(s["ids"]) + gen, np.int64).tobytes()
                    ).hexdigest(),
                ]
                if s["i"] < int(os.environ["OC_R3_SAVE"]):
                    np.savez(
                        os.environ["OUT"] + f".rows{s['i']}.npz",
                        rows=rows,
                        ids=np.asarray(list(s["ids"]) + gen, np.int64),
                    )
            ph["gen_tokens"] += len(gen)
            if COMPACT:
                obs = (
                    text_ids(
                        L * 100000 + s["i"] * 1000 + s["traj"] * 100 + s["turn"],
                        COMPACT_OBS,
                    )
                    if s["mode"] == "turns"
                    else []
                )
                act = compact_advance(s, gen, obs, SUMMARIZE_IDS)
                s["turn"] += 1
                ph["compact_summaries"] = ph.get("compact_summaries", 0) + (
                    act == "new_segment"
                )
                if act == "new_segment":
                    compact_new_segment(s, SYS_IDS)
                elif act == "done":
                    compact_archive(s)
                    done.append(s)
                continue
            obs = (
                text_ids(L * 1000 + s["i"] * 10 + s["turn"], max(16, D - len(gen)))
                if s["turn"] + 1 < TURNS
                else []
            )
            s["ids"] = list(s["ids"]) + gen + obs
            s["turn"] += 1
            if s["turn"] >= TURNS:
                done.append(s)
            if FORK and s["traj"] == 0 and s["turn"] == 1:
                s["fork_snap"] = (list(s["ids"]), np.array(s["rows"], copy=True))
            if FORK and s["traj"] == 0 and s["turn"] == 1 + FORK_DELAY:
                snap_ids, snap_rows = s.pop("fork_snap")
                if FORK_CUT:
                    n_ = int(len(s["rows"]) * FORK_CUT)
                    snap_ids, snap_rows = (
                        list(s["ids"][: n_ + 1]),
                        np.array(s["rows"][:n_], copy=True),
                    )
                for f in inflight:
                    if f.get("fork_wait") and f["g"] == s["g"]:
                        f.update(
                            ids=list(snap_ids),
                            rows=np.array(snap_rows, copy=True),
                            turn=1,
                            fork_wait=False,
                            k0=1,
                            v0=s["v0"],
                            parent_rid0=s["reqs"][0][0],
                            parent_ends=[
                                (r_[5], min(r_[1], len(snap_rows)))
                                for r_ in s["reqs"]
                                if r_[5] <= len(snap_rows)
                            ],
                        )
                        if FORK_PARENT:
                            f["parent"] = "0"
                        ph["forks"] = ph.get("forks", 0) + 1
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
            segs = s["segments"] if COMPACT else [s]
            ph["unique_rows"] += sum(seg["reqs"][-1][1] for seg in segs)
            if ARM != "off":
                guard.arm(f"L{L} trainer read {s['name']}")
                tr = time.time()
                for seg in segs:
                    for j, rec in enumerate(seg["reqs"]):
                        ph["requests"] += 1
                        check_request(
                            seg,
                            rec,
                            ph,
                            full=not READ_LAST_ONLY or j == len(seg["reqs"]) - 1,
                        )
                if COMPACT:
                    compact_check(s, ph)
                ph["read_s"] += time.time() - tr
                group_left[s["g"]] -= 1
                if group_left[s["g"]] == 0:
                    tr = time.time()
                    ph.setdefault("release_removed", {})[s["name"]] = release(s["name"])
                    RESERVED.pop((L, s["g"]), None)
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
        if COMPACT and ph.get("compact_rollouts"):
            ph["compact_bytes_est_per_session"] = int(sess_bytes)
            ph["compact_bytes_written_per_session"] = int(
                sum(ph[f"bytes_{k}"] for k in ("r3", "idx", "hid", "manifest"))
                / ph["compact_rollouts"]
            )
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
                for k in (
                    "prefix_cache",
                    "offload",
                    "external",
                    "kv_transfer",
                    "preempt",
                )
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
