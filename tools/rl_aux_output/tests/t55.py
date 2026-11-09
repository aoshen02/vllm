# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""D20: the oc-apply thread must deliver what the inline (lock-held) path delivers, byte
for byte.

A fake engine drives begin_step / prepare_output / process_output of the real worker
connector (real Writer over an in-memory Mooncake, real base buffer and store from
../fixtures) with the oc-apply thread lagging 0, 1 or all events behind the producers,
and compares every result with OC_APPLY_INLINE=1. Run: python -I t55.py"""

import collections
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import threading
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "pylib"))
os.environ.update(
    OC_MC_MASTER="fake",
    OC_R3_PARTIAL="1",
    OC_EXTRA="1",
    OC_HARD_PIN="0",
    OC_IDX_MODE="nod2h",
    OC_PREFIX="oc/t55",
)

import numpy as np  # noqa: E402
import zstandard  # noqa: E402,F401  (encoding paths must run, not log errors)

BS = 16
SHAPE = (40, 2)
IDX_LAYERS = [0, 10, 20, 30]
K = 4
D = 8


# ------------------------------------------------------------ module stubs


def _mod(name, **attrs):
    m = sys.modules.get(name) or types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    parent, _, leaf = name.rpartition(".")
    if parent:
        setattr(_mod(parent), leaf, m)
    return m


class FakeMooncake:
    def __init__(self):
        self.objs, self.order = {}, []

    def setup(self, *a):
        return 0

    def put(self, key, value, cfg=None):
        self.objs[key] = bytes(value)
        self.order.append(key)
        return 0

    def get(self, key):
        return self.objs.get(key)

    def is_exist(self, key):
        return int(key in self.objs)


MC: list = []


def _make_mc():
    MC.append(FakeMooncake())
    return MC[-1]


_mod("mooncake.store", MooncakeDistributedStore=_make_mc, ReplicateConfig=object)
try:
    import torch  # noqa: F401
except ImportError:
    _mod("torch", nn=types.SimpleNamespace(Module=object), Tensor=object, save=None)

import oc_common as oc  # noqa: E402

_mod("vllm.logger", init_logger=lambda n: types.SimpleNamespace(warning=print))
_mod("vllm.config", VllmConfig=object)
_mod("vllm.distributed.aux_output_connector", oc_common=oc)
sys.modules["vllm.distributed.aux_output_connector.oc_common"] = oc


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    setattr(sys.modules[name.rpartition(".")[0]], name.rpartition(".")[2], m)
    return m


store_mod = _load(
    "vllm.distributed.aux_output_connector.store",
    os.path.join(ROOT, "fixtures", "store.py"),
)
_load(
    "vllm.distributed.aux_output_connector.routed_experts",
    os.path.join(ROOT, "fixtures", "routed_experts.py"),
)
conn_mod = _load(
    "vllm.distributed.aux_output_connector.connector",
    os.path.join(HERE, "connector_fix.py"),
)
_mod(
    "vllm.distributed.parallel_state",
    get_tp_group=lambda: types.SimpleNamespace(is_first_rank=True),
)


class Capturer:
    def __init__(self, **kw):
        self.shape_per_token, self.output_dtype_name = SHAPE, "uint16"

    def snapshot_routing_data(self, n):
        return None


_mod(
    "vllm.model_executor.layers.fused_moe.routed_experts_capturer",
    RoutedExpertsCapturer=Capturer,
    bind_routed_experts_capturer=lambda m, c: None,
)
_mod("vllm.v1.core.kv_cache_utils", resolve_kv_cache_block_sizes=lambda k, v: (BS, BS))
wa = _load(
    "vllm.distributed.aux_output_connector.worker_attr",
    os.environ.get("T55_WORKER") or os.path.join(HERE, "worker_attr.py"),
)
Meta = conn_mod.AuxOutputConnectorMetadata
oc.setup_extra = lambda model, n: None
oc.IDX_CAP = types.SimpleNamespace(
    layer_ids=IDX_LAYERS, shape=[len(IDX_LAYERS), K], topk=K, trimmed=False
)
CFG = types.SimpleNamespace(
    scheduler_config=types.SimpleNamespace(max_num_batched_tokens=512, max_num_seqs=8),
    max_concurrent_batches=2,
    aux_output_config=types.SimpleNamespace(max_bytes=64 << 20),
)


# ------------------------------------------------------------ base store with a
# recorder


class RecordingStore(store_mod.BlockObjectStore):
    """Base store that logs every mutation and read; evicting=True drops an object the
    moment its last reference is released (the worst a full store can do)."""

    evicting = False

    def __init__(self, **kw):
        super().__init__(**kw)
        self.log = []

    def put(self, objects, *, retain_keys=(), release_keys=()):
        retain_keys, release_keys = tuple(retain_keys), tuple(release_keys)
        self.log.append(
            ("put", tuple(o.key for o in objects), retain_keys, release_keys)
        )
        super().put(objects, retain_keys=retain_keys, release_keys=release_keys)
        if self.evicting:
            for k in release_keys:
                if k not in self._references and k in self._slots:
                    self._lru.pop(k, None)
                    slot = self._slots.pop(k)
                    if slot != self._UNALLOCATED_SLOT:
                        self._free_slots.append(slot)

    def get_concatenated(self, keys):
        try:
            out = super().get_concatenated(keys)
        except store_mod.BlockObjectStoreError:
            self.log.append(("get_fail", tuple(keys)))
            raise
        self.log.append(("get", tuple(keys)))
        return out


store_mod.BlockObjectStore = RecordingStore
wa.BlockObjectStore = RecordingStore


# ------------------------------------------------------------ deterministic lag


class Gate:
    """Queue for the oc-apply thread: event i is handed out only once event i + lag was
    posted (or a drain was posted). lag=0 also makes every post wait until its event was
    applied."""

    def __init__(self, lag):
        self.lag, self.items, self.cv = lag, collections.deque(), threading.Condition()
        self.posted = self.taken = self.done = self.flush_to = 0
        self.busy = False

    def put(self, item):
        with self.cv:
            self.items.append(item)
            self.posted += 1
            if item is None or isinstance(
                getattr(item[0], "__self__", None), threading.Event
            ):
                self.flush_to = self.posted
            self.cv.notify_all()
            if self.lag == 0 and item is not None:
                n = self.posted
                self.cv.wait_for(lambda: self.done >= n)

    def get(self):
        with self.cv:
            if self.busy:
                self.done += 1
                self.busy = False
                self.cv.notify_all()
            self.cv.wait_for(
                lambda: self.items
                and (self.taken < self.flush_to or self.posted - self.taken > self.lag)
            )
            self.taken += 1
            self.busy = True
            return self.items.popleft()

    def qsize(self):
        return len(self.items)


# ------------------------------------------------------------ fake engine


def _hash(tokens):
    return hashlib.blake2b(
        np.asarray(tokens, np.int64).tobytes(), digest_size=8
    ).digest()


class Req:
    def __init__(
        self,
        rid,
        sess,
        tokens,
        prompt_len,
        aux=0,
        chunk=24,
        hit=0,
        hash_delay=0,
        keys_from=0,
    ):
        self.rid, self.sess, self.tokens = rid, sess, list(tokens)
        self.prompt_len, self.aux, self.chunk = prompt_len, aux, chunk
        self.computed, self.ntok, self.sent = hit, prompt_len, 0
        self.need_session, self.hash_delay, self.hist, self.keys_from = (
            True,
            hash_delay,
            [],
            keys_from,
        )


class Sim:
    def __init__(self, mode, evicting=False, conn_patch=None):
        oc._writer = None
        oc.WEIGHT_VERSION[0] = 0
        RecordingStore.evicting = evicting
        if mode == "inline":
            os.environ["OC_APPLY_INLINE"] = "1"
            wa.queue = __import__("queue")
        else:
            os.environ.pop("OC_APPLY_INLINE", None)
            wa.queue = (
                __import__("queue")
                if mode == "real"
                else types.SimpleNamespace(
                    SimpleQueue=lambda: Gate({"all": 1 << 60}.get(mode, mode))
                )
            )
        self.conn = wa.AuxOutputWorkerConnector(
            vllm_config=CFG, model=None, kv_cache_config=None
        )
        assert self.conn._oc_inline == (mode == "inline")
        if conn_patch:
            conn_patch(self.conn)
        self.store = self.conn._store._store._store
        self.w = oc.writer()
        self.mc = MC[-1]
        self.gen, self.reqs, self.inflight, self.outputs, self.n = 0, {}, [], [], 0
        self.finished, self.preempted, self.parked = [], [], []

    def add(self, r):
        self.reqs[r.rid] = r
        return r

    def _hashes(self, r, upto_tokens):
        n = upto_tokens // BS
        new = [_hash(r.tokens[: (b + 1) * BS]) for b in range(r.sent, n)]
        r.sent = max(r.sent, n)
        return new

    def finish(self, rid, parked=False, aborted=False):
        r = self.reqs.pop(rid)
        self.finished.append(
            (
                rid,
                []
                if aborted
                else self._hashes(r, r.ntok)
                if not parked
                else [_hash(r.tokens[: (b + 1) * BS]) for b in range(r.ntok // BS)],
            )
        )
        if parked:
            self.parked.append(rid)

    def preempt(self, rid, hit):
        r = self.reqs[rid]
        self.preempted.append(rid)
        r.sent, r.computed, r.need_session = 0, hit, True

    def step(self, sched, gen=False, version=None):
        if version is not None:
            oc.WEIGHT_VERSION[0] = version
        if gen:
            self.gen += 1
            for r in self.reqs.values():
                r.sent = 0
        requests, hashes, sessions, batch = {}, {}, {}, []
        for rid in sched:
            r = self.reqs[rid]
            r.hist.append(r.ntok)
            known = r.hist[max(0, len(r.hist) - 1 - r.hash_delay)]
            h = self._hashes(r, known) if len(r.hist) > r.keys_from else []
            if h:
                hashes[rid] = h
            if r.need_session:
                sessions[rid], r.need_session = r.sess, False
            outputs = r.ntok > r.prompt_len
            requests[rid] = max(min(r.aux, r.prompt_len), r.ntok - 1 if outputs else 0)
            # Compute every known token, then sample the next one (a resumed request
            # recomputes its outputs too).
            ts = r.computed
            n = min(r.chunk, r.ntok - ts)
            sampled = int(ts + n == r.ntok)
            r.computed += n
            r.ntok += sampled
            batch.append((r, ts, n, sampled))
        for rid, h in self.finished:
            if h:
                hashes[rid] = h
        meta = Meta(
            self.gen,
            requests,
            hashes,
            tuple(rid for rid, _ in self.finished),
            sessions or None,
            tuple(self.preempted),
            tuple(self.parked),
        )
        self.finished, self.preempted, self.parked = [], [], []
        self.conn.begin_step(meta)
        if not batch:
            return
        qsl = np.cumsum([0] + [n for _, _, n, _ in batch])
        ib = types.SimpleNamespace(
            req_ids=[r.rid for r, *_ in batch],
            query_start_loc_np=qsl,
            num_computed_tokens_np=np.array([ts for _, ts, _, _ in batch]),
        )
        p = self.conn.prepare_output(ib)
        p.routed_experts = np.concatenate([r3_rows(r, ts, n) for r, ts, n, _ in batch])
        p.num_sampled = np.array([s for *_, s in batch])
        p.num_rejected = np.zeros(len(batch), np.int64)
        p.oc_idx = np.concatenate([idx_rows(r, ts, n) for r, ts, n, _ in batch])
        p.oc_hid_cu = np.arange(len(batch) + 1)
        # As _oc_prepare's raw snapshot after the D2H copy: the rows' bytes, dtype,
        # width. "float16" / "bfloat16": two-byte rows of the same width (bfloat16
        # labelled f16 bits).
        dt = {"raw": np.float32}.get(HID[0], np.float16)
        with np.errstate(over="ignore"):
            p.oc_hq = np.stack(
                [hid_row(r, ts + n - 1).astype(dt) for r, ts, n, _ in batch]
            ).view(np.uint8)
        p.oc_hid_dtype, p.oc_hid_d = {"raw": "float32"}.get(HID[0], HID[0]), D
        self.inflight.append(p)
        if len(self.inflight) == 2:
            self.proc()

    def proc(self):
        p = self.inflight.pop(0)
        self.n += 1
        out = p.process_output()
        for rid in sorted(out):
            o = out[rid]
            self.outputs.append(
                (
                    self.n,
                    rid,
                    int(o.token_start),
                    np.ascontiguousarray(o.rows).tobytes(),
                )
            )

    def drain(self):
        while self.inflight:
            self.proc()

    def close(self):
        self.drain()
        self.conn._oc_drain()
        self.stash_left = sorted(self.conn._oc_preempted)
        self.conn.close()
        return result(self)


HID = ["raw"]


def hid_row(r, pos):
    """The lm_head input of the row sampled at pos (float32, every bit pattern
    meaningful)."""
    rng = np.random.default_rng(
        [pos, len(r.rid), r.tokens[pos] if pos < len(r.tokens) else 0]
    )
    return (rng.standard_normal(D) * 10.0 ** rng.integers(-30, 30, D)).astype(
        np.float32
    )


def check_raw_hidden(mc):
    """Every raw hidden field: codec raw, float32 [D], chunks of rows * D * 4 bytes,
    no scales."""
    n = 0
    for k, v in mc.objs.items():
        if not k.endswith("/m"):
            continue
        fh = {f["name"]: f for f in json.loads(v)["fields"]}.get("hidden")
        if fh is None:
            continue
        assert (
            fh["codec"] == "raw" and fh["dtype"] == "float32" and fh["shape"] == [D]
        ), fh
        assert "scale_dtype" not in fh, fh
        bufs = [mc.objs[fh["key_prefix"] + str(i)] for i in range(fh["chunks"])]
        assert [len(b) for b in bufs] == [c * D * 4 for c in fh["chunk_lens"]], fh
        assert fh["nbytes"] == fh["rows"] * D * 4, fh
        n += fh["rows"]
    assert n > 0
    return n


def r3_rows(r, ts, n):
    pos = np.arange(ts, ts + n)
    tok = np.array(r.tokens)[pos]
    base = tok * 131 + pos * 7 + (ts % 5) * 13 + oc.WEIGHT_VERSION[0] * 1000
    v = (
        base[:, None, None]
        + np.arange(40)[None, :, None] * 17
        + np.arange(2)[None, None, :]
    )
    return (v % 60000 + 1).astype(np.uint16)


def idx_rows(r, ts, n):
    pos = np.arange(ts, ts + n)
    tok = np.array(r.tokens)[pos]
    v = (
        (tok * 3 + pos + (ts % 5))[:, None, None]
        + np.arange(len(IDX_LAYERS))[None, :, None] * 5
        + np.arange(K) * 2
    )
    v = np.sort(v % 5000, axis=-1).astype(np.int32)
    v[:, :, -1][pos % 3 == 0] = -1
    return v


def _canon(x):
    if isinstance(x, np.ndarray):
        return [
            "nd",
            str(x.dtype),
            list(x.shape),
            hashlib.blake2b(np.ascontiguousarray(x).tobytes()).hexdigest(),
        ]
    if isinstance(x, dict):
        return sorted(([_canon(k), _canon(v)] for k, v in x.items()), key=repr)
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


_TIMING = ("t_", "put_", "enc_", "blocked", "oc_apply_", "last_put_rc", "batch_")


def result(sim):
    objs = {}
    for k, v in sim.mc.objs.items():
        if k.endswith("/m"):
            m = json.loads(v)
            m.pop("writer_stats", None)
            v = json.dumps(m, sort_keys=True).encode()
        objs[k] = hashlib.blake2b(v).hexdigest()
    w = sim.w
    tables = {
        name: _canon(getattr(w, name, None))
        for name in (
            "last_tail",
            "last_tail_idx",
            "weight_versions",
            "block_segs",
            "track_alt",
            "tail_hist",
            "cone_list",
            "cone_held",
            "cone_ends",
            "cone_blocks",
        )
    }
    stats = {
        k: v
        for k, v in w.stats.items()
        if not k.startswith(_TIMING) and not k.endswith(("_s", "_ms"))
    }
    # Manifests embed a writer-stats snapshot (compared without it above).
    stats["bytes"] -= stats.pop("bytes_manifest", 0)
    released = collections.Counter(
        k for e in sim.store.log if e[0] == "put" for k in e[3]
    )
    retained = collections.Counter(
        k for e in sim.store.log if e[0] == "put" for k in e[2]
    )
    return dict(
        outputs=sim.outputs,
        objs=objs,
        order=list(sim.mc.order),
        tables=tables,
        stats=stats,
        pinned=retained - released,
        refs=dict(sim.store._references),
        get_fail=[e for e in sim.store.log if e[0] == "get_fail"],
        preempted=sim.stash_left,
        closing=dict(sim.conn._oc_track_closing),
        fallback_reads=getattr(sim.conn, "_oc_fallback_reads", 0),
    )


def compare(a, b, name):
    for key in a:
        if a[key] != b[key]:
            if key == "objs":
                diff = sorted(set(a[key].items()) ^ set(b[key].items()))[:6]
                raise AssertionError(f"{name}: objs differ {diff}")
            raise AssertionError(
                f"{name}: {key} differs\n inline={str(a[key])[:600]}\n other "
                f"={str(b[key])[:600]}"
            )


MODES = (0, 1, "all")


def check(case, **kw):
    ref = case(Sim("inline", **kw))
    for mode in MODES:
        compare(ref, case(Sim(mode, **kw)), f"{case.__name__}/lag={mode}")
    print(
        f"{case.__name__}: {len(ref['objs'])} objects, {len(ref['outputs'])} outputs, "
        f"identical at lag 0/1/all"
    )
    return ref


# ------------------------------------------------------------ cases


def toks(seed, n, prefix=()):
    rng = np.random.default_rng(seed)
    return list(prefix) + [int(x) for x in rng.integers(1, 400, n - len(prefix))]


def case_preempt_resume(s):
    t = toks(1, 200)
    s.add(Req("a", "s1", t, prompt_len=40, chunk=20, hash_delay=2))
    for _ in range(10):
        s.step(["a"])
    # Preempted in begin_step(k+1) while commit k (completing a block whose hash lags)
    # is in flight; resumed three steps later from a prefix hit below the rows it had
    # computed.
    s.preempt("a", hit=32)
    s.step([])
    s.drain()
    s.step([], version=1)
    s.step([])
    for _ in range(12):
        s.step(["a"])
    n = s.reqs["a"].ntok
    s.finish("a")
    s.step([])
    # Next turn of the session: splice from the previous tail.
    s.add(Req("b", "s1", t, prompt_len=n + 6, aux=n - 1, hit=(n - 1) // BS * BS))
    for _ in range(10):
        s.step(["b"])
    s.finish("b")
    s.step([])
    return s.close()


def case_parked_empty_stash(s):
    s.add(Req("p", "s2", toks(3, 100), prompt_len=31, chunk=16))
    s.add(Req("q", "s2b", toks(4, 100), prompt_len=40, chunk=16))
    for _ in range(3):
        s.step(["p", "q"])
    # p's last computed row is 31: its capture ends on a block boundary with every hash
    # delivered, so its preemption stashes nothing.
    assert s.reqs["p"].computed == 32 and s.reqs["p"].ntok == 33
    s.preempt("p", hit=16)
    s.step(["q"])
    s.drain()
    # The output in flight at the preemption ended the request: finish it from the empty
    # stash.
    s.finish("p", parked=True)
    for _ in range(8):
        s.step(["q"])
    s.finish("q")
    s.step([])
    return s.close()


def case_generation(s):
    s.add(Req("a", "s3", toks(5, 200), prompt_len=40, chunk=16))
    s.add(Req("b", "s4", toks(6, 200), prompt_len=30, chunk=16))
    s.add(Req("c", "s5", toks(7, 200), prompt_len=20, chunk=16))
    for _ in range(6):
        s.step(["a", "b", "c"])
    s.drain()
    # Reset with c finished before it and a, b running: c finishes in the old
    # generation, a and b are stashed by the reset and recomputed from scratch in the
    # new one.
    s.finish("c")
    for rid in ("a", "b"):
        s.preempt(rid, hit=0)
    s.step(["a", "b"], gen=True, version=2)
    for _ in range(8):
        s.step(["a", "b"])
    s.finish("a")
    s.step(["b"])
    s.finish("b")
    s.step([])
    return s.close()


def case_fork_parent_queued(s):
    t = toks(8, 200)
    s.add(Req("p", "s6", t, prompt_len=40, chunk=16, hash_delay=2))
    for _ in range(14):
        s.step(["p"])
    # p's last step runs with the next turn of its trajectory (n2) and a fork (c)
    # admitted; p's finish arrives in begin_step(k+1) while commit k is in flight, so
    # n2's first block below its aux start waits for p's deferred teardown.
    end = s.reqs["p"].ntok + 1
    s.add(Req("n2", "s6", t, prompt_len=end + 4, aux=end - 1, chunk=16, hit=32))
    fp = 50
    s.add(
        Req(
            "c",
            "s6\x1fc\x1es6",
            t[:fp] + toks(9, 100),
            prompt_len=fp + 6,
            aux=fp,
            chunk=16,
            hit=(fp // BS) * BS,
        )
    )
    s.step(["p", "n2", "c"])
    s.finish("p")
    for _ in range(8):
        s.step(["n2", "c"])
    s.finish("n2")
    s.finish("c")
    s.step([])
    return s.close()


def case_fork_keys_after_capture(s):
    # The parent ends on a block boundary right after the fork point; the fork's first
    # capture runs before its hashes arrive, so its inherit must see the key count of
    # that moment.
    t = toks(15, 200)
    s.add(Req("p", "s9", t, prompt_len=20, chunk=16))
    while s.reqs["p"].computed < 32:
        s.step(["p"])
    s.finish("p")
    s.step([])
    s.drain()
    s.add(
        Req(
            "c",
            "s9\x1fc\x1es9",
            t[:32] + toks(16, 100),
            prompt_len=40,
            aux=31,
            chunk=16,
            hit=16,
            keys_from=1,
        )
    )
    s.step(["c"])
    s.drain()
    for _ in range(8):
        s.step(["c"])
    s.finish("c")
    s.step([])
    return s.close()


def case_release_after_finish(s):
    # Session B hits session A's system prompt: B's finish reads those blocks from the
    # base store.
    sys_p = toks(10, 48)
    s.add(Req("a", "sA", toks(11, 200, sys_p), prompt_len=60, chunk=24))
    for _ in range(4):
        s.step(["a"])
    s.add(Req("b", "sB", toks(12, 200, sys_p), prompt_len=60, chunk=24, hit=48))
    for _ in range(3):
        s.step(["a", "b"])
    s.finish("a")
    for _ in range(4):
        s.step(["b"])
    s.finish("b")
    s.step([])
    return s.close()


def case_fallback_reads_stash_state(s):
    # x is preempted mid-prefill; the release lets the store drop its blocks, and the
    # resumed request materializes them from the session copies (the fallback reads OC
    # state: drain).
    s.add(Req("x", "s8", toks(14, 200), prompt_len=80, chunk=24))
    s.step(["x"])
    s.step(["x"])
    s.preempt("x", hit=48)
    s.step([])
    s.drain()
    for _ in range(6):
        s.step(["x"])
    s.finish("x")
    s.step([])
    return s.close()


def case_error(s):
    calls = [0]
    orig = s.conn._oc_capture

    def boom(*a):
        calls[0] += 1
        if calls[0] == 5:
            raise ValueError("injected capture failure")
        return orig(*a)

    s.conn._oc_capture = boom
    s.add(Req("e", "s7", toks(13, 100), prompt_len=20, chunk=8))
    try:
        for _ in range(12):
            s.step(["e"])
        s.drain()
        s.conn._oc_drain()
        s.step(["e"])
        s.drain()
    except ValueError as e:
        return str(e), s.w.stats.get("oc_apply_errors", 0)
    return None, s.w.stats.get("oc_apply_errors", 0)


def release_on_base_thread(conn):
    """Negative control: the release published at teardown time, ahead of the queued
    finish."""
    orig = conn._teardown

    def td(ids, release_keys=(), stash_release=()):
        orig(ids, (), [*stash_release, *release_keys])

    conn._teardown = td


def case_parked_finish_pins_reads(s):
    # R19 #1: B's parked finish is queued; C's preemption then drops the last reference
    # to the shared prefix block and D's publishes evict it from a two-block store
    # before B's finish runs.
    tokens = toks(44, 100)
    s.add(Req("a", "sA", tokens, prompt_len=31, chunk=16))
    for _ in range(3):
        s.step(["a"])
    s.drain()
    s.conn._oc_drain()
    for rid in ("b", "c"):
        s.add(Req(rid, "s" + rid, tokens, prompt_len=31, chunk=16, hit=16))
    s.step(["b", "c"])
    s.finish("a")
    s.step([])
    s.conn._oc_drain()
    s.preempt("b", hit=16)
    s.step([])
    s.drain()
    s.conn._oc_drain()
    s.finish("b", parked=True)
    s.step([])
    s.preempt("c", hit=16)
    s.step([])
    s.add(Req("d", "sd", toks(45, 100), prompt_len=31, chunk=16))
    for _ in range(3):
        s.step(["d"])
    return s.close()


def small_store_plain(fn):
    saved = oc.EXTRA, oc.R3_PARTIAL, CFG.aux_output_config.max_bytes
    oc.EXTRA, oc.R3_PARTIAL, CFG.aux_output_config.max_bytes = (
        False,
        False,
        2 * BS * 40 * 2 * 2,
    )
    try:
        return fn()
    finally:
        oc.EXTRA, oc.R3_PARTIAL, CFG.aux_output_config.max_bytes = saved


def final_finish_error(mode):
    # R19 #2: the last finish fails on the apply thread with no later output: close()
    # raises it, and the request's keys are released all the same.
    s = Sim(mode)
    orig = s.conn._oc_finish

    def boom(*a):
        orig(*a)
        raise ValueError("final finish failed")

    s.conn._oc_finish = boom
    s.add(Req("f", "s10", toks(17, 100), prompt_len=20, chunk=16))
    raised = None
    try:
        for _ in range(4):
            s.step(["f"])
        s.drain()
        s.finish("f")
        s.step([])
    except ValueError as e:
        raised = ("begin_step", str(e))
    try:
        s.conn.close()
    except ValueError as e:
        raised = raised or ("close", str(e))
    return raised, dict(s.store._references)


def thread_released(mode):
    # R19 #3: close() stops the apply thread and the connector can be collected.
    import gc
    import weakref

    s = Sim(mode)
    s.add(Req("g", "s11", toks(18, 100), prompt_len=20, chunk=16))
    for _ in range(3):
        s.step(["g"])
    s.close()
    ref = weakref.ref(s.conn)
    del s
    gc.collect()
    return ref() is None, [
        t.name for t in threading.enumerate() if t.name == "oc-apply"
    ]


def finish_error_next_step(mode):
    # R19-fable #1: a finish failing on the apply thread surfaces in the next
    # begin_step, with no process_output in between; its keys are released and no
    # closing count is left behind.
    s = Sim(mode)
    orig = s.conn._oc_finish

    def boom(*a):
        orig(*a)
        raise ValueError("finish failed")

    s.conn._oc_finish = boom
    s.add(Req("f", "s12", toks(19, 100), prompt_len=20, chunk=16))
    where = None
    try:
        for _ in range(4):
            s.step(["f"])
        s.drain()
        s.finish("f")
        where = "teardown step"
        s.step([])
        where = "next step"
        s.step([])
        where = None
    except ValueError:
        pass
    with contextlib.suppress(ValueError):
        s.conn.close()
    return where, dict(s.store._references), dict(s.conn._oc_track_closing)


def posts_under_lock(mode):
    # R19-fable #3: every event is posted while _lock is held, so FIFO order is lock
    # order.
    s = Sim(mode)
    seen = []
    orig = s.conn._oc_post

    def post(fn, *a):
        seen.append((getattr(fn, "__name__", fn), s.conn._lock.locked()))
        orig(fn, *a)

    s.conn._oc_post = post
    s.add(Req("u", "s13", toks(20, 100), prompt_len=20, chunk=16))
    for _ in range(4):
        s.step(["u"])
    s.finish("u")
    s.step([])
    s.close()
    return sorted({name for name, locked in seen if not locked})


def bounded_backlog(bound):
    # R19-fable #2: with a slow apply thread, producers wait at the bound; delayed
    # finish releases then cannot pin more than the bound allows and a 12-object store
    # does not grow.
    import time

    if bound:
        os.environ["OC_APPLY_MAX_PENDING"] = str(bound)
    saved = CFG.aux_output_config.max_bytes
    CFG.aux_output_config.max_bytes = 12 * BS * int(np.prod(SHAPE)) * 2
    try:
        s = Sim("real")
        slots0 = s.store.num_slots
        depth = [0]
        orig = s.conn._oc_post

        def slow(fn):
            def run(*a):
                time.sleep(0.01)
                return fn(*a)

            return run

        def post(fn, *a):
            depth[0] = max(depth[0], s.conn._oc_q.qsize())
            orig(slow(fn), *a)

        s.conn._oc_post = post
        for i in range(10):
            rid = f"r{i}"
            s.add(Req(rid, f"g{i}", toks(100 + i, 100), prompt_len=64, chunk=32))
            for _ in range(3):
                s.step([rid])
            s.drain()
            s.finish(rid)
            s.step([])
            s.drain()
        s.close()
        return s.store.num_slots - slots0, depth[0], s.w.stats.get("oc_apply_wait_n", 0)
    finally:
        os.environ.pop("OC_APPLY_MAX_PENDING", None)
        CFG.aux_output_config.max_bytes = saved


def case_abort_while_preempted(s):
    # R19-fable #6: preempted with its teardown deferred, then aborted before that
    # teardown ran: the stash is written (same objects as before) but no record outlives
    # the request.
    s.add(Req("v", "s14", toks(21, 100), prompt_len=20, chunk=16))
    s.add(Req("w2", "s15", toks(22, 100), prompt_len=20, chunk=16))
    for _ in range(4):
        s.step(["v", "w2"])
    s.preempt("v", hit=16)
    s.step([])
    s.finish("v", aborted=True)
    s.step([])
    s.drain()
    s.add(Req("v2", "s14", toks(23, 100), prompt_len=20, chunk=16))
    s.preempt("w2", hit=16)
    s.step(["v2"])
    s.finish("w2", aborted=True)
    for _ in range(3):
        s.step(["v2"])
    s.finish("v2")
    s.step([])
    return s.close()


def case_hidden_encoding_change(s):
    # R20 #1: a request's snapshots change dtype at a resume (same row width in bytes):
    # its hidden field is withheld and counted, never written as one dtype holding rows
    # of two.
    HID[0] = "float16"
    try:
        s.add(Req("hx", "s16", toks(24, 100), prompt_len=20, chunk=16))
        s.add(Req("k", "s17", toks(25, 100), prompt_len=20, chunk=16))
        for _ in range(5):
            s.step(["hx", "k"])
        s.preempt("hx", hit=16)
        s.step(["k"])
        s.finish("k")
        s.step([])
        s.drain()
        HID[0] = "bfloat16"
        for _ in range(4):
            s.step(["hx"])
        s.finish("hx")
        s.step([])
        return s.close()
    finally:
        HID[0] = "raw"


if __name__ == "__main__":
    ref = check(case_hidden_encoding_change)
    fields = {
        k.split("/")[-2]: {f["name"]: f for f in json.loads(v)["fields"]}
        for k, v in MC[-1].objs.items()
        if k.endswith("/m")
    }
    assert (
        "hidden" not in fields["hx"] and fields["k"]["hidden"]["dtype"] == "float16"
    ), fields
    assert ref["stats"].get("hid_encoding_changed") == 1, ref["stats"]
    print("hidden encoding change: field withheld and counted")
    for case in (
        case_preempt_resume,
        case_generation,
        case_fork_parent_queued,
        case_parked_empty_stash,
    ):
        check(case)
        print("  raw hidden rows checked:", check_raw_hidden(MC[-1]))
    ref = check(case_abort_while_preempted)
    assert ref["preempted"] == [] and ref["stats"].get("preempted") == 2, (
        ref["preempted"],
        ref["stats"],
    )
    for mode in ("inline", 0):
        r = finish_error_next_step(mode)
        assert r == ("teardown step" if mode == "inline" else "next step", {}, {}), (
            mode,
            r,
        )
    print(
        "failed finish: raised by the next begin_step, keys released, no closing "
        "count left"
    )
    for mode in (0, "all"):
        assert posts_under_lock(mode) == [], posts_under_lock(mode)
    print("every event posted under _lock")
    grown, depth, waits = bounded_backlog(8)
    assert grown == 0 and depth <= 16 and waits > 0, (grown, depth, waits)
    print("unbounded (for contrast): grown, depth =", bounded_backlog(0)[:2])
    print(
        f"backlog bounded at 8 events: depth {depth}, {waits} producer waits, store "
        "did not grow"
    )
    ref = small_store_plain(lambda: check(case_parked_finish_pins_reads))
    assert ref["stats"].get("missing_base_blocks", 0) == 0, ref["stats"]
    for mode in ("inline", *MODES):
        r = final_finish_error(mode)
        want = "begin_step" if mode == "inline" else "close"
        assert r == ((want, "final finish failed"), {}), (mode, r)
    print("final finish failure: raised, keys released, in every mode")
    for mode in ("inline", *MODES):
        r = thread_released(mode)
        assert r == (True, []), (mode, r)
    print("close() stops the apply thread; connector collectable")
    ref = check(case_preempt_resume)
    assert ref["stats"].get("preempted") == 1 and ref["stats"].get("resumed") == 1, ref[
        "stats"
    ]
    assert (
        ref["stats"].get("stash_rows", 0) > 0
        and ref["stats"].get("spliced_rows", 0) > 0
    ), ref["stats"]
    ref = check(case_parked_empty_stash)
    m = [json.loads(v) for k, v in MC[-1].objs.items() if k.endswith("/p/m")][0]
    assert m["rows"] == 32 and len(m["blocks"]) == 2 and m["tail_nbytes"] == 0, m
    ref = check(case_generation)
    assert ref["stats"].get("preempted") == 2 and ref["stats"].get("resumed") == 2, ref[
        "stats"
    ]
    ref = check(case_fork_parent_queued)
    assert ref["stats"].get("deferred_blocks", 0) > 0, ref["stats"]
    ref = check(case_fork_keys_after_capture)
    assert ref["stats"].get("fork_tail_missing") == 1, ref["stats"]
    ref = check(case_fallback_reads_stash_state, evicting=True)
    assert ref["fallback_reads"] > 0, ref["fallback_reads"]
    ref = check(case_release_after_finish, evicting=True)
    assert (
        ref["stats"].get("missing_base_blocks", 0) == 0
        and ref["stats"].get("shared_earlier_block", 0) == 0
    )
    assert not ref["get_fail"] and sum(1 for k in ref["objs"] if "/sB/" in k) > 0
    bad = case_release_after_finish(
        Sim("all", evicting=True, conn_patch=release_on_base_thread)
    )
    assert (
        bad["stats"].get("missing_base_blocks", 0) > 0 and bad["objs"] != ref["objs"]
    ), bad["stats"]
    print(
        "negative control: releasing before the queued finish loses",
        bad["stats"]["missing_base_blocks"],
        "base blocks",
    )
    err, n = case_error(Sim("inline"))
    assert err == "injected capture failure" and n == 0, (err, n)
    for mode in MODES:
        e2, n2 = case_error(Sim(mode))
        assert e2 == err and n2 == 1, (mode, e2, n2)
    print(
        "injected oc-apply failure: counted once, re-raised by process_output in "
        "every mode"
    )
    print("all t55 tests passed")
