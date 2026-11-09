# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Output-connector experiment helpers shared by the engine overlay and the reader.

Design (env OC_DESIGN):
  c1  rows of prefix hits come from the local block store (image default, store sized to
      KV)
  c0  prefix hits are capped at the request's aux start (no cache segment)
  ps  prefix hits are capped at the longest prefix whose blocks this session already
      published (the delivery pool doubles as the session's cache)
Delivery is the same for every design: each full block is put to Mooncake under
  {OC_PREFIX}/{session}/{block_hash}
as soon as the block is complete; at request end the tail rows go to
  {OC_PREFIX}/{session}/{request}/t
and a manifest (ordered block hashes, tail start, total rows) to
  {OC_PREFIX}/{session}/{request}/m
With OC_EXTRA=1 two more outputs ride the same writer:
  indexer top-k (every position)  {OC_PREFIX}/{session}/{block_hash}/idx per full block,
                                  {OC_PREFIX}/{session}/{request}/t/idx for the tail;
                                  int32 [layers, topk], -1 padded, packed 20-bit +
                                  zstd-3
  hidden (lm_head input, sampled positions)
                                  {OC_PREFIX}/{session}/{request}/h/{i}, chunks of
                                  OC_HID_CHUNK rows: the rows exactly as they enter
                                  lm_head, in their own dtype (codec "raw")
  top-k logprobs (OC_EXTRA_TOPK=1, sampled positions)
                                  {OC_PREFIX}/{session}/{request}/k/{i}, chunks of
                                  OC_HID_CHUNK rows; per row OC_TOPK int32 ids then
                                  OC_TOPK float32 logprobs of the raw distribution
                                  (temperature 1), ids by descending logit (codec "raw")
The manifest's "fields" list describes every output (name, dtype, shape, codec, nbytes).
"""

from __future__ import annotations

import collections
import concurrent.futures
import contextlib
import hashlib
import itertools
import json
import os
import pathlib
import queue
import shutil
import threading
import time

import numpy as np
import regex as re

DESIGN = os.environ.get("OC_DESIGN", "c1")
PREFIX = os.environ.get("OC_PREFIX", "oc/default")
ABORT_DIR = os.environ.get("OC_ABORT_DIR", "")
# The consumer marks a released session with a file here; the writer then drops its
# bookkeeping.
RELEASE_DIR = os.environ.get("OC_RELEASE_DIR", "")
ENABLED = bool(os.environ.get("OC_MC_MASTER"))
EXTRA = os.environ.get("OC_EXTRA", "0") == "1"
# Overhead attribution: full (default) | raw (no pack/zstd) | noput (indexer rows built,
# never encoded or stored) | nod2h (indexer rows captured on GPU only, no device-to-host
# copy).
IDX_MODE = os.environ.get("OC_IDX_MODE", "full")
# Weight version of the rows being captured; bumped by the trainer's weight update
# (collective_rpc).
WEIGHT_VERSION = [0]


# Scheduler-level backpressure: while a writer's backlog is above BP_HIGH it keeps a
# flag file; the patched scheduler sees it and schedules 0 tokens, so the forward never
# blocks in put().
BP = os.environ.get("OC_BP") == "1"
BP_DIR = "/dev/shm"


def bp_flag_prefix() -> str:
    return "oc_bp_" + PREFIX.replace("/", "_") + "_"


def bp_paused() -> bool:
    if not BP:
        return False
    pre = bp_flag_prefix()
    try:
        return any(n.startswith(pre) for n in os.listdir(BP_DIR))
    except OSError:
        return False


def bump_weight_version() -> int:
    WEIGHT_VERSION[0] += 1
    return WEIGHT_VERSION[0]


DUMP_DIR = os.environ.get("OC_DUMP", "")
DUMP_LOGITS_RE = os.environ.get("OC_DUMP_LOGITS_RE", r"-0$")
IDX_SUFFIX = "/idx"
IDX_CODEC = (
    "dplane20+zstd3" if os.environ.get("OC_IDX_MODE") == "gpudelta" else "pack20+zstd3"
)
# Raw hidden rows: bytes per element by the manifest's dtype (torch dtype names).
HID_ITEMSIZE = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2}
HID_CHUNK = int(os.environ.get("OC_HID_CHUNK", "64"))
TOPK_K = int(os.environ.get("OC_TOPK", "128"))
# A writer whose puts have all failed for this long while puts are pending stops
# retrying and reports a fatal error (the driver aborts the run): disk or pool failures
# do not heal by retrying.
PUT_STALL_S = float(os.environ.get("OC_PUT_STALL_S", "600"))


def block_key(session: str, block_hash_hex: str) -> str:
    return f"{PREFIX}/{session}/{block_hash_hex}"


def idx_tail_key(session: str, request_id: str) -> str:
    return f"{PREFIX}/{session}/{request_id}/t/idx"


def hid_key(session: str, request_id: str, i: int) -> str:
    return f"{PREFIX}/{session}/{request_id}/h/{i}"


def topk_key(session: str, request_id: str, i: int) -> str:
    return f"{PREFIX}/{session}/{request_id}/k/{i}"


def is_topk_key(key: str) -> bool:
    parts = key[len(PREFIX) + 1 :].split("/")
    return len(parts) >= 4 and parts[-2] == "k" and parts[-1].isdigit()


def key_id(request_id: str) -> str:
    # Engine-internal ids are '<caller id>-<8 random chars>'; key by the caller's id.
    return (
        request_id[:-9] if len(request_id) > 9 and request_id[-9] == "-" else request_id
    )


def cone_key(session: str, request_id: str, kind: str = "r3") -> str:
    return f"{PREFIX}/{session}/{request_id}/t/c/{kind}"


def tail_key(session: str, request_id: str) -> str:
    return f"{PREFIX}/{session}/{request_id}/t"


def manifest_key(session: str, request_id: str) -> str:
    return f"{PREFIX}/{session}/{request_id}/m"


INJECTED_KEYS: set[str] = set()


def note_injected(key: str) -> None:
    """Test only: records a key whose write was failed on purpose (OC_INJECT_LOG for
    the checkers)."""
    INJECTED_KEYS.add(key)
    path = os.environ.get("OC_INJECT_LOG")
    if path:
        with open(path, "a") as f:
            f.write(key + "\n")


def injected_keys() -> set[str]:
    keys = set(INJECTED_KEYS)
    path = os.environ.get("OC_INJECT_LOG")
    if path and os.path.exists(path):
        with open(path) as f:
            keys.update(x.strip() for x in f if x.strip())
    return keys


def marker_failed_keys(data) -> list[str]:
    """The failed object names an error marker carries (JSON, or the older "<key>
    <error>" text)."""
    try:
        return [str(k) for k in json.loads(bytes(data))["failed"]]
    except (ValueError, KeyError, TypeError):
        pass
    text = bytes(data).decode("utf-8", "replace")
    key, _, err = text.partition(" ")
    names = re.findall(r"'([^'\\]+)\\?': ", err.partition("names failed objects: ")[2])
    return names if names else [key] if key else []


def err_key(manifest: str) -> str:
    # Set by the writer when a request's manifest could not be written: its reader fails
    # at once instead of timing out. A failed block needs no marker: the manifest still
    # lists it and the reader finds the object missing, for that request only. The
    # release sweep removes it.
    return manifest + "!err"


SSD_OWNER = ".oc_ssd_owner"


def check_ssd_pin() -> None:
    """Hard-pinned objects are never evicted, so with SSD offload the pool stalls
    instead of spilling: the two are refused together (OC_HARD_PIN defaults to 1)."""
    if os.environ.get("OC_MC_SSD_PATH") and os.environ.get("OC_HARD_PIN", "1") != "0":
        raise RuntimeError(
            "OC_MC_SSD_PATH needs OC_HARD_PIN=0: hard-pinned objects cannot be "
            "offloaded"
        )


def _job_tag() -> str:
    return os.environ.get("SLURM_JOB_ID", "")


def _host() -> str:
    import socket

    return socket.gethostname()


# Offload dirs this process created and marked (realpaths): the only ones it may wipe or
# remove.
_SSD_CREATED: set[str] = set()


def ssd_owned(path: str) -> bool:
    """This process created and marked the dir, for this Slurm job: its marker names
    this job, pid and host. A missing job id (in the marker or the environment) never
    matches."""
    if os.path.realpath(path) not in _SSD_CREATED:
        return False
    try:
        with open(os.path.join(path, SSD_OWNER)) as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return False
    job = _job_tag()
    return (
        bool(job)
        and rec.get("job") == job
        and rec.get("pid") == os.getpid()
        and rec.get("host") == _host()
    )


def _pid_alive(pid) -> bool:
    """POSIX only (signal 0); elsewhere a pid is assumed alive, so nothing is wiped
    on its account."""
    if os.name != "posix":
        return True
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, TypeError):
        return True
    return True


def _read_marker(path: str) -> dict | None:
    try:
        with open(os.path.join(path, SSD_OWNER)) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def ssd_wipe_allowed(path: str, rec: dict | None, token: str | None = None) -> bool:
    """A non-empty dir may be wiped (OC_MC_SSD_WIPE=1) when its marker shows it is ours:
    created by this process, or carrying our driver's token, or left by a process on
    this host that is gone. The same job id alone is not enough: another live holder of
    the job may own it."""
    if rec is None:
        return False
    if ssd_owned(path):
        return True
    if token is not None and rec.get("token") == token:
        return True
    return (
        rec.get("host") == _host()
        and rec.get("pid") is not None
        and not _pid_alive(rec["pid"])
    )


def ssd_prepare_dir(path: str, token: str | None = None) -> str:
    """An absolute dir that is new, empty, or holds only the owner marker of the driver
    that spawned this holder (`token`, from OC_MC_SSD_TOKEN). The claim is the marker,
    created exclusively: of two drivers claiming one empty dir, the second fails. A
    non-empty dir is wiped (its contents; the dir stays) only on OC_MC_SSD_WIPE=1 and
    only if its marker shows it is ours (ssd_wipe_allowed); anything else raises: a
    foreign tree is never deleted or shared. Returns the marker's token."""
    if not os.path.isabs(path):
        raise ValueError(f"OC_MC_SSD_PATH must be absolute: {path!r}")
    marker = os.path.join(path, SSD_OWNER)
    created = None
    if os.path.isdir(path):
        names = os.listdir(path)
        rec = _read_marker(path)
        if names == [SSD_OWNER] and token is not None:
            if rec is None or rec.get("token") != token:
                raise RuntimeError(
                    f"SSD offload dir {path} was prepared by another driver; not "
                    "sharing it"
                )
            return token
        if names:
            if os.environ.get("OC_MC_SSD_WIPE") != "1":
                raise RuntimeError(
                    f"SSD offload dir {path} is not empty (or claimed by another "
                    "driver); "
                    "a holder restarted on old buckets re-registers removed keys"
                )
            if not ssd_wipe_allowed(path, rec, token):
                raise RuntimeError(
                    f"SSD offload dir {path} is not empty and its marker does not "
                    "show it is ours; "
                    "not wiping it"
                )
            for name in names:
                p_ = os.path.join(path, name)
                shutil.rmtree(p_) if os.path.isdir(p_) and not os.path.islink(
                    p_
                ) else os.remove(p_)
            created = bool(rec.get("created", False))
    if created is None:
        try:
            os.makedirs(path)
            created = True
        except FileExistsError:
            created = False
    import uuid

    new = f"{_job_tag()}:{os.getpid()}:{_host()}:{uuid.uuid4().hex}"
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise RuntimeError(
            f"SSD offload dir {path} was claimed by another driver at the same time"
        ) from None
    with os.fdopen(fd, "w") as f:
        json.dump(
            dict(
                job=_job_tag(),
                pid=os.getpid(),
                host=_host(),
                token=new,
                created=created,
            ),
            f,
        )
    _SSD_CREATED.add(os.path.realpath(path))
    return new


def ssd_remove_owned(path: str | None) -> bool:
    """Remove an SSD offload dir only if this process created it (see ssd_owned). Call
    only after its holder is dead."""
    if not path:
        return False
    path = os.path.expandvars(path)
    if not os.path.isdir(path) or not ssd_owned(path):
        return False
    created = json.loads(pathlib.Path(os.path.join(path, SSD_OWNER)).read_text()).get(
        "created", True
    )
    if created:
        shutil.rmtree(path)
    else:
        for name in os.listdir(path):
            p_ = os.path.join(path, name)
            shutil.rmtree(p_) if os.path.isdir(p_) and not os.path.islink(
                p_
            ) else os.remove(p_)
    _SSD_CREATED.discard(os.path.realpath(path))
    return True


def manifest_object_keys(m: dict) -> set[str]:
    """Every stored object a manifest names: R3 blocks and tail, indexer blocks and
    tail, hidden chunks, cone patches (R3 and indexer). A request whose manifest names a
    failed object fails."""
    pre = m["key_prefix"]
    keys = {f"{pre}/{b}" for b in m.get("blocks", [])}
    if m.get("tail_key"):
        keys.add(m["tail_key"])
    for f in m.get("fields", []):
        if f.get("name") == "indexer_topk":
            keys.update(
                f"{pre}/{b}{f.get('block_suffix', IDX_SUFFIX)}"
                for b in m.get("blocks", [])
            )
            if f.get("tail_key"):
                keys.add(f["tail_key"])
        elif f.get("name") in ("hidden", "topk"):
            keys.update(f"{f['key_prefix']}{i}" for i in range(int(f.get("chunks", 0))))
    for patch in (m.get("r3_cone") or {}).get("patches", []):
        keys.update(patch[k] for k in ("key", "idx_key") if patch.get(k))
    return keys


def make_store(segment_bytes: int = 0, buffer_bytes: int = 1 << 30):
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    ssd = os.environ.get("OC_MC_SSD_PATH")
    if ssd and segment_bytes:
        check_ssd_pin()
        path = os.path.expandvars(ssd)
        ssd_prepare_dir(path, os.environ.get("OC_MC_SSD_TOKEN"))
        rc = store.setup(
            dict(
                local_hostname=os.environ.get("OC_MC_LOCAL", "127.0.0.1"),
                metadata_server=os.environ.get("OC_MC_METADATA", "P2PHANDSHAKE"),
                global_segment_size=segment_bytes,
                local_buffer_size=buffer_bytes,
                protocol=os.environ.get("OC_MC_PROTOCOL", "tcp"),
                rdma_devices=os.environ.get("OC_MC_DEVICE", ""),
                master_server_addr=os.environ["OC_MC_MASTER"],
                enable_ssd_offload=True,
                ssd_offload_path=path,
            )
        )
        if rc:
            raise RuntimeError(f"mooncake setup failed: {rc}")
        return store
    rc = store.setup(
        os.environ.get("OC_MC_LOCAL", "127.0.0.1"),
        os.environ.get("OC_MC_METADATA", "P2PHANDSHAKE"),
        segment_bytes,
        buffer_bytes,
        os.environ.get("OC_MC_PROTOCOL", "tcp"),
        os.environ.get("OC_MC_DEVICE", ""),
        os.environ["OC_MC_MASTER"],
    )
    if rc:
        raise RuntimeError(f"mooncake setup failed: {rc}")
    return store


def ssd_reserve() -> int:
    return int(os.environ.get("OC_SSD_RESERVE_BYTES", str(16 << 30)))


def ssd_headroom(ssd_path: str, segment_bytes: int) -> int:
    """Offload bytes still free for new sessions: the dir's free space less the memory
    segment (which spills there) and a safety reserve."""
    return (
        shutil.disk_usage(os.path.expandvars(ssd_path)).free
        - segment_bytes
        - ssd_reserve()
    )


def admission_c(
    sessions: int,
    sess_bytes: float,
    group: int,
    pool_bytes: int,
    c_mult: float = 1.0,
    ssd_path: str | None = None,
) -> int:
    """Concurrent session groups the pool admits. With SSD offload: the offload headroom
    over one group's bytes, and an error when not even one group fits. Otherwise 75% of
    the pool, times OC_C_MULT."""
    if ssd_path:
        need = sess_bytes * group
        room = ssd_headroom(ssd_path, pool_bytes)
        if room < need:
            raise RuntimeError(
                f"SSD offload dir {ssd_path} has {room / 2**30:.1f} GiB of headroom "
                "(free less the "
                f"{pool_bytes / 2**30:.1f} GiB segment and the "
                f"{ssd_reserve() / 2**30:.1f} GiB reserve); "
                f"one session group needs {need / 2**30:.1f} GiB"
            )
        return min(sessions, int(room // need))
    return max(1, min(sessions, int(c_mult * 0.75 * pool_bytes / (sess_bytes * group))))


def release_session(store, name: str, release_dir: str | None = None) -> int:
    """Consumer release of a session: mark it released first (the writer then drops its
    queued and retried puts and its dedup state), then remove its objects. Returns the
    objects removed. A put already being written when the marker appeared can still
    land: settle it with a second sweep once writer_drain_time() passes the release
    time."""
    if release_dir:
        os.makedirs(release_dir, exist_ok=True)
        open(os.path.join(release_dir, name), "w").close()
    return store.remove_by_regex("^" + PREFIX + "/" + name + "/", True)


def writer_reports(stats_prefix: str, since: float = 0.0) -> dict[int, dict]:
    """Last complete report of every writer process (OC_STATS_FILE.<pid>) written at or
    after `since`: a previous run's files on the same path are not this run's writers.
    """
    import glob

    out = {}
    for path in glob.glob(glob.escape(stats_prefix) + ".*"):
        try:
            pid = int(path.rsplit(".", 1)[1])
            if os.path.getmtime(path) < since:
                continue
        except (ValueError, OSError):
            continue
        last = None
        try:
            with open(path) as f:
                for line in f:
                    if line.endswith("\n"):
                        last = line
            if last:
                rec = json.loads(last)
                if float(rec.get("t", 0)) >= since:
                    out[pid] = rec
        except (OSError, ValueError):
            continue
    return out


def fence_waiting(reports: dict, writers, session: str, t_release: float) -> list:
    """Expected writers that have not yet reported, after the release, nothing queued or
    in flight for the session. Empty: no put of the session can land any more (a put
    picked up after the release marker is dropped). A writer that never registered or
    stopped reporting keeps it open."""
    waiting = []
    for pid in writers:
        rec = reports.get(pid)
        if (
            rec is None
            or float(rec.get("t", 0)) <= t_release
            or session in (rec.get("pending_sessions") or {})
        ):
            waiting.append(pid)
    return waiting


def call_bounded(fn, timeout_s: float):
    """Run fn on a daemon thread; ("ok", result), ("error", exception) or ("timeout",
    None)."""
    box = {}

    def run():
        try:
            box["r"] = ("ok", fn())
        except BaseException as e:  # noqa: BLE001
            box["r"] = ("error", e)

    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout_s)
    return box.get("r", ("timeout", None))


def refetch_missing(
    get, keys, vals, delays=(0.05, 0.1, 0.2, 0.3, 0.35), sleep=time.sleep
) -> int:
    """Shared backoff for every key that read empty: rounds of single gets of the keys
    still missing, spaced by `delays` (1 s in all, whatever the number of keys). Fills
    `vals` in place; returns the gets issued. A disk-backed object can read empty for a
    moment (mid-offload, holder heartbeat)."""
    miss = [i for i, v in enumerate(vals) if v is None]
    gets = 0
    for d in delays:
        if not miss:
            break
        sleep(d)
        for i in miss:
            v = get(keys[i])
            gets += 1
            vals[i] = bytes(v) if v else None
        miss = [i for i in miss if vals[i] is None]
    return gets


def dead_procs(procs) -> list[dict]:
    """Helper processes (master, holder) that have exited."""
    out = []
    for i, p in enumerate(procs):
        rc = p.poll()
        if rc is not None:
            out.append(
                dict(i=i, args=" ".join(map(str, getattr(p, "args", [])))[:200], rc=rc)
            )
    return out


def _resolve_hit_segs(segs, src):
    """Replace -1 segments (rows from a prefix hit) by the versions `src` records
    there."""

    def at(sg, p):
        v = None
        for s0, v0 in sg:
            if s0 <= p:
                v = v0
            else:
                break
        return v

    bounds = sorted({int(x) for x, _ in segs} | {int(x) for x, _ in src})
    out = []
    for p in bounds:
        v = at(segs, p)
        if v == -1:
            v = at(src, p)
        if v is None or v == -1:
            v = -1
        if not out or out[-1][1] != v:
            out.append([p, v])
    return out


def _segs_match(a, b) -> bool:
    """Whether two block version records (segments [offset, version] within one block)
    describe the same version at every position. Rows before a record's first offset
    came from a prefix hit on the session's object, so they carry that object's versions
    and match either way."""
    if a is None or b is None:
        return True

    def at(segs, p):
        v = None
        for s0, v0 in segs:
            if s0 <= p:
                v = v0
            else:
                break
        return v

    m = max(int(a[0][0]), int(b[0][0]))
    points = sorted(
        {m} | {int(x) for x, _ in a if x > m} | {int(x) for x, _ in b if x > m}
    )
    # -1 marks rows served by a prefix hit: their version is the stored object's, so
    # they match.
    return all(at(a, p) == at(b, p) or -1 in (at(a, p), at(b, p)) for p in points)


PROF = os.environ.get("OC_PROF") == "1"
ENC_THREADS = int(os.environ.get("OC_ENC_THREADS", "4"))
PUT_BATCH = int(os.environ.get("OC_PUT_BATCH", "32"))
# Engine-side hop accounting (worker process): D2H bytes and host time of each step's
# output.
ENGINE_PROF: dict[str, float] = {}


def _prof_kind(key: str) -> str:
    if key.endswith(IDX_SUFFIX):
        return "idx_block"
    if key.endswith("/t/idx"):
        return "idx_tail"
    if "/h/" in key:
        return "hidden"
    if is_topk_key(key):
        return "topk"
    if key.endswith("/m") or key.endswith("!err"):
        return "manifest"
    if key.endswith("/t") or "/t/" in key:
        return "tail"
    return "r3_block"


def _session_of(key: str) -> str:
    return (
        key[len(PREFIX) + 1 :].split("/", 1)[0] if key.startswith(PREFIX + "/") else ""
    )


class Writer:
    """Background, ordered, bounded Mooncake writer. Puts never fail silently:
    a put that cannot fit is retried until it lands (backpressure)."""

    def __init__(
        self, max_pending: int = int(os.environ.get("OC_WRITER_MAX_PENDING", "4096"))
    ) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=max_pending)
        # OC_WRITER_FAIR=1: one FIFO per session, served oldest session first; a session
        # whose head put does not fit is skipped, so a finished session's tail and
        # manifest can land (and the session be released) while other sessions wait for
        # space.
        self._fair = os.environ.get("OC_WRITER_FAIR") == "1"
        self._sq: dict[str, collections.deque] = {}
        self._cv = threading.Condition()
        self._max_pending = max_pending
        self._pending = 0
        self._bp_high, self._bp_low = max_pending // 2, max_pending // 4
        self._bp_flag = os.path.join(BP_DIR, bp_flag_prefix() + str(os.getpid()))
        self._bp_on = False
        self._store = make_store()
        self._lock = threading.Lock()
        self.stats = {"puts": 0, "bytes": 0, "retries": 0, "skipped_dup": 0}
        # Put latency (first attempt to landed, retries included), log-spaced bins 1 us
        # .. 1000 s.
        self._edges = np.logspace(-6, 3, 181)
        self._hist = np.zeros(182, np.int64)
        self._cfg = None
        check_ssd_pin()
        if os.environ.get("OC_HARD_PIN", "1") != "0":
            # Delivered rows must not be evicted: a full pool makes puts fail and retry
            # (flow control); it never drops an object a manifest lists. This Mooncake
            # build has no with_soft_pin; hard-pinned objects are never evicted, only
            # removed (the trainer's per-session release, or the abort fallback).
            from mooncake.store import ReplicateConfig

            self._cfg = ReplicateConfig()
            self._cfg.with_hard_pin = True
        self.stats["hard_pin"] = self._cfg is not None
        self._stall_since: float | None = None
        # Object keys whose write failed, per session (never rewritten; requests naming
        # them fail).
        self._failed: dict[str, dict[str, str]] = {}
        self._sess_pending: collections.Counter = collections.Counter()
        self._sess_lock = threading.Lock()
        self._published: dict[str, set[str]] = {}
        self._digest: dict[str, dict[str, bytes]] = {}
        self._first_tag: dict = {}
        self._owner: dict[str, dict[str, str]] = {}
        self._poisoned: dict[str, set[str]] = {}
        self._segs: dict[str, dict] = {}
        self._track_has: dict[tuple[str, str], dict[str, str]] = {}
        self.last_tail: dict[str, tuple[int, np.ndarray]] = {}
        self.last_tail_idx: dict[str, tuple[int, np.ndarray]] = {}
        # Per session: [(first position, weight version), ...] of the rows delivered so
        # far.
        self.weight_versions: dict[str, list[list[int]]] = {}
        # Stored size of every key put, for the manifest's per-field nbytes.
        self.sizes: dict[str, int] = {}
        self._t_enq: dict[str, float] = {}
        self._enc_pool = None
        self.prof: dict[str, dict] = {}
        self._t_start = time.time()
        self._thread = threading.Thread(
            target=self._run_fair if self._fair else self._run,
            daemon=True,
            name="oc-writer",
        )
        if os.environ.get("OC_STATS_FILE"):
            threading.Thread(
                target=self._stats_loop, daemon=True, name="oc-stats"
            ).start()
        elif RELEASE_DIR:
            threading.Thread(
                target=self._prune_loop, daemon=True, name="oc-prune"
            ).start()
        self._thread.start()

    def _prof_add(self, kind: str, field: str, v) -> None:
        d = self.prof.setdefault(kind, {})
        d[field] = d.get(field, 0) + v

    def published(self, session: str) -> set[str]:
        with self._lock:
            return self._published.setdefault(session, set())

    def put_block(
        self, session: str, block_hash_hex: str, payload, suffix: str = ""
    ) -> None:
        with self._lock:
            done = self._published.setdefault(session, set())
            if block_hash_hex + suffix in done:
                self.stats["skipped_dup"] += 1
                return
            done.add(block_hash_hex + suffix)
        self.put(block_key(session, block_hash_hex) + suffix, payload)

    def note_has(self, session: str, tag: str, names: dict) -> None:
        # Blocks a request listed (computed or hit) are what its trajectory holds from
        # now on: a later recompute of the same hash by this trajectory is dropped.
        with self._lock:
            done = self._published.setdefault(session, set())
            has = self._track_has.setdefault((session, tag), {})
            for h, name in names.items():
                # Only objects that were actually published: a block this request lists
                # but nobody delivered must stay recoverable by a later recompute.
                if name in done:
                    has.setdefault(h, name)

    def object_segs(self, session: str, name: str):
        with self._lock:
            return self._segs.get(session, {}).get(name)

    def holds(self, session: str, tag: str, block_hash_hex: str) -> bool:
        with self._lock:
            return block_hash_hex in self._track_has.get((session, tag), {})

    def inherit_has(self, session: str, parent_tag: str, tag: str, hashes) -> None:
        # A fork holds what its parent held at the fork point: only the prefix blocks it
        # starts from, not blocks the parent computed later (in the same step as the
        # fork's own).
        with self._lock:
            src = self._track_has.get((session, parent_tag))
            if src:
                dst = self._track_has.setdefault((session, tag), {})
                for h in hashes:
                    if h in src:
                        dst.setdefault(h, src[h])

    def share_if_same_version(self, session: str, block_hash_hex: str, tag: str, segs):
        """Name of the object this trajectory may list for a block it did not compute
        exactly: its own, or the session's if the versions match. None when there is
        none."""
        with self._lock:
            has = self._track_has.setdefault((session, tag), {})
            if block_hash_hex in has:
                return has[block_hash_hex]
            if (
                block_hash_hex in self._published.get(session, ())
                and block_hash_hex not in self._poisoned.get(session, ())
                and _segs_match(segs, self._segs.get(session, {}).get(block_hash_hex))
            ):
                has[block_hash_hex] = block_hash_hex
                return block_hash_hex
        return None

    def put_block_checked(
        self, session: str, block_hash_hex: str, data: bytes, tag: str, segs=None
    ) -> tuple[str, bool]:
        """Publish a block this trajectory computed. First computation wins per
        trajectory: if the trajectory already published this hash (under it or under its
        own name), the new copy is dropped. If another trajectory published the hash
        with the same bytes, it is shared; with different bytes (the same tokens
        prefilled in different chunks), it is kept under the trajectory's own name,
        which the manifest lists. Returns (name, newly published)."""
        d = hashlib.blake2b(data, digest_size=16).digest()
        own = f"{block_hash_hex}~{tag}"
        with self._lock:
            done = self._published.setdefault(session, set())
            dig = self._digest.setdefault(session, {})
            owner = self._owner.setdefault(session, {})
            has = self._track_has.setdefault((session, tag), {})
            if block_hash_hex in self._poisoned.get(session, ()):
                # A failed name is never rewritten or replaced: requests listing it
                # fail.
                self.stats["lost_kept_missing"] = (
                    self.stats.get("lost_kept_missing", 0) + 1
                )
                return block_hash_hex, False
            if block_hash_hex in has:
                # This trajectory already has this block (its own copy or a shared
                # identical one).
                self.stats["skipped_dup"] += 1
                return has[block_hash_hex], False
            if block_hash_hex in done:
                first = self._segs.get(session, {}).get(block_hash_hex)
                if owner.get(block_hash_hex) == tag or (
                    dig.get(block_hash_hex, d) == d and _segs_match(segs, first)
                ):
                    self.stats["skipped_dup"] += 1
                    has[block_hash_hex] = block_hash_hex
                    return block_hash_hex, False
                self.stats["dup_diverged"] = self.stats.get("dup_diverged", 0) + 1
                same_version = _segs_match(segs, first)
                # Same weight version: under the train-inference consistent
                # configuration every computation of these tokens is bitwise identical,
                # so the session's first copy serves every trajectory (OC_ALT=1 keeps
                # per-trajectory copies for diagnosis). Another version is not
                # interchangeable: this trajectory keeps its own copy.
                if same_version and os.environ.get("OC_ALT", "0") != "1":
                    return block_hash_hex, False
                if (
                    segs is not None
                    and first is not None
                    and any(v == -1 for _, v in segs)
                ):
                    # Rows marked -1 came from a prefix hit on the session's object for
                    # this block: this copy carries that object's versions there.
                    segs = _resolve_hit_segs(segs, first)
                if not same_version:
                    self.stats["dup_other_version"] = (
                        self.stats.get("dup_other_version", 0) + 1
                    )
                    if self.stats["dup_other_version"] <= 10:
                        print(
                            f"OC_OTHER_VERSION session={session} "
                            f"hash={block_hash_hex[:12]} tag={tag} "
                            f"first_owner={owner.get(block_hash_hex)} first={first} "
                            f"mine={segs}",
                            flush=True,
                        )
                name = own
                if own in self._poisoned.get(session, ()):
                    # This trajectory's own copy failed earlier: it stays missing, never
                    # rewritten.
                    self.stats["lost_kept_missing"] = (
                        self.stats.get("lost_kept_missing", 0) + 1
                    )
                    has[block_hash_hex] = own
                    return own, False
            else:
                name = block_hash_hex
            done.add(name)
            dig[name] = d
            owner[name] = tag
            has[block_hash_hex] = name
            self._segs.setdefault(session, {}).setdefault(name, segs)
        self.put(block_key(session, name), data)
        return name, True

    def put(self, key: str, payload) -> None:
        """payload: bytes, or a callable run on the writer thread (encoding,
        manifests)."""
        if IDX_MODE == "noput" and (key.endswith(IDX_SUFFIX) or key.endswith("/t/idx")):
            self.stats["idx_noput"] = self.stats.get("idx_noput", 0) + 1
            return
        if (
            ENC_THREADS
            and callable(payload)
            and (key.endswith(IDX_SUFFIX) or key.endswith("/t/idx"))
        ):
            # Indexer encoding (zstd, GIL released) runs on a pool ahead of the writer,
            # which only takes the result in order. Manifests stay on the writer thread:
            # they read the sizes of the objects put before them.
            if self._enc_pool is None:
                self._enc_pool = concurrent.futures.ThreadPoolExecutor(
                    ENC_THREADS, "oc-enc"
                )
            payload = self._enc_pool.submit(payload)
        session = _session_of(key)
        with self._sess_lock:
            self._sess_pending[session] += 1
        if not self._fair:
            self._q.put((key, payload))
            return
        with self._cv:
            if self._pending >= self._max_pending:
                t_wait = time.time()
                while self._pending >= self._max_pending:
                    self._cv.wait()
                # Time the engine's forward (the caller) spent blocked on a full writer
                # queue.
                self.stats["put_block_s"] = (
                    self.stats.get("put_block_s", 0.0) + time.time() - t_wait
                )
                self.stats["put_block_n"] = self.stats.get("put_block_n", 0) + 1
            self._pending += 1
            if PROF:
                self._t_enq[key] = time.perf_counter()
            self._sq.setdefault(session, collections.deque()).append((key, payload))
            if BP and not self._bp_on and self._pending >= self._bp_high:
                self._bp_on = True
                self._bp_t = time.time()
                self.stats["bp_on_n"] = self.stats.get("bp_on_n", 0) + 1
                open(self._bp_flag, "w").close()
            self._cv.notify_all()

    def qsize(self) -> int:
        return self._pending if self._fair else self._q.qsize()

    def flush(self) -> None:
        if not self._fair:
            self._q.join()
            return
        with self._cv:
            while self._pending:
                self._cv.wait()

    def _sess_done(self, session: str, n: int = 1) -> None:
        with self._sess_lock:
            self._sess_pending[session] -= n
            if self._sess_pending[session] <= 0:
                del self._sess_pending[session]
            if not self._sess_pending:
                self._stall_since = None

    def pending_sessions(self) -> dict[str, int]:
        with self._sess_lock:
            return dict(self._sess_pending)

    def _put_failed(self, first_attempt: bool) -> bool:
        """A put attempt failed; True (drop it) once no put has succeeded for
        PUT_STALL_S since the oldest pending put first failed. A put's first attempt is
        never dropped."""
        now = time.time()
        if self._stall_since is None:
            self._stall_since = now
        if first_attempt or now - self._stall_since < PUT_STALL_S:
            return False
        if "fatal" not in self.stats:
            self.stats["fatal"] = (
                f"put_stall: no put succeeded for {now - self._stall_since:.0f} s"
            )
            print(f"oc-writer FATAL {self.stats['fatal']}", flush=True)
        self.stats["dropped_put_stall"] = self.stats.get("dropped_put_stall", 0) + 1
        return True

    def _drop(self, key: str, n: int = 1) -> None:
        released = RELEASE_DIR and os.path.exists(
            os.path.join(RELEASE_DIR, _session_of(key))
        )
        name = "dropped_after_release" if released else "dropped_aborted"
        self.stats[name] = self.stats.get(name, 0) + n

    def _done(self, session: str, n: int = 1) -> None:
        self._sess_done(session, n)
        with self._cv:
            q = self._sq.get(session)
            for _ in range(n):
                q.popleft()
            if not q:
                del self._sq[session]
            self._pending -= n
            if self._bp_on and self._pending <= self._bp_low:
                self._bp_on = False
                self.stats["bp_on_s"] = (
                    self.stats.get("bp_on_s", 0.0) + time.time() - self._bp_t
                )
                with contextlib.suppress(OSError):
                    os.unlink(self._bp_flag)
            self._cv.notify_all()

    def _try_put(self, key: str, payload) -> bool:
        rc = (
            self._store.put(key, payload)
            if self._cfg is None
            else self._store.put(key, payload, self._cfg)
        )
        # Every TP rank publishes the same keys; a key that already exists is done.
        return rc == 0 or getattr(self._store, "is_exist", lambda _k: 0)(key) == 1

    def _run_fair(self) -> None:
        enc: dict[str, bytes] = {}
        t_first: dict[str, float] = {}
        while True:
            with self._cv:
                while not self._sq:
                    self._cv.wait()
                order = list(self._sq.items())  # dict order = first put of the session
            progressed = False
            for session, q in order:
                if self._put_ready_batch(session, q, enc, t_first):
                    progressed = True
                    break
                key, payload = q[0]
                if self._aborted(key):
                    n = len(q)
                    self._drop(key, n)
                    for k_, _ in list(q)[:n]:
                        enc.pop(k_, None)
                        t_first.pop(k_, None)
                    self._done(session, n)
                    progressed = True
                    break
                try:
                    if key not in enc:
                        t_enc = time.perf_counter()
                        if isinstance(payload, concurrent.futures.Future):
                            enc[key] = payload.result()
                        else:
                            enc[key] = payload() if callable(payload) else payload
                        self.stats["enc_s"] = (
                            self.stats.get("enc_s", 0.0) + time.perf_counter() - t_enc
                        )
                        if PROF:
                            k_ = _prof_kind(key)
                            self._prof_add(k_, "enc_s", time.perf_counter() - t_enc)
                            t0_ = self._t_enq.pop(key, None)
                            if t0_ is not None:
                                self._prof_add(k_, "qwait_s", t_enc - t0_)
                        self.sizes[key] = len(enc[key])
                        t_first[key] = time.time()
                        self.stats.setdefault("t_first_put", t_first[key])
                    data = enc[key]
                    if self._manifest_fails(key, data):
                        enc.pop(key, None)
                        t_first.pop(key, None)
                        self._done(session)
                        progressed = True
                        break
                    if self._aborted(key):
                        # Released while it was encoded or retried: never written.
                        self._drop(key)
                        enc.pop(key, None)
                        t_first.pop(key, None)
                        self._done(session)
                        progressed = True
                        break
                    inj = int(os.environ.get("OC_INJECT_ERR_EVERY", "0") or 0)
                    self._inj_n = getattr(self, "_inj_n", 0) + 1
                    if (
                        inj
                        and (
                            not key.endswith("/m")
                            or os.environ.get("OC_INJECT_INCLUDE_M") == "1"
                        )
                        and "/probe-" not in key
                        and self._inj_n % inj == 0
                    ):
                        # Test only: fail one block write so the error path is
                        # exercised.
                        self.stats["injected_errors"] = (
                            self.stats.get("injected_errors", 0) + 1
                        )
                        note_injected(key)
                        raise RuntimeError("injected write error")
                    t_call = time.perf_counter()
                    ok_ = self._try_put(key, data)
                    if PROF:
                        k_ = _prof_kind(key)
                        self._prof_add(k_, "call_s", time.perf_counter() - t_call)
                        self._prof_add(k_, "calls", 1)
                        if ok_:
                            self._prof_add(k_, "n", 1)
                            self._prof_add(k_, "bytes", len(data))
                    if not ok_:
                        if self._put_failed(t_first[key] > 0):
                            self._fail_object(key, RuntimeError("put_stall"))
                            enc.pop(key, None)
                            t_first.pop(key, None)
                            self._done(session)
                            progressed = True
                            break
                        if t_first[key] > 0:
                            self.stats["blocked_puts"] = (
                                self.stats.get("blocked_puts", 0) + 1
                            )
                            t_first[key] = -t_first[key]
                        self.stats["retries"] += 1
                        continue
                    self._stall_since = None
                    kind = (
                        "idx"
                        if key.endswith(IDX_SUFFIX)
                        else "hid"
                        if "/h/" in key
                        else "topk"
                        if is_topk_key(key)
                        else "manifest"
                        if key.endswith("/m")
                        else "r3"
                    )
                    self.stats["bytes_" + kind] = self.stats.get(
                        "bytes_" + kind, 0
                    ) + len(data)
                    self.stats["puts"] += 1
                    self.stats["bytes"] += len(data)
                    now = time.time()
                    dt = now - abs(t_first.pop(key))
                    enc.pop(key)
                    self.stats["t_last_put"] = now
                    self.stats["put_s"] = self.stats.get("put_s", 0.0) + dt
                    if dt > 0.06:
                        self.stats["blocked_s"] = self.stats.get("blocked_s", 0.0) + dt
                    self._hist[int(np.searchsorted(self._edges, dt))] += 1
                    self.stats["put_lat_max_ms"] = max(
                        self.stats.get("put_lat_max_ms", 0.0), dt * 1e3
                    )
                    if self.stats["puts"] % 64 == 0:
                        self._lat_summary()
                except (
                    Exception
                ) as e:  # never let the writer thread die; the request fails
                    self.stats["errors"] = self.stats.get("errors", 0) + 1
                    print(f"oc-writer error on {key}: {e!r}", flush=True)
                    if not self._put_raised(key, e):
                        self._forget(key)
                    enc.pop(key, None)
                    t_first.pop(key, None)
                self._done(session)
                progressed = True
                break  # back to the oldest session after every landed put
            if not progressed:
                time.sleep(0.05)

    def _put_ready_batch(self, session: str, q, enc: dict, t_first: dict) -> bool:
        """Write the run of already encoded block objects at the head of a session's
        queue in one call. Tails and manifests are written alone, after the blocks they
        list. Any failure leaves the items queued for the single-put path, which owns
        retries and error markers."""
        pb = getattr(self._store, "put_batch", None)
        if PUT_BATCH < 2 or pb is None or os.environ.get("OC_INJECT_ERR_EVERY"):
            return False
        with self._cv:
            head = list(itertools.islice(q, PUT_BATCH))
        keys, vals = [], []
        for key, payload in head:
            if (
                "/h/" not in key
                and not is_topk_key(key)
                and not (
                    key.endswith(IDX_SUFFIX)
                    or "/" not in key[len(PREFIX) + 1 + len(session) + 1 :]
                )
            ):
                break
            if self._aborted(key):
                break
            if key in enc:
                data = enc[key]
            elif isinstance(payload, concurrent.futures.Future):
                if not payload.done() or payload.exception() is not None:
                    break
                data = payload.result()
            elif callable(payload):
                break
            else:
                data = payload
            keys.append(key)
            vals.append(data)
        if len(keys) < 2:
            return False
        t0 = time.perf_counter()
        try:
            rc = pb(keys, vals) if self._cfg is None else pb(keys, vals, self._cfg)
            ok = rc == 0 if isinstance(rc, int) else all(int(x) == 0 for x in rc)
        except Exception as e:
            print(f"oc-writer batch put failed: {e!r}", flush=True)
            ok = False
        dt = time.perf_counter() - t0
        if not ok:
            self.stats["batch_put_fallback"] = (
                self.stats.get("batch_put_fallback", 0) + 1
            )
            return False
        now = time.time()
        for key, data in zip(keys, vals):
            self.sizes[key] = len(data)
            enc.pop(key, None)
            t_first.pop(key, None)
            kind = (
                "idx"
                if key.endswith(IDX_SUFFIX)
                else "hid"
                if "/h/" in key
                else "topk"
                if is_topk_key(key)
                else "r3"
            )
            self.stats["bytes_" + kind] = self.stats.get("bytes_" + kind, 0) + len(data)
            self.stats["bytes"] += len(data)
            self._hist[int(np.searchsorted(self._edges, dt / len(keys)))] += 1
            if PROF:
                k_ = _prof_kind(key)
                self._prof_add(k_, "n", 1)
                self._prof_add(k_, "calls", 1)
                self._prof_add(k_, "bytes", len(data))
                self._prof_add(k_, "call_s", dt / len(keys))
                t_q = self._t_enq.pop(key, None)
                if t_q is not None:
                    self._prof_add(k_, "qwait_s", t0 - t_q)
        self.stats["puts"] += len(keys)
        self._stall_since = None
        self.stats["batch_puts"] = self.stats.get("batch_puts", 0) + 1
        self.stats["put_s"] = self.stats.get("put_s", 0.0) + dt
        self.stats.setdefault("t_first_put", now)
        self.stats["t_last_put"] = now
        self._done(session, len(keys))
        return True

    def _forget(self, key: str) -> None:
        # A block whose write failed is never rewritten under the same name: a manifest
        # that listed it may still be read, and new bytes there (maybe of a later weight
        # version) would be read under its old version labels. The hash is poisoned
        # instead: it stays missing, and every request listing it fails.
        if not key.startswith(PREFIX + "/"):
            return
        session, _, name = key[len(PREFIX) + 1 :].partition("/")
        if "/" in name and not name.endswith(
            IDX_SUFFIX
        ):  # tails, hidden, manifests: no dedup
            return
        if name.endswith(IDX_SUFFIX):
            # A failed indexer object stays missing (its name stays in the published
            # set, so it is never rewritten); the block's R3 object and its sharing are
            # unaffected.
            with self._lock:
                self.stats["poisoned_idx"] = self.stats.get("poisoned_idx", 0) + 1
            return
        base = name
        h = base
        with self._lock:
            # Poison the exact object name: the session's first copy (the bare hash) or
            # one trajectory's own copy (hash~tag).
            self._poisoned.setdefault(session, set()).add(h)
            for (sess, _tag), has in self._track_has.items():
                if sess == session:
                    for k in [k for k, n in has.items() if n == base or k == h]:
                        del has[k]
            self.stats["poisoned"] = self.stats.get("poisoned", 0) + 1

    def _exists(self, key: str) -> bool:
        try:
            return getattr(self._store, "is_exist", lambda _k: 0)(key) == 1
        except Exception:  # noqa: BLE001
            return False

    def _put_raised(self, key: str, e: Exception) -> bool:
        """A put raised: True if the object is in the store anyway (committed, or a
        duplicate)."""
        if self._exists(key):
            self.stats["exist_after_error"] = self.stats.get("exist_after_error", 0) + 1
            return True
        self._fail_object(key, e)
        return False

    def _fail_object(self, key: str, e: Exception) -> None:
        """A write failed for good: record the exact name and why. A request fails when
        its manifest lists the name (_manifest_fails); a failed manifest marks its own
        request."""
        with self._lock:
            self._failed.setdefault(_session_of(key), {})[key] = repr(e)[:200]
        self.stats["failed_objects"] = self.stats.get("failed_objects", 0) + 1
        if key.endswith("/m"):
            self._mark_error(key, e, [key])

    def _manifest_fails(self, key: str, data) -> bool:
        """A manifest about to be written names a failed object: write the request's
        error marker instead (the reader fails the request); requests not naming one are
        unaffected."""
        if not key.endswith("/m"):
            return False
        with self._lock:
            failed = self._failed.get(_session_of(key))
            failed = dict(failed) if failed else None
        if not failed:
            return False
        try:
            names = sorted(manifest_object_keys(json.loads(bytes(data))) & set(failed))
        except (ValueError, KeyError, TypeError):
            return False
        # Another writer may have stored a name since it failed here: the store decides.
        names = [k for k in names if not self._exists(k)]
        if not names:
            return False
        self.stats["failed_requests"] = self.stats.get("failed_requests", 0) + 1
        why = {k: failed[k] for k in names[:3]}
        # No marker: write the manifest after all; the reader fails on the absent
        # object.
        return bool(
            self._mark_error(key, RuntimeError(f"names failed objects: {why}"), names)
        )

    def _mark_error(self, mkey: str, e: Exception, failed: list[str]) -> bool:
        """Marks the request whose manifest key is mkey (its own key + '!err'),
        naming the failed objects."""
        data = json.dumps(
            dict(manifest=mkey, failed=failed, error=repr(e)[:500])
        ).encode()
        for _ in range(20):  # ~1 s on the writer thread: the pool may be full right now
            try:
                if self._try_put(err_key(mkey), data):
                    return True
            except Exception as e2:
                print(f"oc-writer cannot mark error for {mkey}: {e2!r}", flush=True)
            time.sleep(0.05)
        self.stats["err_marker_lost"] = self.stats.get("err_marker_lost", 0) + 1
        return False

    @staticmethod
    def _aborted(key: str) -> bool:
        # The driver marks an aborted session with a file; its queued puts are dropped
        # instead of retried, so they neither hold the queue head nor refill the pool.
        if not (ABORT_DIR or RELEASE_DIR) or not key.startswith(PREFIX + "/"):
            return False
        session = key[len(PREFIX) + 1 :].split("/", 1)[0]
        if RELEASE_DIR and os.path.exists(os.path.join(RELEASE_DIR, session)):
            # Released: the trainer has consumed the session and deleted its prefix; a
            # late put would recreate an object nobody deletes.
            return True
        return bool(ABORT_DIR) and os.path.exists(os.path.join(ABORT_DIR, session))

    def _purge_released(self) -> int:
        """FIFO queue: drop every queued put of a released or aborted session, so an
        item stuck at the head retrying against a full pool never keeps them (and their
        session's fence) waiting."""
        if not (ABORT_DIR or RELEASE_DIR):
            return 0
        with self._q.mutex:
            keys = {_session_of(it[0]): it[0] for it in self._q.queue}
        # Marker checks (filesystem) outside the mutex, once per session per retry
        # round.
        gone_sessions = {sess for sess, key in keys.items() if self._aborted(key)}
        if not gone_sessions:
            return 0
        with self._q.mutex:
            drop = [it for it in self._q.queue if _session_of(it[0]) in gone_sessions]
            if not drop:
                return 0
            self._q.queue = collections.deque(
                it for it in self._q.queue if _session_of(it[0]) not in gone_sessions
            )
            self._q.not_full.notify_all()
        for key, _ in drop:
            self._drop(key)
            self._sess_done(_session_of(key))
            self._q.task_done()
        return len(drop)

    def _run(self) -> None:
        while True:
            key, payload = self._q.get()
            try:
                if self._aborted(key):
                    self._drop(key)
                    continue
                if isinstance(payload, concurrent.futures.Future):
                    payload = payload.result()
                if callable(payload):
                    t_enc = time.perf_counter()
                    payload = payload()
                    self.stats["enc_s"] = (
                        self.stats.get("enc_s", 0.0) + time.perf_counter() - t_enc
                    )
                if self._manifest_fails(key, payload):
                    continue
                self.sizes[key] = len(payload)
                kind = (
                    "idx"
                    if key.endswith(IDX_SUFFIX)
                    else "hid"
                    if "/h/" in key
                    else "topk"
                    if is_topk_key(key)
                    else "manifest"
                    if key.endswith("/m")
                    else "r3"
                )
                self.stats["bytes_" + kind] = self.stats.get("bytes_" + kind, 0) + len(
                    payload
                )
                t_put = time.time()
                self.stats.setdefault("t_first_put", t_put)
                first_try = True
                dropped = False
                attempts = 0
                while True:
                    if self._aborted(key):
                        # Released while it was encoded or retried: never written.
                        self._drop(key)
                        dropped = True
                        break
                    attempts += 1
                    rc = (
                        self._store.put(key, payload)
                        if self._cfg is None
                        else self._store.put(key, payload, self._cfg)
                    )
                    if rc != 0 and first_try:
                        self.stats["last_put_rc"] = int(rc)
                        self.stats["blocked_puts"] = (
                            self.stats.get("blocked_puts", 0) + 1
                        )
                    first_try = False
                    # Every TP rank publishes the same keys; a key that already exists
                    # is done.
                    if (
                        rc == 0
                        or getattr(self._store, "is_exist", lambda _k: 0)(key) == 1
                    ):
                        self._stall_since = None
                        break
                    if self._put_failed(attempts == 1):
                        self._fail_object(key, RuntimeError("put_stall"))
                        dropped = True
                        break
                    self.stats["retries"] += 1
                    self._purge_released()
                    time.sleep(0.05)
                if dropped:
                    continue
                self.stats["puts"] += 1
                self.stats["bytes"] += len(payload)
                now = time.time()
                dt = now - t_put
                self.stats["t_last_put"] = now
                self.stats["put_s"] = self.stats.get("put_s", 0.0) + dt
                if dt > 0.06:
                    self.stats["blocked_s"] = self.stats.get("blocked_s", 0.0) + dt
                self._hist[int(np.searchsorted(self._edges, dt))] += 1
                self.stats["put_lat_max_ms"] = max(
                    self.stats.get("put_lat_max_ms", 0.0), dt * 1e3
                )
                if self.stats["puts"] % 64 == 0:
                    self._lat_summary()
            except Exception as e:  # never let the writer thread die; the request fails
                self.stats["errors"] = self.stats.get("errors", 0) + 1
                print(f"oc-writer error on {key}: {e!r}", flush=True)
                if not self._put_raised(key, e):
                    self._forget(key)
            finally:
                self._sess_done(_session_of(key))
                self._q.task_done()

    def table_sizes(self) -> dict:
        with self._lock:
            return dict(
                sessions=len(self._published),
                published=sum(len(v) for v in self._published.values()),
                digest=sum(len(v) for v in self._digest.values()),
                track_has=sum(len(v) for v in self._track_has.values()),
                sizes=len(self.sizes),
                block_segs=len(getattr(self, "block_segs", {}) or {}),
                track_alt=len(getattr(self, "track_alt", {}) or {}),
                last_tail=len(getattr(self, "last_tail", {}) or {}),
                weight_versions=len(getattr(self, "weight_versions", {}) or {}),
            )

    def _prune_released(self) -> None:
        # Bookkeeping of a released session is dead: its objects are gone, and keeping
        # the dedup state would make a reused session name skip writes. Tables keyed by
        # trajectory track ("session" or "sessiontraj") and by (session, ...) are
        # pruned together.
        if not RELEASE_DIR or not os.path.isdir(RELEASE_DIR):
            return
        released = set(os.listdir(RELEASE_DIR)) - getattr(self, "_pruned", set())
        if not released:
            return

        with self._lock:
            # Hashes this session published: under P-S their base-store version records
            # go with it (P-S truncates cross-session hits, so no other session reads
            # them). Other designs share base objects across sessions; their records
            # stay until the generation goes.
            gone = set()
            if DESIGN == "ps":
                for sess in released:
                    for name in self._published.get(sess, ()):
                        gone.add(name.split("~", 1)[0].split("/", 1)[0])
            for d in (
                self._published,
                self._digest,
                self._owner,
                self._poisoned,
                self._segs,
                self._failed,
            ):
                for sess in released:
                    d.pop(sess, None)
            for k in [k for k in self._track_has if k[0] in released]:
                del self._track_has[k]
            for k in [
                k
                for k in self.sizes
                if k.startswith(PREFIX + "/")
                and k[len(PREFIX) + 1 :].split("/", 1)[0] in released
            ]:
                del self.sizes[k]
            # Tables the engine thread mutates without the lock are pruned on that
            # thread (prune_engine_tables at the start of each step).
            pend = getattr(self, "_prune_engine", None) or (set(), set())
            self._prune_engine = (pend[0] | released, pend[1] | gone)
            self._pruned = getattr(self, "_pruned", set()) | released
            self.stats["pruned_sessions"] = len(self._pruned)

    def prune_engine_tables(self) -> None:
        with self._lock:
            todo = getattr(self, "_prune_engine", None)
            if not todo:
                return
            self._prune_engine = None
        released, gone = todo

        def mine(track) -> bool:
            if isinstance(track, tuple):
                track = track[0] if track[0] else None
            return isinstance(track, str) and track.split("", 1)[0] in released

        for name in (
            "last_tail",
            "last_tail_idx",
            "weight_versions",
            "track_alt",
            "last_end",
            "cone_list",
            "cone_held",
            "cone_blocks",
            "cone_ends",
        ):
            d = getattr(self, name, None)
            if d:
                for k in [k for k in list(d) if mine(k)]:
                    d.pop(k, None)
        bs = getattr(self, "block_segs", None)
        if bs:
            for k in [
                k
                for k in list(bs)
                if k and (k[0] in released or (k[0] == "" and k[-1] in gone))
            ]:
                bs.pop(k, None)
        th = getattr(self, "tail_hist", None)
        if th:
            for k in [k for k in list(th) if mine(k[1])]:
                th.pop(k, None)

    def _prune_loop(self) -> None:
        while True:
            time.sleep(5)
            try:
                self._prune_released()
            except Exception as e:  # noqa: BLE001
                print(f"OC_PRUNE_ERROR {e!r}", flush=True)

    def _stats_loop(self) -> None:
        path = os.environ["OC_STATS_FILE"] + f".{os.getpid()}"
        while True:
            time.sleep(5)
            try:
                self._prune_released()
                self._lat_summary()
                st = dict(
                    self.stats,
                    t=time.time(),
                    qsize=self.qsize(),
                    pending=self._pending if self._fair else self._q.unfinished_tasks,
                    pending_sessions=self.pending_sessions(),
                    lat_hist=self._hist.tolist(),
                )
                if PROF:
                    st["prof"] = dict(self.prof, wall_s=time.time() - self._t_start)
                st["table_sizes"] = self.table_sizes()
                with open(path, "a") as f:
                    f.write(json.dumps(st) + "\n")
            except Exception:
                pass

    def _lat_summary(self) -> None:
        """Percentiles from the histogram: upper edge of the bin (bins are ~12%
        wide)."""
        c = np.cumsum(self._hist)
        n = int(c[-1])
        if n == 0:
            return
        for q in (50, 90, 99, 99.9):
            i = int(np.searchsorted(c, q / 100 * n))
            self.stats[f"put_lat_p{q}_ms"] = round(
                float(self._edges[min(i, len(self._edges) - 1)]) * 1e3, 4
            )

    def get(self, key: str, timeout_s: float = 30.0) -> bytes | None:
        end = time.time() + timeout_s
        while True:
            data = self._store.get(key)
            if data:
                return bytes(data)
            if time.time() > end:
                return None
            time.sleep(0.02)


_writer: Writer | None = None


def writer() -> Writer | None:
    global _writer
    if not ENABLED:
        return None
    if _writer is None:
        _writer = Writer()
    return _writer


# ---------------- extra outputs: indexer top-k and hidden states ----------------

_ZC_LOCAL = threading.local()


IDX_ZSTD = os.environ.get(
    "OC_IDX_ZSTD", "auto"
)  # "1" always, "0" never, "auto" when it pays


def idx_drop_deep(rows: np.ndarray) -> np.ndarray:
    """Partial capture: block and tail rows keep only the slots below the first trimmed
    layer. Readers take the deep slots from cone patches; stored as pad they cost
    nothing after zstd."""
    cap = IDX_CAP
    if not R3_PARTIAL or cap is None:
        return rows
    deep = [j for j, lyr in enumerate(cap.layer_ids) if lyr >= R3_FIRST_TRIMMED]
    if not deep:
        return rows
    r = np.array(rows, copy=True)
    if r.dtype == np.uint8:
        per = r.shape[1] // len(cap.layer_ids)
        r[:, deep[0] * per :] = 0
    else:
        r.reshape(len(r), len(cap.layer_ids), -1)[:, deep[0] :] = -1
    return r


def idx_shallow_width(rows) -> int:
    """Columns of a captured indexer row that blocks keep: all of them, or under partial
    capture the slots below the first trimmed layer (packed rows: bytes; int32 rows: the
    whole row)."""
    cap = IDX_CAP
    if not R3_PARTIAL or cap is None or rows.dtype != np.uint8 or rows.ndim != 2:
        return rows.shape[1] if rows.ndim >= 2 else 0
    deep = [j for j, lyr in enumerate(cap.layer_ids) if lyr >= R3_FIRST_TRIMMED]
    return (
        rows.shape[1] if not deep else deep[0] * (rows.shape[1] // len(cap.layer_ids))
    )


def idx_encode(rows: np.ndarray) -> bytes:
    """int32 rows (-1 padded) -> tag byte + zstd-3 of 20-bit packed (value + 1).
    Tag 1: pack20; tag 2: raw int32 (a value outside [-1, 2**20 - 2])."""
    import zstandard

    # A ZstdCompressor is not thread-safe: one per encoding thread.
    rows = idx_drop_deep(rows)
    _ZC = getattr(_ZC_LOCAL, "c", None)
    if _ZC is None:
        _ZC = _ZC_LOCAL.c = zstandard.ZstdCompressor(level=3)
    if (
        rows.dtype == np.uint8
    ):  # packed on the GPU already (OC_IDX_MODE=gpupack / gpudelta)
        raw = np.ascontiguousarray(rows).tobytes()
        tz, tr = (b"\x05", b"\x06") if IDX_MODE == "gpudelta" else (b"\x01", b"\x04")
        if IDX_ZSTD != "0":
            c = _ZC.compress(raw)
            if IDX_ZSTD == "1" or len(c) <= 0.9 * len(raw):
                return tz + c
        # Tag 4: packed bytes as they are. At long context the top-k positions spread
        # over the whole compressed sequence and do not compress; this saves the
        # writer's CPU.
        return tr + raw
    if IDX_MODE == "raw":
        return b"" + np.ascontiguousarray(rows, dtype=np.int32).tobytes()
    v = np.ascontiguousarray(rows, dtype=np.int64).reshape(-1) + 1
    if v.size and (v.min() < 0 or v.max() >= (1 << 20)):
        return b"\x02" + _ZC.compress(
            np.ascontiguousarray(rows, dtype=np.int32).tobytes()
        )
    if v.size % 2:
        v = np.append(v, 0)
    pairs = (v[0::2] | (v[1::2] << 20)).astype("<u8")
    return b"\x01" + _ZC.compress(pairs.view(np.uint8).reshape(-1, 8)[:, :5].tobytes())


def idx_rows_like(buf: bytes, like: np.ndarray):
    """Rows of a stored indexer object in the layout of `like`: the GPU-packed uint8
    rows (5 bytes per value pair) or int32 top-k rows."""
    import zstandard

    if like.dtype == np.uint8:
        if buf[:1] not in (
            (b"\x05", b"\x06") if IDX_MODE == "gpudelta" else (b"\x01", b"\x04")
        ):
            return None
        raw = (
            bytes(buf[1:])
            if buf[:1] in (b"\x04", b"\x06")
            else zstandard.ZstdDecompressor().decompress(buf[1:])
        )
        return np.frombuffer(raw, np.uint8).reshape(-1, *like.shape[1:])
    return idx_decode(buf, like.shape[1:])


def idx_decode(buf: bytes, shape_per_token) -> np.ndarray:
    import zstandard

    if buf[:1] == b"":
        return np.frombuffer(buf, np.int32, offset=1).reshape(-1, *shape_per_token)
    raw = (
        bytes(buf[1:])
        if buf[:1] in (b"\x04", b"\x06")
        else zstandard.ZstdDecompressor().decompress(buf[1:])
    )
    if buf[:1] in (b"\x05", b"\x06"):
        return dplane20(raw, shape_per_token)
    if buf[:1] == b"\x02":
        return np.frombuffer(raw, np.int32).reshape(-1, *shape_per_token)
    assert buf[:1] in (b"\x01", b"\x04"), buf[:1]
    return unpack20(raw, shape_per_token)


def unpack20(raw, shape_per_token) -> np.ndarray:
    """20-bit pairs (5 bytes, little endian, value + 1) -> int32 rows; one pass per
    byte."""
    per = int(np.prod(shape_per_token))
    b = np.frombuffer(raw, np.uint8)[: len(raw) // 5 * 5].reshape(-1, 5)
    out = np.empty((len(b), 2), np.int32)
    lo = b[:, 2].astype(np.int32)
    out[:, 0] = (
        b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | ((lo & 15) << 16)
    )
    out[:, 1] = (
        (lo >> 4) | (b[:, 3].astype(np.int32) << 4) | (b[:, 4].astype(np.int32) << 12)
    )
    out -= 1
    flat = out.reshape(-1)
    n = len(flat) // per * per
    return flat[:n].reshape(-1, *shape_per_token)


_DEC_POOL = None
_ZD_LOCAL = threading.local()


def _idx_payload(buf: bytes) -> bytes:
    z = getattr(_ZD_LOCAL, "d", None)
    if z is None:
        import zstandard

        z = _ZD_LOCAL.d = zstandard.ZstdDecompressor()
    return bytes(buf[1:]) if buf[:1] in (b"", b"\x06") else z.decompress(buf[1:])


def idx_decode_gpu(bufs, shape_per_token, device="cuda", threads=8):
    """Stored indexer objects -> one int32 tensor on the GPU, for a consumer that
    replays there: decompress on a pool (zstd releases the GIL), copy the packed bytes,
    unpack once."""
    import torch

    global _DEC_POOL
    delta = all(b[:1] in (b"\x05", b"\x06") for b in bufs)
    if not delta and any(b[:1] not in (b"", b"") for b in bufs):
        return torch.from_numpy(
            np.concatenate([idx_decode(b, shape_per_token) for b in bufs])
        ).to(device)
    if _DEC_POOL is None:
        _DEC_POOL = concurrent.futures.ThreadPoolExecutor(threads, "oc-dec")
    raw = b"".join(_DEC_POOL.map(_idx_payload, bufs))
    packed = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(device)
    return (
        dplane20_torch(packed, shape_per_token)
        if delta
        else unpack20_torch(packed, shape_per_token)
    )


def unpack20_torch(packed, shape_per_token):
    """The same on the GPU for a consumer that replays there: `packed` is a uint8 CUDA
    tensor of the stored bytes (H2D of 5 bytes per value pair, 37% less than int32)."""
    import torch

    per = int(np.prod(shape_per_token))
    b = packed[: packed.numel() // 5 * 5].view(-1, 5).to(torch.int32)
    out = torch.empty((b.shape[0], 2), dtype=torch.int32, device=packed.device)
    out[:, 0] = b[:, 0] | (b[:, 1] << 8) | ((b[:, 2] & 15) << 16)
    out[:, 1] = (b[:, 2] >> 4) | (b[:, 3] << 4) | (b[:, 4] << 12)
    flat = (out - 1).view(-1)
    n = flat.numel() // per * per
    return flat[:n].view(-1, *shape_per_token)


def dplane20_encode_torch(v, layers: int, topk: int):
    """GPU: v = top-k + 1 (0 = pad), [n, layers * topk], values below 2**20. Each (row,
    layer) set is sorted, differenced, and stored as a low byte plane, a middle byte
    plane and the high nibbles in pairs: 2.5 bytes per value like pack20, laid out per
    layer."""
    import torch

    n = v.shape[0]
    d = torch.sort(v.view(n, layers, topk).to(torch.int32), dim=-1).values
    d[..., 1:] -= d[..., :-1].clone()
    out = torch.empty((n, layers, topk * 5 // 2), dtype=torch.uint8, device=v.device)
    out[..., :topk] = d & 255
    out[..., topk : 2 * topk] = (d >> 8) & 255
    d >>= 16
    out[..., 2 * topk :] = d[..., 0::2] | (d[..., 1::2] << 4)
    return out.view(n, -1)


def _dplane_split(b, per_layer: int, topk: int):
    lo, mid, hi = b[..., :topk], b[..., topk : 2 * topk], b[..., 2 * topk : per_layer]
    return lo, mid, hi


def dplane20(raw, shape_per_token) -> np.ndarray:
    """Stored gpudelta rows -> int32 top-k rows (sorted ascending, -1 pad first)."""
    layers, topk = int(shape_per_token[0]), int(shape_per_token[1])
    per_layer = topk * 5 // 2
    b = np.frombuffer(raw, np.uint8)
    b = b[: len(b) // (layers * per_layer) * layers * per_layer].reshape(
        -1, layers, per_layer
    )
    lo, mid, hi = (x.astype(np.int64) for x in _dplane_split(b, per_layer, topk))
    h = np.empty(lo.shape, np.int64)
    h[..., 0::2], h[..., 1::2] = hi & 15, hi >> 4
    out = (np.cumsum(lo | (mid << 8) | (h << 16), axis=-1) - 1).astype(np.int32)
    # Sorting put the pads first; the engine emits ascending positions with pads last.
    npad = (out < 0).sum(-1, keepdims=True)
    k = out.shape[-1]
    return np.take_along_axis(out, (np.arange(k) + npad) % k, axis=-1)


def unpack_idx_rows(raw, shape_per_token, codec=None) -> np.ndarray:
    """GPU-packed uint8 rows -> int32 rows, by the codec the manifest names (one writer
    per trajectory writes all its patches with one codec)."""
    codec = IDX_CODEC if codec is None else codec
    return (
        dplane20(raw, shape_per_token)
        if codec.startswith("dplane20")
        else unpack20(raw, shape_per_token)
    )


def dplane20_torch(packed, shape_per_token):
    """The same on the GPU."""
    import torch

    layers, topk = int(shape_per_token[0]), int(shape_per_token[1])
    per_layer = topk * 5 // 2
    m = packed.numel() // (layers * per_layer) * layers * per_layer
    b = packed[:m].view(-1, layers, per_layer)
    lo, mid, hi = _dplane_split(b, per_layer, topk)
    d = lo.to(torch.int32)
    d |= mid.to(torch.int32) << 8
    hi = hi.to(torch.int32)
    d[..., 0::2] |= (hi & 15) << 16
    d[..., 1::2] |= (hi >> 4) << 16
    out = torch.cumsum(d, dim=-1, dtype=torch.int32).sub_(1)
    npad = (out < 0).sum(-1, keepdim=True)
    k = out.shape[-1]
    return torch.gather(out, -1, (torch.arange(k, device=out.device) + npad) % k)


def hid_snapshot(h):
    """Main stream: a stable copy of the lm_head input rows for the D2H copy. Returns
    (the rows' own bytes as uint8 [n, d * itemsize], dtype name, d). A dtype raw readers
    do not know (outside HID_ITEMSIZE) is stored as float32, exact for every narrower
    float, and counted as hid_upcast."""
    import torch

    name = str(h.dtype).removeprefix("torch.")
    if name in HID_ITEMSIZE:
        rows = h.detach().clone(memory_format=torch.contiguous_format)
    else:
        rows, name = (
            h.detach().to(torch.float32, memory_format=torch.contiguous_format),
            "float32",
        )
        w = writer()
        if w is not None:
            w.stats["hid_upcast"] = w.stats.get("hid_upcast", 0) + 1
    return rows.view(torch.uint8), name, int(h.shape[-1])


def topk_snapshot(logits, k: int):
    """Main stream: each logits row's top-k of the raw distribution at temperature 1, by
    the sampler's own logprob kernel (the inline logprobs' numbers). uint8 [n, 8k]: k
    int32 ids by descending logit, then their k float32 logprobs."""
    import torch

    ids = torch.topk(logits, min(int(k), logits.shape[-1]), dim=-1).indices
    try:
        from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

        lps = compute_token_logprobs(logits, ids)
    except ImportError:
        lps = torch.log_softmax(logits.to(torch.float32), dim=-1).gather(-1, ids)
    return torch.cat(
        [
            ids.to(torch.int32).view(torch.uint8),
            lps.to(torch.float32).contiguous().view(torch.uint8),
        ],
        dim=1,
    )


def hid_row_nbytes(field: dict) -> int | None:
    """Stored bytes per hidden row, from the manifest field alone. None: an encoding
    this code does not know (only codec "raw" with a known dtype)."""
    size = HID_ITEMSIZE.get(field.get("dtype")) if field.get("codec") == "raw" else None
    return None if size is None else int(field["shape"][0]) * size


def hid_check(field: dict, bufs) -> str | None:
    """Why stored hidden chunks cannot be decoded: "encoding" (unknown), "meta"
    (chunk_lens missing or not describing the chunks), or None."""
    rb = hid_row_nbytes(field)
    if rb is None:
        return "encoding"
    lens = field.get("chunk_lens")
    bufs = list(bufs)
    if (
        lens is None
        or len(lens) != field["chunks"]
        or len(bufs) != len(lens)
        or sum(lens) != field["rows"]
        or any(len(b) != k * rb for b, k in zip(bufs, lens))
    ):
        return "meta"
    return None


def topk_check(field: dict, bufs, gen_ids, lps, start: int, tol: float = 1e-5) -> dict:
    """Checks stored top-k rows against the generated tokens: meta_bad (encoding, chunk
    row counts, start or row count wrong; nothing else checked), bad_rows (repeated or
    negative id, NaN or increasing logprob, or a sampled token outside the top-k whose
    inline logprob beats the k-th), in_topk (sampled token among the ids), lp_mismatch
    (its stored logprob differs from the inline logprob `lps` by more than tol),
    lp_err_max."""
    k = int(field["shape"][0])
    rb = 8 * k
    out = dict(
        meta_bad=0,
        rows=0,
        bad_rows=0,
        in_topk=0,
        lp_checked=0,
        lp_mismatch=0,
        lp_err_max=0.0,
    )
    lens, bufs, G = field.get("chunk_lens"), list(bufs), len(gen_ids)
    if (
        field.get("codec") != "raw"
        or field.get("dtype") != "int32+float32"
        or lens is None
        or len(lens) != field["chunks"]
        or len(bufs) != len(lens)
        or sum(lens) != field["rows"]
        or any(b is None or len(b) != r * rb for b, r in zip(bufs, lens))
        or int(field["start"]) != int(start)
        or int(field["rows"]) < G
    ):
        out["meta_bad"] = 1
        return out
    rows = np.frombuffer(b"".join(bytes(b) for b in bufs), np.uint8).reshape(-1, rb)[:G]
    ids = rows[:, : 4 * k].copy().view(np.int32)
    lp = rows[:, 4 * k :].copy().view(np.float32)
    srt = np.sort(ids, axis=1)
    bad = (
        (srt[:, 1:] == srt[:, :-1]).any(1)
        | (ids < 0).any(1)
        | np.isnan(lp).any(1)
        | (np.diff(lp, axis=1) > 0).any(1)
    )
    hit = ids == np.asarray(gen_ids, np.int64)[:, None]
    inr = hit.any(1)
    out["rows"], out["in_topk"] = G, int(inr.sum())
    if len(lps) == G:
        inl = np.asarray(lps, np.float32)
        bad |= ~inr & (inl > lp[:, -1] + tol)
        stored = lp[np.arange(G), hit.argmax(1)]
        err = np.abs(stored - inl)[inr]
        out["lp_checked"], out["lp_mismatch"] = int(inr.sum()), int((err > tol).sum())
        out["lp_err_max"] = float(err.max()) if len(err) else 0.0
    out["bad_rows"] = int(bad.sum())
    return out


def hid_torch(field: dict, bufs, n: int | None = None, device="cuda"):
    """Concatenated raw hidden chunks -> torch rows in the stored dtype."""
    import torch

    rb = hid_row_nbytes(field)
    if rb is None:
        raise ValueError(f"unknown hidden encoding: {field}")
    u8 = np.frombuffer(b"".join(bufs), np.uint8).reshape(-1, rb)[:n]
    return torch.from_numpy(u8.copy()).to(device).view(getattr(torch, field["dtype"]))


class IdxCapture:
    """Copies each index-source layer's top-k rows into one persistent GPU buffer
    [layers, max_tokens, topk]. Called from the attention eager break; every op is a GPU
    op on persistent buffers, so it is also recorded into FULL CUDA graphs and replays
    there."""

    def __init__(self, layer_ids, max_tokens: int, topk: int, device) -> None:
        import torch

        self.layer_ids = sorted(layer_ids)
        self.slot = {lid: i for i, lid in enumerate(self.layer_ids)}
        self.topk = topk
        self.shape = (len(self.layer_ids), topk)
        L = len(self.layer_ids)
        self.buf = torch.full(
            (L, max_tokens, topk), -1, dtype=torch.int32, device=device
        )
        # Per-layer forward counter (advances in graph replays too); read only by the
        # dump.
        self.count = torch.zeros(L, dtype=torch.int64, device=device)
        # Rows a trimmed replay stage left without an exact result this step.
        self.invalid = torch.zeros(max_tokens, dtype=torch.bool, device=device)
        self.trimmed = False
        if DUMP_DIR:
            # Validation only: a second, separate copy of the same rows plus their
            # positions.
            self.dbuf = torch.full(
                (L, max_tokens, topk), -1, dtype=torch.int32, device=device
            )
            self.dpos = torch.full(
                (L, max_tokens), -1, dtype=torch.int64, device=device
            )

    def record(self, layer, positions) -> None:
        slot = self.slot.get(layer.layer_id)
        if slot is None or layer.topk_indices_buffer is None:
            return
        n = min(int(positions.shape[0]), self.buf.shape[1])
        src = layer.topk_indices_buffer[:n, : self.topk]
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context()
        rows = getattr(ctx, "r3_rows", None)
        if rows is not None:
            assert len(rows) == n, (len(rows), n)
            chosen = None if R3_PARTIAL else ctx.r3_exact
            exact = rows if chosen is None else rows[chosen]
            kept = src if chosen is None else src[chosen]
            self.buf[slot].fill_(-1)
            self.buf[slot, exact] = kept
            if not R3_PARTIAL:
                self.invalid.fill_(True)
                self.invalid[exact] = False
                self.trimmed = True
            self.count[slot] += 1
            if DUMP_DIR:
                self.dbuf[slot].fill_(-1)
                self.dbuf[slot, exact] = kept
                self.dpos[slot].fill_(-1)
                self.dpos[slot, exact] = (
                    positions[:n] if chosen is None else positions[:n][chosen]
                )
            return
        self.buf[slot, :n].copy_(src)
        self.count[slot] += 1
        if DUMP_DIR:
            self.dbuf[slot, :n].copy_(src)
            self.dpos[slot, :n].copy_(positions[:n])


IDX_CAP: IdxCapture | None = None
HID_ON = False
HIDDEN: list = [None]
TOPK_ON = False
TOPK: list = [None]
DUMP_LOGITS: list = []


def setup_extra(model, max_tokens: int) -> None:
    """Worker output rank only: enable indexer and hidden capture."""
    global IDX_CAP, HID_ON, TOPK_ON
    import torch

    layers = [
        m
        for m in model.modules()
        if getattr(m, "is_index_source", False)
        and getattr(m, "indexer", None) is not None
    ]
    if layers and os.environ.get("OC_EXTRA_IDX", "1") != "0":
        IDX_CAP = IdxCapture(
            [m.layer_id for m in layers],
            max_tokens,
            int(layers[0].indexer.topk_tokens),
            torch.device("cuda", torch.accelerator.current_device_index()),
        )
    HID_ON = os.environ.get("OC_EXTRA_HID", "1") != "0"
    TOPK_ON = os.environ.get("OC_EXTRA_TOPK", "0") == "1"
    if IDX_CAP is not None and (
        IDX_MODE == "gpudelta" or (R3_PARTIAL and IDX_MODE == "gpupack")
    ):
        # Packed value pairs must not straddle a layer: a patch slices rows by layer,
        # and gpudelta packs high nibbles in pairs per layer.
        assert IDX_CAP.topk % 2 == 0, "packed indexer rows need an even top-k"
    print(
        "oc extra outputs: indexer "
        f"layers={None if IDX_CAP is None else IDX_CAP.layer_ids} "
        f"hidden={HID_ON} topk={TOPK_K if TOPK_ON else 0} dump={bool(DUMP_DIR)}",
        flush=True,
    )


R3_PARTIAL = os.environ.get("OC_R3_PARTIAL", "0") == "1"
CONE_ZSTD = os.environ.get("OC_CONE_ZSTD", "1") == "1"
R3_FIRST_TRIMMED = 20
R3_LAST_LAYER = 39
R3_WINDOW = 128
IDX_NO_REPLAY = -2


def cone_size(layer: int) -> int:
    """Trailing rows of a step that come out exact at this layer; the layer past the cut
    stands in for the cut itself."""
    return R3_WINDOW + (R3_LAST_LAYER - max(layer, R3_FIRST_TRIMMED + 1)) * (
        R3_WINDOW - 1
    )


CONE_MAX = cone_size(R3_FIRST_TRIMMED)


def cone_need(layer: int) -> int:
    """Rows before a sampled token whose cell at this layer reaches it: one window step
    per later trimmed layer."""
    return 1 + (R3_LAST_LAYER - layer) * (R3_WINDOW - 1)


def cone_boundary_cut(
    aux: int,
    splice_end: int,
    capped: bool,
    bs: int,
    prev,
    cur,
    patches,
    fork: bool = False,
) -> int:
    """Row from which a request's cells replace earlier patches of its trajectory. When
    row aux completes a block an earlier patch reaches past, it stays the earlier
    computation only if this request holds the same block (same content up to aux)."""
    # A fork's list also holds the parent's later turns, which took row aux as prompt:
    # only a request that generated row aux computed it on the fork's path.
    if (aux + 1) % bs == 0 and any(
        p["end"] > aux
        and (not fork or (p.get("prompt_end") is not None and p["prompt_end"] <= aux))
        for p in patches
    ):
        b = aux // bs
        same = prev is not None and b < len(prev) and b < len(cur) and prev[b] == cur[b]
        return aux + 1 if same else aux
    return aux if capped else splice_end


class ConeParts:
    """One request's exact cells of the trimmed layers, in capture order. rows: [n,
    layers, ...] holding only those layers. Steps no longer than the window are exact on
    every layer; they arrive one token at a time and collect into one part."""

    def __init__(self, layers):
        self.layers = [int(x) for x in layers]
        # [first row, step end (None: every cell exact), arrays, end row, version,
        # replay start]
        self.parts = []

    def add(
        self, start: int, step_end: int, rows, version: int = 0, replay_start: int = 0
    ) -> None:
        n = len(rows)
        if step_end - start <= R3_WINDOW:
            last = self.parts[-1] if self.parts else None
            # Short steps keep a view: the step's host arrays are not written again.
            if (
                last is not None
                and last[1] is None
                and last[3] == start
                and last[4] == version
                and last[5] == replay_start
            ):
                last[2].append(rows)
                last[3] = start + n
            else:
                self.parts.append(
                    [start, None, [rows], start + n, version, replay_start]
                )
            return
        first = max(start, step_end - CONE_MAX)
        if any(
            p[1] == step_end
            and p[5] == replay_start
            and p[0] <= first
            and p[3] >= start + n
            for p in self.parts
        ):
            # A recompute after a preemption of cells this request already holds: first
            # wins.
            return
        if first < start + n:
            kept = np.array(rows[first - start :], copy=True)
            self.parts.append(
                [first, step_end, [kept], first + len(kept), version, replay_start]
            )

    def trim(self, floor: int) -> None:
        parts = []
        for p in self.parts:
            if p[3] <= floor:
                continue
            if p[1] is None and p[0] <= floor - 2 * R3_WINDOW:
                rows = np.concatenate(p[2])
                p[2], p[0] = [np.array(rows[floor - p[0] :], copy=True)], floor
            parts.append(p)
        self.parts = parts

    def _cells(self, part, layer: int, lo: int, hi: int):
        first, step_end = part[0], part[1]
        a = max(
            lo, first if step_end is None else max(first, step_end - cone_size(layer))
        )
        if part[5] > 0:
            # A replay that starts inside the context reaches a cell only after this
            # many rows.
            a = max(a, part[5] + (layer + 1) * (R3_WINDOW - 1))
        return a, min(part[3], hi)

    def cover(self, total: int):
        """(lo, hi, cells held by this request), or None."""
        if not self.parts:
            return None
        lo, hi = (
            min(p[0] for p in self.parts),
            min(max(p[3] for p in self.parts), total),
        )
        if lo >= hi:
            return None
        cov = np.zeros((hi - lo, len(self.layers)), bool)
        for p in self.parts:
            for j, layer in enumerate(self.layers):
                a, b = self._cells(p, layer, lo, hi)
                if a < b:
                    cov[a - lo : b - lo, j] = True
        return lo, hi, cov

    def encode(self, total: int, keep) -> bytes:
        """First computation wins inside the request. keep: the cells to store (others
        are held by earlier patches of the trajectory)."""
        lo, hi, _ = self.cover(total)
        like = self.parts[0][2][0]
        versions = []
        for p in self.parts:
            if p[4] not in versions:
                versions.append(p[4])
        final = keep
        any_ = np.flatnonzero(final.any(axis=1))
        k0, k1 = (lo + int(any_[0]), lo + int(any_[-1]) + 1) if len(any_) else (lo, lo)
        # Only rows with a kept cell are materialized; first computation wins inside
        # them.
        data = np.zeros((k1 - k0, *like.shape[1:]), like.dtype)
        own = np.zeros(
            (k1 - k0, len(self.layers)), np.uint8 if len(versions) <= 256 else np.uint16
        )
        done = np.zeros((k1 - k0, len(self.layers)), bool)
        tail = (1,) * (like.ndim - 2)
        for p in self.parts:
            if p[0] >= k1 or p[3] <= k0:
                continue
            rows = None
            v = versions.index(p[4])
            for j, layer in enumerate(self.layers):
                a, b = self._cells(p, layer, lo, hi)
                a, b = max(a, k0), min(b, k1)
                if a >= b:
                    continue
                if rows is None:
                    rows = p[2][0] if len(p[2]) == 1 else np.concatenate(p[2])
                d = done[a - k0 : b - k0, j]
                src_ = rows[a - p[0] : b - p[0], j]
                if not d.any():
                    data[a - k0 : b - k0, j] = src_
                    own[a - k0 : b - k0, j] = v
                elif not d.all():
                    free = ~d
                    np.copyto(
                        data[a - k0 : b - k0, j], src_, where=free.reshape(-1, *tail)
                    )
                    np.copyto(own[a - k0 : b - k0, j], v, where=free)
                d[:] = True
        fk = final[k0 - lo : k1 - lo]
        body = np.packbits(final).tobytes() + data[fk].tobytes()
        if len(versions) > 1:
            body += own[fk].tobytes()
        meta = dict(
            lo=int(lo),
            hi=int(hi),
            layers=self.layers,
            dtype=str(data.dtype),
            shape=[hi - lo, *data.shape[1:]],
            cov=list(final.shape),
            versions=versions,
        )
        if CONE_ZSTD and len(body) > 4096:
            import zstandard

            z = getattr(_ZC_LOCAL, "c", None)
            if z is None:
                z = _ZC_LOCAL.c = zstandard.ZstdCompressor(level=3)
            c = z.compress(body)
            if len(c) < len(body):
                body, meta["zstd"] = c, 1
        head = json.dumps(meta).encode()
        return len(head).to_bytes(4, "little") + head + body


def cone_decode(buf):
    k = int.from_bytes(buf[:4], "little")
    h = json.loads(bytes(buf[4 : 4 + k]))
    if h.get("zstd"):
        import zstandard

        buf = bytes(buf[: 4 + k]) + zstandard.ZstdDecompressor().decompress(
            bytes(buf[4 + k :])
        )
    nc = int(np.prod(h["cov"]))
    packed = (nc + 7) // 8
    cov = (
        np.unpackbits(np.frombuffer(buf, np.uint8, packed, 4 + k), count=nc)
        .astype(bool)
        .reshape(h["cov"])
    )
    data = np.zeros(h["shape"], h["dtype"])
    cell = int(np.prod(h["shape"][2:])) * data.dtype.itemsize
    cells = int(cov.sum())
    at = 4 + k + packed
    data[cov] = np.frombuffer(
        buf, h["dtype"], cells * cell // data.dtype.itemsize, at
    ).reshape(-1, *h["shape"][2:])
    own = np.zeros(h["cov"], np.int64)
    versions = h["versions"]
    if len(versions) > 1:
        own[cov] = np.asarray(versions, np.int64)[
            np.frombuffer(
                buf,
                np.uint8 if len(versions) <= 256 else np.uint16,
                cells,
                at + cells * cell,
            )
        ]
    else:
        own[cov] = versions[0]
    return h, data, cov, own


def overlay_cones(rows, first_deep_slot, no_replay, patches, unpack=None):
    """rows: [n, slots, ...] as stored for the blocks. Deep slots become `no_replay`;
    patches, each (bytes, valid end), in request order, fill cells nobody filled yet.
    Returns the rows, the covered mask and each covered cell's weight version."""
    out = np.array(rows, copy=True)
    n = len(out)
    out[:, first_deep_slot:] = no_replay
    filled = np.zeros((n, out.shape[1] - first_deep_slot), bool)
    ver = np.full(filled.shape, -1, np.int64)
    for buf, end in patches:
        h, data, cov, own = cone_decode(buf)
        if unpack is not None:
            data = unpack(data)
        lo, hi = h["lo"], min(h["hi"], n, end)
        if lo >= hi:
            continue
        take = cov[: hi - lo] & ~filled[lo:hi]
        out[lo:hi, first_deep_slot:][take] = data[: hi - lo][take]
        ver[lo:hi][take] = own[: hi - lo][take]
        filled[lo:hi] |= take
    return out, filled, ver


def cone_mask(n: int, ends, layers) -> np.ndarray:
    """Cells a trajectory's sampled tokens depend on: rows [n, layers] inside each
    prompt's cone or past the prompt. ends: (prompt rows, rows at its end) per request.
    """
    mask = np.zeros((n, len(layers)), bool)
    for p, e in ends:
        for j, layer in enumerate(layers):
            mask[max(0, p - cone_need(layer)) : min(e, n), j] = True
    return mask


def r3_has_sentinel(rows) -> bool:
    """Rows a trimmed replay stage did not compute exactly. One byte ids have no spare
    value: those models never trim."""
    if rows.dtype.itemsize < 2:
        return False
    if R3_PARTIAL:
        # Readers ignore the trimmed layers of a block; a sentinel below them is a row
        # never computed.
        rows = rows[:, :R3_FIRST_TRIMMED]
    return bool((rows == np.iinfo(rows.dtype).max).any())


def step_begin() -> None:
    DUMP_LOGITS.clear()
    HIDDEN[0] = None
    TOPK[0] = None


# ---------------- scheduler-side hit policy ----------------


class HitPolicy:
    """Prefix-hit cap per design, evaluated in the scheduler process."""

    def __init__(self) -> None:
        self.session_blocks: dict[str, set[bytes]] = {}
        self.hbs = 16
        self.capped = 0
        self.cap_tokens_lost = 0

    def note_blocks(self, session: str | None, hashes) -> None:
        if session is not None:
            self.session_blocks.setdefault(session, set()).update(
                bytes(h) for h in hashes
            )

    def cap(self, request, max_len: int, hash_block_size: int) -> int:
        sp = request.sampling_params
        if sp is None:
            return max_len
        if DESIGN == "c0":
            cap = min(max_len, sp.routed_experts_prompt_start)
        elif DESIGN == "ps":
            session = (sp.extra_args or {}).get("oc_session")
            have = self.session_blocks.get(session, ()) if session else ()
            n = 0
            for h in request.block_hashes:
                if bytes(h) not in have:
                    break
                n += 1
            cap = min(max_len, max(sp.routed_experts_prompt_start, n * hash_block_size))
        else:
            return max_len
        if cap < max_len:
            self.capped += 1
        return cap


POLICY = HitPolicy()


def dump_stats(path: str) -> None:
    w = _writer
    out = {
        "design": DESIGN,
        "capped": POLICY.capped,
        "writer": None if w is None else w.stats,
    }
    with open(path, "a") as f:
        f.write(json.dumps(out) + "\n")


def rows_from_blocks(
    blocks: list[bytes], tail: bytes, dtype, shape_per_token, total_rows: int
):
    arr = np.frombuffer(b"".join(blocks) + tail, dtype=dtype).reshape(
        (-1, *shape_per_token)
    )
    return arr[:total_rows]
