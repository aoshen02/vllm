# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for patch_swa_snapshot.py against a copy of the image's vllm sources.

Runs on a bare python3 (no torch, no pytest): the methods under test are cut out of the
real source files with ast and executed inside small stand-in classes. The SWA manager,
the coordinator's caching/free path, KVCacheManager.cache_final_snapshot, the
scheduler's _free_request_blocks, BlockPool.cache_full_blocks and SlidingWindowSpec
admission are the real code; the block pool's storage, requests and specs are fakes.

    python3 test_swa_snapshot.py <vllm dir>      (patched copy: all pass; pristine:
    fails) VLLM_SRC=<vllm dir> pytest test_swa_snapshot.py
"""

import ast
import itertools
import os
import pathlib
import sys
import traceback
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

ENV = "VLLM_SWA_DECODE_SNAPSHOT"
WINDOW, SWA_BLOCK, ALIGN = 128, 32, 128

try:
    import pytest

    parametrize = pytest.mark.parametrize
except ImportError:
    pytest = None

    def parametrize(names, cases, ids=None):
        def wrap(fn):
            fn._params = (names, cases, ids)
            return fn

        return wrap


def src_root():
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        return sys.argv[1]
    return os.environ["VLLM_SRC"]


def _class_methods(path, cls_name, names=None):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            out = []
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                if names is not None and item.name not in names:
                    continue
                first = min([d.lineno for d in item.decorator_list] + [item.lineno])
                out.append(NL.join(lines[first - 1 : item.end_lineno]))
            return out
    raise KeyError(cls_name)


def _function(path, name):
    text = pathlib.Path(path).read_text(encoding="utf-8")
    lines = text.splitlines()
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return NL.join(lines[node.lineno - 1 : node.end_lineno])
    raise KeyError(name)


NL = chr(10)


class Block:
    _ids = itertools.count(1)

    def __init__(self, null=False):
        self.block_id = 0 if null else next(Block._ids)
        self.ref_cnt = 0
        self.block_hash = None
        self.block_hash_num_tokens = None
        self.is_null = null

    def set_block_hash(self, key, num_tokens=None):
        self.block_hash = key
        self.block_hash_num_tokens = num_tokens

    def reset_hash(self):
        self.block_hash = None
        self.block_hash_num_tokens = None

    def __repr__(self):
        return (
            "NULL"
            if self.is_null
            else f"B{self.block_id}{'#' if self.block_hash else ''}"
        )


class AttentionSpec:
    pass


class CircularBufferSpec:
    pass


@dataclass(frozen=True)
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int = WINDOW
    block_size: int = SWA_BLOCK
    extra_retained_tokens: int = 0
    dcp_sharded: bool = False
    prefix_cacheable: bool = True
    prefix_replay_tokens: int = 0


class FakeRequest:
    def __init__(self, rid, tokens, num_prompt_tokens=None):
        self.request_id = rid
        self.tokens = list(tokens)
        self.num_prompt_tokens = (
            len(tokens) if num_prompt_tokens is None else num_prompt_tokens
        )
        self.num_computed_tokens = 0
        self.num_in_flight_tokens = 0
        self.shared_prefix_boundary = 0
        self.block_hashes = []
        self._rehash()

    @property
    def num_tokens(self):
        return len(self.tokens)

    def _rehash(self):
        while (len(self.block_hashes) + 1) * SWA_BLOCK <= len(self.tokens):
            i = len(self.block_hashes)
            parent = self.block_hashes[-1] if self.block_hashes else None
            chunk = tuple(self.tokens[i * SWA_BLOCK : (i + 1) * SWA_BLOCK])
            self.block_hashes.append(hash((parent, chunk)))

    def append(self, token):
        self.tokens.append(token)
        self._rehash()


def _cdiv(a, b):
    return -(a // -b)


def _globals():
    def trim_null_head(blocks):
        n = 0
        while n < len(blocks) and blocks[n].is_null:
            n += 1
        return blocks[n:]

    return {
        "__name__": "swa_under_test",
        "cdiv": _cdiv,
        "os": os,
        "itertools": itertools,
        "defaultdict": defaultdict,
        "Sequence": Sequence,
        "AttentionSpec": AttentionSpec,
        "CircularBufferSpec": CircularBufferSpec,
        "SlidingWindowSpec": SlidingWindowSpec,
        "resolve_block_hashes": lambda hashes, *a, **k: hashes,
        "trim_null_head": trim_null_head,
        "make_block_hash_with_group_id": lambda h, g: (h, g),
        "maybe_convert_block_hash": lambda h: h,
    }


POOL_FAKE = """
class BlockPool:
    def __init__(self):
        self.null_block = Block(null=True)
        self.hash_block_size = SWA_BLOCK
        self.enable_kv_cache_events = False
        self.cache = {}
        self.cached_block_hashes_by_block = {}
        self.freed = []

    def get_new_blocks(self, n):
        out = [Block() for _ in range(n)]
        for b in out:
            b.ref_cnt = 1
        return out

    def touch(self, blocks):
        for b in blocks:
            if not b.is_null:
                b.ref_cnt += 1

    def free_blocks(self, blocks):
        for b in blocks:
            if b.is_null:
                continue
            b.ref_cnt -= 1
            self.freed.append(b)

    def get_cached_block(self, block_hash, group_ids):
        out = []
        for g in group_ids:
            b = self.cache.get((block_hash, g))
            if b is None:
                return None
            out.append(b)
        return out

    def _remove_cached_block_hashes(self, block):
        raise AssertionError("unexpected re-hash of " + repr(block))

    def _emit_block_removed_events(self, hashes):
        pass

    def _insert_block_hash(self, key, block, num_tokens):
        if block.block_hash is None:
            block.set_block_hash(key, num_tokens=num_tokens)
        self.cache[key] = block

{cache_full_blocks}
"""

SWA_TEMPLATE = """
class Base:
    supports_fine_grained_hash_lookup = False
    retains_longer_hit = False

    def _has_partial_local_hit(self, *a):
        return False

{base}


class SlidingWindowManager(Base):
{swa}


class Coordinator:
    def __init__(self, managers, enable_caching=True):
        self.single_type_managers = managers
        self.scheduler_block_size = ALIGN
        self.eagle_group_ids = set()
        self.retention_interval = 0
        self.num_reprefillable_tokens = 0
        self.swa_decode_snapshot = enable_caching and os.environ.get(ENV, "0") == "1"

{coord}


class KVCacheManager:
    def __init__(self, coordinator, enable_caching=True):
        self.coordinator = coordinator
        self.enable_caching = enable_caching

{kvm}


class Scheduler:
    def __init__(self, kv_cache_manager):
        self.kv_cache_manager = kv_cache_manager
        self.defer_block_free = False
        self.deferred_frees = []

{sched}
"""


def build(root):
    stm = f"{root}/v1/core/single_type_kv_cache_manager.py"
    g = _globals()
    g.update(Block=Block, SWA_BLOCK=SWA_BLOCK, ALIGN=ALIGN, ENV=ENV)
    pool_src = POOL_FAKE.replace(
        "{cache_full_blocks}",
        NL.join(
            _class_methods(
                f"{root}/v1/core/block_pool.py", "BlockPool", ["cache_full_blocks"]
            )
        ),
    )
    base = _class_methods(
        stm,
        "SingleTypeKVCacheManager",
        [
            "__init__",
            "cache_blocks",
            "_remove_blocks_in_range",
            "remove_skipped_blocks",
            "pop_blocks_for_free",
            "free",
            "get_num_hit_skipped_tokens",
            "get_num_blocks_to_allocate",
        ],
    )
    coord_path = f"{root}/v1/core/kv_cache_coordinator.py"
    coord = _class_methods(
        coord_path,
        "KVCacheCoordinator",
        [
            "cache_blocks",
            "get_num_cacheable_tokens",
            "free",
            "remove_skipped_blocks",
            "cache_final_snapshot",
            "_validate_swa_decode_snapshot",
        ],
    )
    kvm = _class_methods(
        f"{root}/v1/core/kv_cache_manager.py",
        "KVCacheManager",
        ["free", "cache_final_snapshot"],
    )
    sched = _class_methods(
        f"{root}/v1/core/sched/scheduler.py",
        "Scheduler",
        ["_free_request_blocks", "_request_blocks_can_be_freed"],
    )
    module = NL.join(
        [
            "from __future__ import annotations",
            pool_src,
            _function(coord_path, "replay_boundaries"),
            SWA_TEMPLATE.format(
                base=NL.join(base),
                swa=NL.join(_class_methods(stm, "SlidingWindowManager")),
                coord=NL.join(coord),
                kvm=NL.join(kvm),
                sched=NL.join(sched),
            ),
        ]
    )
    exec(compile(module, "swa_under_test", "exec"), g)
    return g


def _admission_method(root):
    g = _globals()
    src = NL.join(
        _class_methods(
            f"{root}/v1/kv_cache_interface.py",
            "SlidingWindowSpec",
            ["max_admission_blocks_per_request"],
        )
    )
    exec(
        compile(
            "from __future__ import annotations" + NL + "class S:" + NL + src,
            "spec",
            "exec",
        ),
        g,
    )
    return g["S"].max_admission_blocks_per_request


class Env:
    def __init__(self, on):
        self.on = on

    def __enter__(self):
        self.old = os.environ.get(ENV)
        os.environ[ENV] = "1" if self.on else "0"

    def __exit__(self, *a):
        if self.old is None:
            os.environ.pop(ENV, None)
        else:
            os.environ[ENV] = self.old


class Engine:
    """One SWA group driven the way allocate_slots / the scheduler drive it."""

    def __init__(self, on=True, spec=None):
        self.g = build(src_root())
        with Env(on):
            self.pool = self.g["BlockPool"]()
            self.mgr = self.g["SlidingWindowManager"](
                spec or SlidingWindowSpec(),
                block_pool=self.pool,
                enable_caching=True,
                kv_cache_group_id=0,
                scheduler_block_size=ALIGN,
            )
            self.coord = self.g["Coordinator"]([self.mgr])
        self.kvm = self.g["KVCacheManager"](self.coord)
        self.sched = self.g["Scheduler"](self.kvm)

    def lookup(self, req):
        _, hit = self.mgr.find_longest_cache_hit(
            req.block_hashes,
            req.num_tokens - 1,
            [0],
            self.pool,
            self.mgr.kv_cache_spec,
            False,
            ALIGN,
        )
        return hit

    def admit(self, req):
        hit = self.lookup(req)
        if hit:
            blocks, _ = self.mgr.find_longest_cache_hit(
                req.block_hashes,
                req.num_tokens - 1,
                [0],
                self.pool,
                self.mgr.kv_cache_spec,
                False,
                ALIGN,
            )
            self.pool.touch(blocks[0])
            self.mgr.req_to_blocks[req.request_id] = list(blocks[0])
            self.mgr.num_cached_block[req.request_id] = len(blocks[0])
        req.num_computed_tokens = hit
        return hit

    def schedule(self, req, num_new):
        rid = req.request_id
        self.coord.remove_skipped_blocks(
            rid, req.num_computed_tokens - req.num_in_flight_tokens
        )
        blocks = self.mgr.req_to_blocks[rid]
        need = _cdiv(req.num_computed_tokens + num_new, SWA_BLOCK)
        blocks.extend(self.pool.get_new_blocks(max(0, need - len(blocks))))
        self.coord.cache_blocks(
            req, min(req.num_computed_tokens + num_new, req.num_tokens)
        )
        req.num_computed_tokens += num_new

    def run(self, req, num_output, chunk=512):
        """Prefill in chunks, then decode num_output tokens."""
        self.admit(req)
        while req.num_computed_tokens < req.num_tokens:
            self.schedule(req, min(chunk, req.num_tokens - req.num_computed_tokens))
        req.append(10_000_000 + req.num_tokens)
        for _ in range(num_output - 1):
            self.schedule(req, 1)
            req.append(10_000_000 + req.num_tokens)

    def live_blocks(self, req):
        return sum(not b.is_null for b in self.mgr.req_to_blocks[req.request_id])


def _prompt(n, salt=0):
    return [salt * 1_000_000 + i for i in range(n)]


class TestDecodeEndSnapshot:
    @parametrize(
        "prompt_len,num_output",
        [(1000, 300), (1000, 40), (777, 1), (1024, 512), (100, 900)],
        ids=[
            "long-decode",
            "decode-crosses-one-boundary",
            "single-token",
            "aligned-prompt",
            "short-prompt",
        ],
    )
    def test_next_turn_hits_decode_end(self, prompt_len, num_output):
        eng = Engine(on=True)
        r1 = FakeRequest("t1", _prompt(prompt_len))
        eng.run(r1, num_output)
        n1 = r1.num_tokens
        eng.sched._free_request_blocks(r1)
        r2 = FakeRequest("t2", r1.tokens + _prompt(200, salt=7))
        assert eng.lookup(r2) == (n1 - 1) // ALIGN * ALIGN

    @parametrize(
        "prompt_len,num_output",
        [(1000, 300), (1000, 40)],
        ids=["long-decode", "one-boundary"],
    )
    def test_env_off_hits_only_prompt_end(self, prompt_len, num_output):
        eng = Engine(on=False)
        r1 = FakeRequest("t1", _prompt(prompt_len))
        eng.run(r1, num_output)
        eng.sched._free_request_blocks(r1)
        r2 = FakeRequest("t2", r1.tokens + _prompt(200, salt=7))
        assert eng.lookup(r2) == (prompt_len - 1) // ALIGN * ALIGN

    def test_resend_still_hits_prompt_end(self):
        eng = Engine(on=True)
        r1 = FakeRequest("t1", _prompt(1000))
        eng.run(r1, 300)
        eng.sched._free_request_blocks(r1)
        assert eng.lookup(FakeRequest("again", _prompt(1000))) == 896

    def test_preemption_snapshot_resumes_there(self):
        eng = Engine(on=True)
        r1 = FakeRequest("t1", _prompt(1000))
        eng.run(r1, 200)
        committed = r1.num_computed_tokens
        eng.sched._free_request_blocks(r1)
        r1.num_computed_tokens = 0
        assert eng.admit(r1) == committed // ALIGN * ALIGN

    def test_hold_bounds_live_blocks(self):
        eng = Engine(on=True)
        off = Engine(on=False)
        peak = peak_off = 0
        r, r_off = FakeRequest("a", _prompt(1000)), FakeRequest("a", _prompt(1000))
        for e, req in ((eng, r), (off, r_off)):
            e.admit(req)
            e.schedule(req, req.num_tokens)
            req.append(1)
        for _ in range(400):
            for e, req in ((eng, r), (off, r_off)):
                e.schedule(req, 1)
                req.append(5)
            peak = max(peak, eng.live_blocks(r))
            peak_off = max(peak_off, off.live_blocks(r_off))
        hold = ALIGN + WINDOW - WINDOW
        assert peak_off < peak <= peak_off + hold // SWA_BLOCK

    @parametrize(
        "num_tokens,num_computed,in_flight,expected_point",
        [
            (1300, 1299, 0, 1280),
            (1300, 1300, 1, 1280),
            (1281, 1280, 0, 1280),
            (1281, 1281, 2, 1152),
        ],
        ids=[
            "sync-finish",
            "async-finish",
            "boundary-commit",
            "in-flight-not-committed",
        ],
    )
    def test_committed_point(self, num_tokens, num_computed, in_flight, expected_point):
        eng = Engine(on=True)
        req = FakeRequest("c", _prompt(num_tokens))
        blocks = eng.pool.get_new_blocks(_cdiv(num_tokens, SWA_BLOCK))
        eng.mgr.req_to_blocks["c"] = blocks
        eng.mgr.num_cached_block["c"] = len(blocks)
        req.num_computed_tokens, req.num_in_flight_tokens = num_computed, in_flight
        eng.kvm.cache_final_snapshot(req)
        hashed = [i for i, b in enumerate(blocks) if b.block_hash is not None]
        end = expected_point // SWA_BLOCK
        assert hashed == list(range(end - WINDOW // SWA_BLOCK, end))

    def test_uncommitted_remote_load_is_not_hashed(self):
        eng = Engine(on=True)
        req = FakeRequest("r", _prompt(2000))
        blocks = eng.pool.get_new_blocks(_cdiv(2000, SWA_BLOCK))
        eng.mgr.req_to_blocks["r"] = blocks
        req.num_computed_tokens = 1999
        eng.sched._free_request_blocks(req)
        assert all(b.block_hash is None for b in blocks)

    def test_incomplete_tail_is_not_hashed(self):
        eng = Engine(on=True)
        req = FakeRequest("n", _prompt(1300))
        blocks = eng.pool.get_new_blocks(_cdiv(1300, SWA_BLOCK))
        blocks[37] = eng.pool.null_block
        eng.mgr.req_to_blocks["n"] = blocks
        eng.mgr.num_cached_block["n"] = len(blocks)
        req.num_computed_tokens = 1299
        eng.kvm.cache_final_snapshot(req)
        assert all(b.is_null or b.block_hash is None for b in blocks)

    def test_snapshot_precedes_free(self):
        eng = Engine(on=True)
        r1 = FakeRequest("t1", _prompt(1000))
        eng.run(r1, 300)
        end = (r1.num_tokens - 1) // ALIGN * ALIGN // SWA_BLOCK
        tail = eng.mgr.req_to_blocks["t1"][end - WINDOW // SWA_BLOCK : end]
        eng.sched._free_request_blocks(r1)
        assert all(not b.is_null and b.block_hash is not None for b in tail), tail
        assert all(b in eng.pool.freed and b.ref_cnt == 0 for b in tail)
        assert "t1" not in eng.mgr.req_to_blocks


class TestUnchangedWhenOff:
    @parametrize("computed", [0, 127, 128, 200, 1000, 1279], ids=str)
    def test_skipped_tokens_match_window(self, computed):
        eng = Engine(on=False)
        assert eng.mgr.get_num_skipped_tokens(computed) == max(0, computed - WINDOW + 1)

    @parametrize(
        "computed,expected",
        [(127, 0), (200, 0), (255, 0), (256, 128), (1000, 768), (1279, 1024)],
        ids=str,
    )
    def test_skipped_tokens_hold_when_on(self, computed, expected):
        eng = Engine(on=True)
        assert eng.mgr.get_num_skipped_tokens(computed) == expected

    @parametrize("computed", [128, 1000, 1024], ids=str)
    def test_hit_allocation_unchanged(self, computed):
        on, off = Engine(on=True), Engine(on=False)
        assert on.mgr.get_num_hit_skipped_tokens(
            computed
        ) == off.mgr.get_num_hit_skipped_tokens(computed)


class TestAdmissionAndValidation:
    def test_admission_cap_covers_hold(self):
        method = _admission_method(src_root())
        spec = SlidingWindowSpec()
        with Env(False):
            base = method(spec, max_in_flight_tokens=8192, max_model_len=1 << 20)
        with Env(True):
            held = method(spec, max_in_flight_tokens=8192, max_model_len=1 << 20)
        assert held - base == 2 * WINDOW // SWA_BLOCK

    @parametrize(
        "change,message",
        [
            ({"eagle": True}, "speculative"),
            ({"lookahead": 2}, "speculative"),
            ({"retention": None}, "retention"),
            ({"retention": 512}, "retention"),
            ({"cacheable": False}, "prefix-cacheable"),
            ({"extra": 2}, "sliding-window group"),
        ],
        ids=[
            "eagle",
            "mtp-lookahead",
            "dense-retention",
            "interval-retention",
            "bounded-replay",
            "extra-retained",
        ],
    )
    def test_refused_configs(self, change, message):
        spec = SlidingWindowSpec(
            prefix_cacheable=change.get("cacheable", True),
            extra_retained_tokens=change.get("extra", 0),
        )
        eng = Engine(on=True, spec=spec)
        eng.coord.retention_interval = change.get("retention", 0)
        try:
            eng.coord._validate_swa_decode_snapshot(
                change.get("eagle", False), change.get("lookahead", 0)
            )
        except ValueError as e:
            assert message in str(e), e
        else:
            raise AssertionError("not refused")

    def test_supported_config_accepted(self):
        eng = Engine(on=True)
        eng.coord._validate_swa_decode_snapshot(False, 0)


def _cases():
    for cls in (
        TestDecodeEndSnapshot,
        TestUnchangedWhenOff,
        TestAdmissionAndValidation,
    ):
        for name in sorted(n for n in vars(cls) if n.startswith("test_")):
            fn = getattr(cls, name)
            params = getattr(fn, "_params", None)
            if params is None:
                yield f"{cls.__name__}::{name}", cls, fn, {}
                continue
            names, cases, ids = params
            keys = [k.strip() for k in names.split(",")]
            for i, case in enumerate(cases):
                case = case if isinstance(case, tuple) else (case,)
                cid = ids(case[0]) if callable(ids) else (ids[i] if ids else str(i))
                yield f"{cls.__name__}::{name}[{cid}]", cls, fn, dict(zip(keys, case))


if __name__ == "__main__":
    passed = failed = 0
    for label, cls, fn, kwargs in _cases():
        try:
            fn(cls(), **kwargs)
            passed += 1
            print("PASS", label)
        except Exception as e:
            failed += 1
            last = traceback.extract_tb(e.__traceback__)[-1]
            print(
                "FAIL",
                label,
                f"{type(e).__name__}: {str(e)[:120]}",
                f"(line {last.lineno})",
            )
    print(f"{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
