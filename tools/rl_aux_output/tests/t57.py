# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57d: SSD dir ownership, SSD + hard pin refusal, admission from free SSD bytes,
release order, get backoff, helper-process polling, and the x10 driver's wiring of them.
Pure logic with fakes (no Mooncake, no engine). Run: python -I t57.py"""

import json
import os
import pathlib
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pylib"))
for k in (
    "OC_MC_SSD_PATH",
    "OC_MC_SSD_WIPE",
    "OC_HARD_PIN",
    "SLURM_JOB_ID",
    "OC_RELEASE_DIR",
):
    os.environ.pop(k, None)
os.environ["OC_MC_MASTER"] = "fake:1"

SETUPS = []


class FakeMC:
    def setup(self, *a):
        SETUPS.append(a)
        return 0

    def put(self, *a):
        return 0

    def get(self, k):
        return None


sys.modules["mooncake"] = types.ModuleType("mooncake")
sys.modules["mooncake.store"] = types.SimpleNamespace(
    MooncakeDistributedStore=FakeMC, ReplicateConfig=object
)

import oc_common as oc  # noqa: E402

TMP = tempfile.mkdtemp(prefix="t57_")


def env(**kw):
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def t_ssd_dir_ownership():
    env(OC_HARD_PIN="0", SLURM_JOB_ID="111")
    # New dir: created and marked by make_store.
    new = os.path.join(TMP, "new")
    env(OC_MC_SSD_PATH=new, OC_MC_SSD_WIPE=None)
    oc.make_store(segment_bytes=1 << 20)
    assert (
        json.loads(pathlib.Path(os.path.join(new, oc.SSD_OWNER)).read_text())["job"]
        == "111"
    )
    assert SETUPS[-1][0]["ssd_offload_path"] == new
    # A foreign non-empty tree is never wiped, even with OC_MC_SSD_WIPE=1.
    foreign = os.path.join(TMP, "shared_nvme")
    os.makedirs(foreign)
    pathlib.Path(os.path.join(foreign, "precious")).write_text("x")
    env(OC_MC_SSD_PATH=foreign, OC_MC_SSD_WIPE="1")
    try:
        oc.make_store(segment_bytes=1 << 20)
        raise AssertionError("wiped a dir it does not own")
    except RuntimeError as e:
        assert "not wiping" in str(e), e
    assert os.path.exists(os.path.join(foreign, "precious"))
    # Our own dir from an earlier holder of this job: wiped on request, marked again.
    pathlib.Path(os.path.join(new, "bucket0")).write_text("old")
    env(OC_MC_SSD_PATH=new, OC_MC_SSD_WIPE=None)
    try:
        oc.make_store(segment_bytes=1 << 20)
        raise AssertionError("reused a non-empty dir without OC_MC_SSD_WIPE")
    except RuntimeError:
        pass
    env(OC_MC_SSD_WIPE="1")
    oc.make_store(segment_bytes=1 << 20)
    assert sorted(os.listdir(new)) == [oc.SSD_OWNER]
    # finish(): only a dir make_store created for this job is removed.
    env(SLURM_JOB_ID="222")
    assert not oc.ssd_remove_owned(new) and os.path.isdir(new)
    env(SLURM_JOB_ID="111")
    assert not oc.ssd_remove_owned(foreign) and os.path.exists(
        os.path.join(foreign, "precious")
    )
    assert oc.ssd_remove_owned(new) and not os.path.exists(new)
    assert not oc.ssd_remove_owned(None)
    print(
        "SSD dir: created+marked, foreign never wiped, own wiped on request, "
        "removed only if ours"
    )


def t_ssd_refuses_hard_pin():
    env(
        OC_MC_SSD_PATH=os.path.join(TMP, "pin"), OC_MC_SSD_WIPE=None, SLURM_JOB_ID="111"
    )
    for pin in (None, "1"):
        env(OC_HARD_PIN=pin)
        for make in (lambda: oc.make_store(segment_bytes=1 << 20), oc.Writer):
            try:
                make()
                raise AssertionError(f"accepted SSD offload with OC_HARD_PIN={pin}")
            except RuntimeError as e:
                assert "OC_HARD_PIN" in str(e), e
    env(OC_HARD_PIN="0")
    oc.make_store(segment_bytes=1 << 20)
    env(OC_MC_SSD_PATH=None)
    print("SSD offload with hard pin (default or 1) refused by make_store and Writer")


def t_admission():
    # Without SSD offload the pool formula is unchanged (the SSD rule is tested in
    # t57e).
    for S, sb, g, pool, cm in (
        (500, 2e9, 2, 96 << 30, 1.0),
        (500, 3e8, 1, 8 << 30, 1.5),
        (3, 1e12, 4, 1 << 30, 1.0),
    ):
        assert oc.admission_c(S, sb, g, pool, cm) == max(
            1, min(S, int(cm * 0.75 * pool / (sb * g)))
        )
    print("admission without SSD: the pool formula")


def t_release_order():
    rel = os.path.join(TMP, "released")
    calls = []

    class Store:
        def remove_by_regex(self, rx, force):
            calls.append(
                ("remove", rx, force, os.path.exists(os.path.join(rel, "sess1")))
            )
            return 7

    n = oc.release_session(Store(), "sess1", rel)
    assert n == 7 and calls == [("remove", "^" + oc.PREFIX + "/sess1/", True, True)], (
        calls
    )
    assert oc.release_session(Store(), "sess2", None) == 7
    print(
        "release: marker written before the objects are removed; removed count returned"
    )


def t_dead_procs():
    P = lambda rc, args: types.SimpleNamespace(poll=lambda: rc, args=args)  # noqa: E731
    assert oc.dead_procs([P(None, ["mooncake_master"]), P(None, ["python"])]) == []
    d = oc.dead_procs([P(None, ["mooncake_master"]), P(-9, ["python", "-c", "holder"])])
    assert d == [dict(i=1, args="python -c holder", rc=-9)], d
    print("helper processes: an exited master/holder is reported with its rc")


def t_driver_wiring():
    src = pathlib.Path(os.path.join(HERE, "x10_driver.py")).read_text(encoding="utf8")
    # Every release goes through release(): marker, remove, fence; counts recorded.
    assert (
        src.count("oc.release_session(") == 1 and src.count('release(s["name"])') == 1
    )
    assert (
        "release(name)" in src and "release_removed" in src and "probe_removed" in src
    )
    assert 'remove_by_regex("^" + oc.PREFIX + "/" + s["name"]' not in src
    assert "oc.admission_c(" in src and "0.75 * POOL" not in src
    assert "oc.refetch_missing(reader.get, keys, out)" in src
    tick = src[src.index("def tick():") :]
    assert tick.index("check_helpers()") < tick.index("OC_NO_TICK")
    fin = "".join(
        src[
            src.index("def finish(deadlock=False):") : src.index(
                "def _rank_dump_compare"
            )
        ].split()
    )
    i_sweep, i_kill = fin.index("remove_by_regex("), fin.index('_step("terminate"')
    i_wait, i_rm, i_sum = (
        fin.index('_step("wait"'),
        fin.index("oc.ssd_remove_owned("),
        fin.index("write_summary(json.dumps("),
    )
    assert i_sweep < i_kill < i_wait < i_rm < i_sum, (
        i_sweep,
        i_kill,
        i_wait,
        i_rm,
        i_sum,
    )
    print(
        "x10 driver: release, admission, backoff, helper polling and cleanup wired "
        "as above"
    )


if __name__ == "__main__":
    t_ssd_dir_ownership()
    t_ssd_refuses_hard_pin()
    t_admission()
    t_release_order()
    t_dead_procs()
    t_driver_wiring()
    print("all t57 tests passed")
