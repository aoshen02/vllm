# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57p (R29): with a manifest, the verdict is whether every listed object exists (a
marker beside it is stale, counted, ignored); the marker fails a request fast only while
its manifest is absent. The injection log is set before the engine starts. A failed
request is injected only if all its failed objects are injected keys. The reader never
downloads objects to test existence. Legacy text markers yield their failed keys.
Run: python -I t57p.py [x10_driver.py] [x9_driver.py]  (uses t57f/t57n/t57o fakes)"""

import json
import os
import pathlib
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oc_reader as rd  # noqa: E402
import t57f as base  # noqa: E402
from t57f import oc  # noqa: E402
from t57n import X9, ns_from  # noqa: E402
from t57o import build, checker_ph  # noqa: E402

DRIVERS = (base.DRIVER, X9)


def t_manifest_beats_marker():
    w, mk, total, m, T = build()
    w.objs[oc.err_key(mk)] = json.dumps(
        dict(failed=[oc.block_key("s", m["blocks"][0])])
    ).encode()
    smp = rd.read_sample(w.get, mk, total)
    assert smp.ok and smp.stale_marker, smp.error
    key = oc.block_key("s", m["blocks"][1])
    v = w.objs.pop(key)
    smp = rd.read_sample(w.get, mk, total)
    assert not smp.ok and smp.error == f"listed object absent: {key}", smp.error
    w.objs[key] = v
    raw = w.objs.pop(mk)
    smp = rd.read_sample(w.get, mk, total)
    assert not smp.ok and smp.error.startswith("error marker"), smp.error
    w.objs[mk] = raw
    for path in DRIVERS:
        src = pathlib.Path(path).read_text(encoding="utf8")
        ns = ns_from(
            src,
            ["get_manifest", "READ_STATS"],
            path,
            reader=types.SimpleNamespace(get=w.get),
            say=lambda *a: None,
        )
        assert (
            ns["get_manifest"]("s", "r0", timeout=1) == m
            and ns["READ_STATS"].get("stale_marker") == 1
        ), path
        ph = checker_ph(path, w, total, T)
        assert ph["failed_requests"] == 0 and ph["r3_rows_checked"] == total, (path, ph)
    print(
        "a complete manifest wins over a stale marker (reader, x10, x9); the marker "
        "fails only "
        "without a manifest"
    )


def t_inject_log_before_engine():
    for path in DRIVERS:
        src = pathlib.Path(path).read_text(encoding="utf8")
        at, llm = (
            src.find('os.environ.setdefault("OC_INJECT_LOG"'),
            src.find("\nllm = LLM("),
        )
        assert 0 <= at < llm, (path, at, llm)
    print("x10 and x9 set OC_INJECT_LOG before the engine (and its workers) start")


def t_mixed_causes_are_genuine():
    inj, gen = oc.block_key("mx", "bi"), oc.block_key("mx", "bg")
    oc.INJECTED_KEYS.add(inj)
    store = types.SimpleNamespace(objs={})
    store.get = store.objs.get
    store.objs[oc.err_key(oc.manifest_key("mx", "m"))] = json.dumps(
        dict(failed=[inj, gen])
    ).encode()
    for path in DRIVERS:
        ns = ns_from(
            pathlib.Path(path).read_text(encoding="utf8"),
            ["request_failed", "FAILED_REQ"],
            path,
            reader=types.SimpleNamespace(get=store.get),
            INJECT_EVERY=1,
        )
        ph = {}
        ns["request_failed"](
            ph, dict(name="mx"), "a", "listed object absent", [inj, gen]
        )
        ns["request_failed"](ph, dict(name="mx"), "m", "error marker")
        ns["request_failed"](ph, dict(name="mx"), "b", "listed object absent", [inj])
        assert (
            ph.get("failed_requests") == 2 and ph.get("injected_failed_requests") == 1
        ), (path, ph)
    print("a request with any non-injected failed object is a genuine failure")


def t_reader_never_downloads_for_existence():
    w, mk, total, m, _ = build()
    gets = []

    def get(k):
        gets.append(k)
        return w.objs.get(k)

    try:
        rd.read_sample(get, mk, 10, want=("r3",))
        raise AssertionError("no existence check available: must raise")
    except ValueError as e:
        assert "existence" in str(e), e
    scalar = types.SimpleNamespace(is_exist=lambda k: 1 if k in w.objs else 0)
    smp = rd.read_sample(get, mk, 10, want=("r3",), store=scalar)
    assert smp.ok, smp.error
    assert gets == [mk, oc.block_key("s", m["blocks"][0])], gets
    print(
        "reader: existence by exists=/batch_is_exist/store=, never by downloading; "
        "none -> clear error"
    )


def t_legacy_text_marker():
    inj, mk = oc.block_key("lg", "b1"), oc.manifest_key("lg", "r")
    why = {inj: "RuntimeError('injected write error')"}
    gated = f"{mk} {RuntimeError(f'names failed objects: {why}')!r}".encode()
    own = f"{mk} {RuntimeError('FILE_WRITE_FAIL')!r}".encode()
    assert oc.marker_failed_keys(gated) == [inj], oc.marker_failed_keys(gated)
    assert oc.marker_failed_keys(own) == [mk], oc.marker_failed_keys(own)
    print("legacy text markers: failed keys parsed (gated manifest, failed manifest)")


if __name__ == "__main__":
    t_manifest_beats_marker()
    t_inject_log_before_engine()
    t_mixed_causes_are_genuine()
    t_reader_never_downloads_for_existence()
    t_legacy_text_marker()
    print("all t57p tests passed")
