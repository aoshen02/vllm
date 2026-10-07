"""Round-2 mutations for client v3 (audit fixes): v1 parse_and_validate_reply vs v3 bench-body
on identical bytes. Covers duplicate keys in checked objects, Python's int-digit and nesting
limits (whole-document depth), and NaN extras in periodic rows.

  BIN=... python xround2.py BODIES_DIR... [--out file.json]
Each DIR holds body-0000.json (n=4096 recorded bodies); format inferred from the dir name.
"""
import os as _genopt_os, sys as _genopt_sys  # rfc57479 harness: site settings come from the environment
GENOPT_ROOT = _genopt_os.environ.get("GENOPT_ROOT", ".")
GENOPT_PYTHON = _genopt_os.environ.get("GENOPT_PYTHON", _genopt_sys.executable)
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

CC = Path(__file__).resolve().parent.parent / "crosscheck.py"
spec = importlib.util.spec_from_file_location("cc", CC)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
R = GENOPT_ROOT + "/agent_run"
BIN = os.environ["BIN"]
POOL = R + "/results/frontend-mock-256k-20261004/frozen-r1/model-assets/token-pool-v2.json"
cc.client1.POOL_IDS = np.asarray([e["id"] for e in json.load(open(POOL))], dtype=np.int64)
N = 4096


def nest(depth_inside):
    """A value that adds `depth_inside` nested containers."""
    return "[" * depth_inside + "1" + "]" * depth_inside


def mutations(s, fmt):
    m = []
    top = s.index("{") + 1
    ch = s.index('"choices":[{') + len('"choices":[{')
    m.append(("dup top-level request_id: wrong first, right last (Python: last wins)",
              s[:top] + '"request_id":"mock-0042",' + s[top:]))
    m.append(("dup top-level request_id: right first, wrong last",
              s[:-1] + ',"request_id":"mock-0042"}'))
    m.append(("dup finish_reason: length first, abort last", s[:ch] + '"finish_reason":"length",' + s[ch:]))
    m.append(("dup unknown top-level key (both valid)", s[:top] + '"zz":1,"zz":2,' + s[top:]))
    if fmt == "compact":
        b = s.index('"compact_logprobs":{') + len('"compact_logprobs":{')
        m.append(("dup compact byteorder: big first, little last", s[:b] + '"byteorder":"big",' + s[b:]))
        m.append(("dup compact num_slots: same value twice", s[:b] + '"num_slots":129,' + s[b:]))
    # whole-document depth: top object = 1 level; "x": value adds k levels
    for k in (9988, 9989, 9990, 9995, 9996, 9997):
        m.append((f"extra top-level field nested {k} (document depth {k + 1})", s[:-1] + ',"x":' + nest(k) + "}"))
    for k in (9990, 9996):
        # inside the choice object (depth 3) -> document depth k + 3
        m.append((f"extra choice field nested {k} (document depth {k + 3})", s[:ch] + '"x":' + nest(k) + "," + s[ch:]))
    for d, sign in ((4300, ""), (4301, ""), (4300, "-"), (4301, "-")):
        m.append((f"extra int {sign}{d} digits", s[:-1] + ',"x":' + sign + "7" * d + "}"))
    m.append(("extra float 6000-digit mantissa", s[:-1] + ',"x":' + "7" * 6000 + ".5}"))
    m.append(("extra int 4301 digits inside array", s[:-1] + ',"x":[1,' + "7" * 4301 + "]}"))
    if fmt == "openai":
        m.append(("NaN extra field in every row (Python NaN singleton -> rows equal)",
                  s.replace('"top_logprobs":[', '"zz":NaN,"top_logprobs":[')))
        first = s.index('"top_logprobs":[')
        m.append(("NaN extra field only in row 0 (period break)",
                  s[:first] + '"zz":NaN,' + s[first:]))
    return [(name, text.encode()) for name, text in m]


def main(dirs, out):
    rows = []
    for d in dirs:
        fmt = "compact" if "compact" in Path(d).name else "openai"
        s = (Path(d) / "body-0000.json").read_bytes().decode()
        for validated in (True, False):
            for name, data in mutations(s, fmt):
                v1 = cc.v1_verdict(data, 0, validated, fmt, N)
                v3 = cc.v3_verdict(BIN, data, 0, validated, fmt, N, POOL)
                same = v1["status"] == v3["status"] and (
                    v1["status"] == "FAIL" or all(v1[k] == v3[k] for k in ("sha256", "bytes", "top", "validated")))
                kind = "agree" if same else ("v3-stricter" if v3["status"] == "FAIL" else "V3-WEAKER")
                rows.append({"body": Path(d).name, "validated": validated, "mutation": name, "v1": v1["status"],
                             "v3": v3["status"], "kind": kind, "v1_error": v1.get("error"), "v3_error": v3.get("error")})
                print(f"{kind:11} {Path(d).name} val={int(validated)} v1={v1['status']} v3={v3['status']} {name}"
                      + ("" if same else f" | v1:{v1.get('error')} | v3:{v3.get('error')}"), flush=True)
    if out:
        Path(out).write_text(json.dumps(rows, indent=1) + "\n")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print("summary", kinds, "of", len(rows))


if __name__ == "__main__":
    args = sys.argv[1:]
    out = None
    if "--out" in args:
        i = args.index("--out")
        out = args[i + 1]
        del args[i:i + 2]
    main(args, out)
