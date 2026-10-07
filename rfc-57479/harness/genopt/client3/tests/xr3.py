"""Amendment-v3 cross-check: v1 parse_and_validate_reply vs v3 bench-body on identical bytes.

Covers compact_include_sampled / compact_include_ranks switches and R3 routed_experts
(.npy header / shape / values), each body presented as validated and structural-only.

  BIN=... python xr3.py --out result.json DIR...
Body dirs are named bodies-[wt-]<case> (from smoke.py --dump-dir); the case name selects
the client configuration: compact|openai, nosampled, noranks, r3 (4 layers), n=4096.
"""
import os as _genopt_os, sys as _genopt_sys  # rfc57479 harness: site settings come from the environment
GENOPT_ROOT = _genopt_os.environ.get("GENOPT_ROOT", ".")
GENOPT_PYTHON = _genopt_os.environ.get("GENOPT_PYTHON", _genopt_sys.executable)
import base64
import importlib.util
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("client1", HERE.parent.parent / "genopt-client.py")
client1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client1)

R = GENOPT_ROOT + "/agent_run"
POOL = R + "/results/frontend-mock-256k-20261004/frozen-r1/model-assets/token-pool-v2.json"
BIN = os.environ["BIN"]
N, P = 4096, 16384
client1.POOL_IDS = np.asarray([e["id"] for e in json.load(open(POOL))], dtype=np.int64)


class Reply:
    status_code = 200

    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


def conf(case):
    return dict(
        fmt="compact" if "compact" in case else "openai",
        sampled="nosampled" not in case,
        ranks="noranks" not in case,
        layers=4 if "r3" in case else 0,
    )


def v1(body, index, validated, c):
    client1.CONFIG = types.SimpleNamespace(
        logprobs_format="compact" if c["fmt"] == "compact" else None,
        validate_requests=index + 1 if validated else index,
        output_tokens=N, input_tokens=P,
        compact_include_sampled=c["sampled"], compact_include_ranks=c["ranks"],
        routed_experts_layers=c["layers"],
    )
    try:
        r = client1.parse_and_validate_reply(index, Reply(body))
        return {"status": "PASS", "sha256": r["sha256"], "bytes": r["response_bytes"],
                "top": r["top_entries_per_position"]}
    except Exception as e:  # noqa: BLE001
        return {"status": "FAIL", "error": f"{type(e).__name__}: {str(e)[:120]}"}


def v3(body, index, validated, c):
    with tempfile.NamedTemporaryFile(dir="/dev/shm", suffix=".json") as f:
        f.write(body)
        f.flush()
        cmd = [BIN, "bench-body", "--body", f.name, "--index", str(index), "--output-tokens", str(N),
               "--input-tokens", str(P), "--token-pool", POOL, "--logprobs-format", c["fmt"],
               "--validated", "1" if validated else "0", "--routed-experts-layers", str(c["layers"])]
        if not c["sampled"]:
            cmd.append("--compact-no-sampled")
        if not c["ranks"]:
            cmd.append("--compact-no-ranks")
        out = subprocess.run(cmd, capture_output=True, text=True)
    lines = out.stdout.strip().splitlines()
    if not lines:
        return {"status": "CRASH", "error": out.stderr[-300:]}
    r = json.loads(lines[-1])
    if r["status"] != "PASS":
        return {"status": "FAIL", "error": r["result"]["error"][:120]}
    return {"status": "PASS", "sha256": r["result"]["sha256"], "bytes": r["bytes"],
            "top": r["result"]["top_entries_per_position"]}


def npy_bytes(header_dict_text, data, version=(1, 0)):
    h = header_dict_text
    pre = 10 if version[0] == 1 else 12
    pad = (64 - (pre + len(h) + 1) % 64) % 64
    h = (h + " " * pad + "\n").encode("latin1")
    out = b"\x93NUMPY" + bytes(version)
    out += struct.pack("<H", len(h)) if version[0] == 1 else struct.pack("<I", len(h))
    return out + h + data


def mutations(s, c):
    m = [("unmodified", s)]
    rows = P + N - 1
    key = '"routed_experts":'
    if c["layers"]:
        a = s.index(key + '"') + len(key) + 1
        e = s.index('"', a)
        raw = base64.b64decode(s[a:e])
        major = raw[6]
        hl = struct.unpack("<H", raw[8:10])[0] if major == 1 else struct.unpack("<I", raw[8:12])[0]
        hs = 10 if major == 1 else 12
        header, data = raw[hs:hs + hl].decode("latin1"), raw[hs + hl:]
        d = header.strip()

        def put(newraw):
            return s[:a] + base64.b64encode(newraw).decode() + s[e:]

        good = f"{{'descr': '|u1', 'fortran_order': False, 'shape': ({rows}, 4, 8), }}"
        flipped = bytearray(data)
        flipped[12345] ^= 0x40
        m += [
            ("R3: header as produced, re-encoded v1.0 (valid)", put(npy_bytes(d, data))),
            ("R3: npy version 2.0 (valid)", put(npy_bytes(d, data, (2, 0)))),
            ("R3: npy version 3.0 (np.load accepts; v3 strict 1.0/2.0)", put(npy_bytes(d, data, (3, 0)))),
            ("R3: bad magic", put(b"\x93NUMPZ" + raw[6:])),
            ("R3: one value changed", put(raw[:hs + hl] + bytes(flipped))),
            ("R3: descr '<u1' (np: uint8; v3 strict '|u1')", put(npy_bytes(d.replace("'|u1'", "'<u1'"), data))),
            ("R3: descr '|i1'", put(npy_bytes(d.replace("'|u1'", "'|i1'"), data))),
            ("R3: fortran_order True", put(npy_bytes(d.replace("False", "True"), data))),
            ("R3: shape rows-1, data truncated", put(npy_bytes(d.replace(f"({rows},", f"({rows - 1},"), data[:-32]))),
            ("R3: shape rows-1, data unchanged", put(npy_bytes(d.replace(f"({rows},", f"({rows - 1},"), data))),
            ("R3: shape 5 layers", put(npy_bytes(d.replace(", 4, 8)", ", 5, 8)"), data))),
            ("R3: data truncated by 1", put(raw[:-1])),
            ("R3: one trailing data byte (np.load ignores; v3 strict)", put(raw + b"\x00")),
            ("R3: double-quoted keys, no trailing comma (valid)",
             put(npy_bytes(f'{{"shape": ({rows}, 4, 8), "fortran_order": False, "descr": "|u1"}}', data))),
            ("R3: extra header key", put(npy_bytes(good[:-1] + "'x': 1, }", data))),
            ("R3: routed_experts null", s[:a - 1] + "null" + s[e + 1:]),
            ("R3: routed_experts key removed", s[:s.index(key) - 1] + s[e + 1:] if s[s.index(key) - 1] == "," else None),
            ("R3: invalid base64 char", s[:a + 100] + "!" + s[a + 101:]),
        ]
    else:
        if key + "null" in s:
            i = s.index(key + "null")
            m.append(("R3 off: routed_experts set to a string", s[:i] + key + '"AAAA"' + s[i + len(key) + 4:]))
            m.append(("R3 off: routed_experts removed (valid)", s[:i - 1] + s[i + len(key) + 4:] if s[i - 1] == "," else None))
    if c["fmt"] == "compact":
        blk = s.index('"compact_logprobs":{') + len('"compact_logprobs":{')
        if not c["sampled"]:
            m += [
                ("sampled_slot removed", s.replace('"sampled_slot":false,', "", 1) if '"sampled_slot":false,' in s else None),
                ("sampled_slot 0", s.replace('"sampled_slot":false', '"sampled_slot":0', 1)),
                ("sampled_slot true", s.replace('"sampled_slot":false', '"sampled_slot":true', 1)),
                ("num_slots 129 (arrays unchanged)", s.replace('"num_slots":128', '"num_slots":129', 1)),
            ]
        else:
            m.append(("sampled_slot false added while include_sampled", s[:blk] + '"sampled_slot":false,' + s[blk:]))
        ranks_ok = base64.b64encode(((np.arange(N) % 128) + 1).astype("<i4").tobytes()).decode()
        if not c["ranks"]:
            m += [
                ("ranks added (correct size/values)", s[:blk] + f'"ranks":"{ranks_ok}",' + s[blk:]),
                ("ranks added (wrong size)", s[:blk] + '"ranks":"AAAA",' + s[blk:]),
            ]
        else:
            rk = re.search(r',"ranks":"[^"]*"', s)
            m.append(("ranks removed while include_ranks", s[:rk.start()] + s[rk.end():] if rk else None))
    return [(n, t.encode()) for n, t in m if t is not None]


def main(out, dirs):
    rows = []
    for d in dirs:
        case = Path(d).name.removeprefix("bodies-").removeprefix("wt-")
        c = conf(case)
        s = (Path(d) / "body-0000.json").read_bytes().decode()
        for validated in (True, False):
            for name, data in mutations(s, c):
                a, b = v1(data, 0, validated, c), v3(data, 0, validated, c)
                same = a["status"] == b["status"] and (a["status"] == "FAIL" or all(a[k] == b[k] for k in ("sha256", "bytes", "top")))
                kind = "agree" if same else ("v3-stricter" if b["status"] == "FAIL" and a["status"] == "PASS" else "V3-WEAKER")
                rows.append({"case": case, "validated": validated, "mutation": name, "v1": a["status"], "v3": b["status"],
                             "kind": kind, "v1_error": a.get("error"), "v3_error": b.get("error")})
                print(f"{kind:11} {case} val={int(validated)} v1={a['status']} v3={b['status']} {name}"
                      + ("" if same else f" | v1:{a.get('error')} | v3:{b.get('error')}"), flush=True)
        # other bodies of the cell: unmodified only
        for f in sorted(Path(d).glob("body-*.json"))[1:]:
            idx = int(f.stem.split("-")[1])
            data = f.read_bytes()
            for validated in (True, False):
                a, b = v1(data, idx, validated, c), v3(data, idx, validated, c)
                same = a["status"] == b["status"] == "PASS" and all(a[k] == b[k] for k in ("sha256", "bytes", "top"))
                rows.append({"case": case, "validated": validated, "mutation": f"unmodified body {idx}",
                             "v1": a["status"], "v3": b["status"], "kind": "agree" if same else "V3-WEAKER-or-FAIL"})
    Path(out).write_text(json.dumps(rows, indent=1) + "\n")
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print("summary", kinds, "of", len(rows))


if __name__ == "__main__":
    args = sys.argv[1:]
    assert args[0] == "--out"
    main(args[1], args[2:])
