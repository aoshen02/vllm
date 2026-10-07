"""Offline equivalence + negative controls: v1/v2 validation vs v3 on the SAME bytes.

For each recorded body (from a v3 smoke run with GENOPT_CLIENT3_DUMP_DIR) and a
set of mutations, run genopt-client.py's parse_and_validate_reply (the exact
function client v2 calls) and `genopt-client3 bench-body`, and compare the
verdict (PASS/FAIL), sha256, byte count and top_entries_per_position.

  python crosscheck.py --bodies DIR --format compact|openai --output-tokens N \
      --client3 BIN --token-pool POOL [--out result.json]
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import subprocess
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("client1", HERE.parent / "genopt-client.py")
client1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client1)


class Reply:
    def __init__(self, content):
        self.status_code = 200
        self.content = content

    def raise_for_status(self):
        pass


def v1_verdict(body, index, validated, fmt, n):
    client1.CONFIG = types.SimpleNamespace(
        logprobs_format=fmt if fmt == "compact" else None,
        validate_requests=index + 1 if validated else index,
        output_tokens=n,
        # amendment v3 fields read by the updated genopt-client.py (defaults = off)
        input_tokens=16384,
        compact_include_sampled=True,
        compact_include_ranks=True,
        routed_experts_layers=0,
    )
    try:
        r = client1.parse_and_validate_reply(index, Reply(body))
        return {"status": "PASS", "sha256": r["sha256"], "bytes": r["response_bytes"],
                "top": r["top_entries_per_position"], "validated": r["semantically_validated"]}
    except Exception as e:  # noqa: BLE001
        return {"status": "FAIL", "error": f"{type(e).__name__}: {str(e)[:120]}"}


def v3_verdict(binary, body, index, validated, fmt, n, pool):
    with tempfile.NamedTemporaryFile(dir="/dev/shm", suffix=".json") as f:
        f.write(body)
        f.flush()
        cmd = [binary, "bench-body", "--body", f.name, "--index", str(index), "--output-tokens", str(n),
               "--token-pool", pool, "--logprobs-format", fmt, "--validated", "1" if validated else "0"]
        out = subprocess.run(cmd, capture_output=True, text=True)
    r = json.loads(out.stdout.strip().splitlines()[-1])
    if r["status"] != "PASS":
        return {"status": "FAIL", "error": r["result"]["error"][:120]}
    res = r["result"]
    return {"status": "PASS", "sha256": res["sha256"], "bytes": r["bytes"],
            "top": res["top_entries_per_position"], "validated": res["validated"]}


def dumps_like(obj):
    return json.dumps(obj, separators=(",", ":")).encode()


def mutations(fmt, body):
    """(name, index_offset, bytes). index_offset: body is presented as request index+offset."""
    out = [("unmodified", 0, body)]
    text = body
    if fmt == "compact":
        key = b'"token_ids":"'
        i = text.index(key, text.index(b'"compact_logprobs"')) + len(key) + 4000
        flip = b"B" if text[i:i + 1] != b"B" else b"C"
        out.append(("b64 token_ids char changed (valid alphabet)", 0, text[:i] + flip + text[i + 1:]))
        out.append(("b64 token_ids invalid char '!'", 0, text[:i] + b"!" + text[i + 1:]))
        key = b'"logprobs":"'
        j = text.index(key, text.index(b'"compact_logprobs"')) + len(key) + 8000
        flip = b"A" if text[j:j + 1] != b"A" else b"Q"
        out.append(("b64 logprobs char changed (float bits)", 0, text[:j] + flip + text[j + 1:]))
        key = b'"ranks":"'
        k = text.index(key) + len(key) + 40
        flip = b"A" if text[k:k + 1] != b"A" else b"Q"
        out.append(("b64 ranks char changed", 0, text[:k] + flip + text[k + 1:]))
        out.append(("base64 '/' escaped as '\\/' in JSON", 0, None))  # filled below
        obj = json.loads(body)
        b = obj["choices"][0]["compact_logprobs"]
        b["num_slots"] = 128
        out.append(("num_slots=128", 0, dumps_like(obj)))
        obj = json.loads(body)
        obj["choices"][0]["compact_logprobs"]["byteorder"] = "big"
        out.append(("byteorder=big", 0, dumps_like(obj)))
        # escaped-slash variant: valid JSON, same decoded string
        s = text.index(b'"token_ids":"', text.index(b'"compact_logprobs"'))
        e = text.index(b'"', s + 13)
        seg = text[s:e].replace(b"/", b"\\/")
        out[5] = ("base64 '/' escaped as '\\/' in JSON (valid)", 0, text[:s] + seg + text[e:])
    else:
        obj = json.loads(body)
        content = obj["choices"][0]["logprobs"]["content"]
        n = len(content)

        def mut(name, f):
            o = copy.deepcopy(obj)
            f(o["choices"][0]["logprobs"]["content"], o)
            out.append((name, 0, dumps_like(o)))

        def bump(rows, pos, delta):
            for p in range(pos, n, 1024):
                rows[p]["top_logprobs"][3]["logprob"] += delta

        mut("re-serialized compact separators (valid)", lambda c, o: None)
        mut("top entry logprob +1e-3 at phase 5 (all periods)", lambda c, o: bump(c, 5, 1e-3))
        mut("top entry logprob +1e-9 at phase 5 (within tol)", lambda c, o: bump(c, 5, 1e-9))
        mut("top entry logprob +1e-9 at pos 5 only (period break)", lambda c, o: c[5]["top_logprobs"][3].__setitem__("logprob", c[5]["top_logprobs"][3]["logprob"] + 1e-9))
        mut("sampled logprob changed at pos 2000", lambda c, o: c[2000].__setitem__("logprob", c[2000]["logprob"] - 0.5))
        mut("duplicate top entry appended (phase 7, all periods)", lambda c, o: [c[p]["top_logprobs"].append(dict(c[p]["top_logprobs"][0])) for p in range(7, n, 1024)])
        mut("top entry token replaced at pos 9", lambda c, o: c[9]["top_logprobs"][4].__setitem__("token", "token_id:999999999"))
        mut("row token replaced at pos 3", lambda c, o: c[3].__setitem__("token", "token_id:1"))
        mut("content truncated by one row", lambda c, o: c.pop())
        mut("token_ids[10] changed", lambda c, o: o["choices"][0]["token_ids"].__setitem__(10, 1))
        mut("rows >=1024 at phase 11: key order reversed (valid)", lambda c, o: [c.__setitem__(p, dict(reversed(list(c[p].items())))) for p in range(1024 + 11, n, 1024)])
        mut("bytes field of pos 1500 changed", lambda c, o: c[1500].__setitem__("bytes", [0]))
        out.append(("indent=1 re-serialization (valid)", 0, json.dumps(obj, indent=1).encode()))
    o = json.loads(body)
    o["choices"][0]["finish_reason"] = "length"
    out.append(("finish_reason=length", 0, dumps_like(o)))
    out.append(("trailing comma before final brace", 0, body[:-1] + b",}"))
    out.append(("whitespace after first colon (valid)", 0, body.replace(b":", b":  \n", 1)))
    out.append(("truncated body", 0, body[:-7]))
    out.append(("presented as wrong request index", 1, body))
    out.append(("deep syntax error: '][' swap near middle", 0, None))
    mid = len(body) // 2
    k = body.index(b"]", mid)
    out[-1] = ("deep syntax error: extra ']' near middle", 0, body[:k] + b"]" + body[k:])
    return out


def main(a):
    rows = []
    pool = json.loads(Path(a.token_pool).read_text())
    client1.POOL_IDS = np.asarray([e["id"] for e in pool], dtype=np.int64)
    for path in sorted(a.bodies.glob("body-*.json")):
        index = int(path.stem.split("-")[1])
        body = path.read_bytes()
        for validated in (True, False):
            muts = mutations(a.format, body) if index == 0 else [("unmodified", 0, body)]
            for name, off, data in muts:
                v1 = v1_verdict(data, index + off, validated, a.format, a.output_tokens)
                v3 = v3_verdict(a.client3, data, index + off, validated, a.format, a.output_tokens, a.token_pool)
                same = v1["status"] == v3["status"] and (
                    v1["status"] == "FAIL" or all(v1[k] == v3[k] for k in ("sha256", "bytes", "top", "validated")))
                row = {"body": path.name, "index": index + off, "validated": validated, "mutation": name,
                       "v1": v1["status"], "v3": v3["status"], "agree": same,
                       "sha256": v3.get("sha256", "")[:16], "v1_error": v1.get("error"), "v3_error": v3.get("error")}
                rows.append(row)
                print(f"{'OK ' if same else 'DIFF'} idx={index + off} val={int(validated)} v1={v1['status']} v3={v3['status']} {name}"
                      + ("" if same else f" | v1:{v1.get('error')} | v3:{v3.get('error')}"), flush=True)
    if a.out:
        a.out.write_text(json.dumps(rows, indent=1) + "\n")
    print("agree", sum(r["agree"] for r in rows), "/", len(rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bodies", type=Path, required=True)
    p.add_argument("--format", choices=("openai", "compact"), required=True)
    p.add_argument("--output-tokens", type=int, required=True)
    p.add_argument("--client3", required=True)
    p.add_argument("--token-pool", required=True)
    p.add_argument("--out", type=Path)
    main(p.parse_args())
