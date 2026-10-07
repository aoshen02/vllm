"""Duplicate-key mutations on choices[0].logprobs (openai) — v1 vs frozen v3."""
import os as _genopt_os, sys as _genopt_sys  # rfc57479 harness: site settings come from the environment
GENOPT_ROOT = _genopt_os.environ.get("GENOPT_ROOT", ".")
GENOPT_PYTHON = _genopt_os.environ.get("GENOPT_PYTHON", _genopt_sys.executable)
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

CC = Path(GENOPT_ROOT + "/agent_run/scripts/genopt/client3/crosscheck.py")
spec = importlib.util.spec_from_file_location("cc", CC)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
R = GENOPT_ROOT + "/agent_run"
import os
BIN = os.environ.get("BIN", R + "/results/generate-opt-20261004/src/client3/genopt-client3-5bfdc72e")
POOL = R + "/results/frontend-mock-256k-20261004/frozen-r1/model-assets/token-pool-v2.json"
cc.client1.POOL_IDS = np.asarray([e["id"] for e in json.load(open(POOL))], dtype=np.int64)

for d in sys.argv[1:]:
    body = (Path(d) / "body-0000.json").read_bytes()
    s = body.decode()
    i = s.index('"logprobs"')
    j = s.index("{", i)
    _, e = json.JSONDecoder().raw_decode(s, j)  # end of logprobs object
    muts = [
        ("dup choice.logprobs: valid object then null", s[:e] + ',"logprobs":null' + s[e:]),
        ("dup choice.logprobs: valid object then {}", s[:e] + ',"logprobs":{}' + s[e:]),
        ("dup choice.logprobs: valid object then {\"content\":null}", s[:e] + ',"logprobs":{"content":null}' + s[e:]),
        ("dup choice.logprobs: null then valid object", s[:i] + '"logprobs":null,' + s[i:]),
        ("dup logprobs.content: valid then [] ", s[:e - 1] + ',"content":[]' + s[e - 1:]),
    ]
    for validated in (True, False):
        for name, text in muts:
            data = text.encode()
            v1 = cc.v1_verdict(data, 0, validated, "openai", 4096)
            v3 = cc.v3_verdict(BIN, data, 0, validated, "openai", 4096, POOL)
            tag = "OK  " if v1["status"] == v3["status"] else "DIFF"
            print(f"{tag} {Path(d).name} val={int(validated)} v1={v1['status']} v3={v3['status']} {name}"
                  f" | v1:{v1.get('error')} | v3:{v3.get('error')}", flush=True)
