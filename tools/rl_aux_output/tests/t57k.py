# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""v57j: the session string each warmup, probe and measured request carries (oc_session)
is the name its release removes, and the release regex matches every key the writer
forms for that session (nonce and phase in the name). Executes the driver's own warmup
block, probe_session and the measured-session request construction with a fake engine.
Run: python -I t57k.py [driver.py]"""

import os
import sys
import textwrap
import types
import uuid

import numpy as np
import regex as re

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t57f as base  # noqa: E402
from t57f import oc  # noqa: E402

SRC = base.SRC


class Engine:
    """Fake LLM: records each request's oc_session, returns outputs with request ids."""

    def __init__(self):
        self.sessions, self.n = [], 0

    def generate(self, prompts, params, use_tqdm=False):
        params = params if isinstance(params, list) else [params] * len(prompts)
        outs = []
        for p in params:
            self.sessions.append((p.extra_args or {}).get("oc_session"))
            self.n += 1
            outs.append(
                types.SimpleNamespace(
                    request_id=f"req{self.n}",
                    outputs=[
                        types.SimpleNamespace(
                            routed_experts=np.zeros((0, 2), np.uint16),
                            token_ids=[1],
                            logprobs=None,
                        )
                    ],
                )
            )
        return outs


def SamplingParams(**kw):
    return types.SimpleNamespace(
        extra_args=kw.get("extra_args"),
        **{k: v for k, v in kw.items() if k != "extra_args"},
    )


def TokensPrompt(**kw):
    return kw


def keys_of(session, rid):
    """Every key the worker/writer forms for one request of a session."""
    return [
        oc.block_key(session, "ab12"),
        oc.block_key(session, "ab12") + oc.IDX_SUFFIX,
        oc.tail_key(session, rid),
        oc.idx_tail_key(session, rid),
        oc.hid_key(session, rid, 0),
        oc.cone_key(session, rid),
        oc.cone_key(session, rid, "idx"),
        oc.manifest_key(session, rid),
        oc.err_key(oc.manifest_key(session, rid)),
    ]


def regex_covers(name, session, rid):
    rx = re.compile("^" + oc.PREFIX + "/" + name + "/")
    return all(rx.match(k) for k in keys_of(session, rid))


def t_warmup_names():
    eng, released = Engine(), []
    blk = SRC[
        SRC.index("# ---------------- warmup (untimed) ----------------") : SRC.index(
            'say("warmup done")'
        )
    ]
    ns = dict(
        os=os,
        PHASE=[""],
        guard=types.SimpleNamespace(arm=lambda t: None, disarm=lambda: None),
        llm=eng,
        TokensPrompt=TokensPrompt,
        SamplingParams=SamplingParams,
        text_ids=lambda seed, n: [1] * n,
        DESIGN="ps",
        RUN_NONCE=uuid.uuid4().hex[:8],
        reader=object(),
        FENCE_ON=False,
        STATS={},
        SESSION_RIDS={},
        get_manifest=lambda s, r, timeout=0: {},
        release=lambda n: released.append(n) or 0,
    )
    exec(blk, ns)
    warm = [s for s in eng.sessions if s is not None]
    assert len(warm) == 4 and len(set(warm)) == 1 and released == [warm[0]], (
        warm,
        released,
    )
    assert regex_covers(released[0], warm[0], "req1")
    return warm[0]


def t_probe_names():
    eng, released = Engine(), []
    fn = SRC[SRC.index("def probe_session(tag):") : SRC.index("def _rows_changed")]
    m = dict(blocks=["ab12"], tail_nbytes=0, dtype="uint16", shape=[2], fields=[])
    ns = dict(
        os=os,
        np=np,
        oc=oc,
        llm=eng,
        TokensPrompt=TokensPrompt,
        SamplingParams=SamplingParams,
        text_ids=lambda seed, n: [1] * n,
        REP="",
        RUN_NONCE=uuid.uuid4().hex[:8],
        STATS=dict(phases=[{}]),
        SESSION_RIDS={},
        KEY_HINTS={},
        tick=lambda: None,
        get_manifest=lambda s, r: m,
        get_many=lambda keys: [b"\0\0" * 2 for _ in keys],
        release=lambda n: released.append(n) or 0,
    )
    exec(fn, ns)
    ns["probe_session"]("v0a")
    assert eng.sessions == released and len(released) == 1, (eng.sessions, released)
    assert regex_covers(released[0], eng.sessions[0], "req1")
    return released[0]


def t_measured_names():
    # The request construction of the step loop, on one admitted trajectory.
    loop = SRC[
        SRC.index("        prompts, params = [], []") : SRC.index(
            '        ctx = max(len(s["ids"]) for s in active)'
        )
    ]
    adm = SRC[SRC.index("                inflight.append(\n") :]
    end = "\n                )\n"
    adm = adm[: adm.index(end, adm.index("fork_wait=FORK and j > 0,")) + len(end)]
    ns = dict(
        inflight=[],
        ph=dict(prefill_tokens=0),
        L=1000,
        ARM="full",
        DESIGN="ps",
        REP="",
        RUN_NONCE="ab12cd34",
        STATS=dict(phases=[{}, {}]),
        ids0=[1, 2, 3],
        g=7,
        j=0,
        FORK=False,
        NOR3=False,
        TRAJ_TAG=False,
        GROUP=1,
        TokensPrompt=TokensPrompt,
        SamplingParams=SamplingParams,
        gen_len=lambda s: 4,
        COMPACT=False,
        os=os,
    )
    exec(textwrap.dedent(adm), ns)
    exec(textwrap.dedent(loop), ns)
    sent = ns["params"][0].extra_args["oc_session"]
    s = ns["inflight"][0]
    rel = SRC[SRC.index('if group_left[s["g"]] == 0:') :]
    rel = rel[: rel.index("RESERVED.pop")]
    assert 'release(s["name"])' in rel and sent == s["name"], (sent, s["name"])
    assert regex_covers(s["name"], sent, "req1")
    return sent


if __name__ == "__main__":
    print(
        "warmup session",
        t_warmup_names(),
        "released under the same name; regex covers its keys",
    )
    print(
        "probe session",
        t_probe_names(),
        "released under the same name; regex covers its keys",
    )
    print(
        "measured session",
        t_measured_names(),
        "released under the same name; regex covers its keys",
    )
    print("all t57k tests passed")
