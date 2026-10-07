"""Auditor's extended mutation set: v1 parse_and_validate_reply vs frozen v3 bench-body.

Reuses crosscheck.py's v1_verdict / v3_verdict unchanged (imported, not modified).
  python xcheck.py --bodies DIR --format compact|openai --n 4096 --client3 BIN --token-pool P --out OUT
"""
import os as _genopt_os, sys as _genopt_sys  # rfc57479 harness: site settings come from the environment
GENOPT_ROOT = _genopt_os.environ.get("GENOPT_ROOT", ".")
GENOPT_PYTHON = _genopt_os.environ.get("GENOPT_PYTHON", _genopt_sys.executable)
import argparse
import base64
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

CC = Path(GENOPT_ROOT + "/agent_run/scripts/genopt/client3/crosscheck.py")
spec = importlib.util.spec_from_file_location("cc", CC)
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)
client1 = cc.client1


def sub_once(b, old, new, start=0):
    i = b.index(old, start)
    return b[:i] + new + b[i + len(old):]


def row_spans(s):
    """s: str. Returns list of (start, end) of content rows."""
    k = s.index('"content"')
    k = s.index('[', k) + 1
    dec = json.JSONDecoder()
    spans = []
    i = k
    ws = ' \n\r\t'
    while True:
        while s[i] in ws:
            i += 1
        if s[i] == ']':
            break
        _, e = dec.raw_decode(s, i)
        spans.append((i, e))
        i = e
        while s[i] in ws:
            i += 1
        if s[i] == ',':
            i += 1
    return spans


def replace_rows(s, spans, repl):
    """repl: {row_index: new_text}; returns str."""
    out = []
    prev = 0
    for idx in sorted(repl):
        a, e = spans[idx]
        out.append(s[prev:a])
        out.append(repl[idx])
        prev = e
    out.append(s[prev:])
    return "".join(out)


def common_mutations(body, fmt):
    m = []
    rid_key = b'"request_id"'
    i = body.index(rid_key)
    j = body.index(b'"', body.index(b':', i) + 1)  # opening quote of value
    k = body.index(b'"', j + 1)
    rid = body[j + 1:k]  # e.g. mock-0000 / generate-tokens-mock-0000
    pre, post = body[:j], body[k + 1:]
    esc = rid.replace(b"mock-0", b"mock-\\u0030", 1)
    m.append(("request_id with \\u0030 escape (valid)", pre + b'"' + esc + b'"' + post))
    esc2 = rid.replace(b"mock-", b"mock\\u002d", 1)
    m.append(("request_id with \\u002d escape (valid)", pre + b'"' + esc2 + b'"' + post))
    m.append(("request_id last digit fullwidth \\uff10 escape", pre + b'"' + rid[:-1] + b'\\uff10"' + post))
    m.append(("request_id last digit fullwidth raw UTF-8", pre + b'"' + rid[:-1] + "０".encode() + b'"' + post))
    m.append(("request_id trailing NUL escape \\u0000", pre + b'"' + rid + b'\\u0000"' + post))
    # duplicate keys
    m.append(("dup request_id: wrong first, right last (py last-wins)",
              b'{"request_id":"mock-9999",' + body[1:]))
    m.append(("dup request_id: right first, wrong last",
              pre + b'"' + rid + b'","request_id":"mock-9999"' + post))
    # finish_reason variants
    fr = b'"finish_reason":"abort"' if b'"finish_reason":"abort"' in body else b'"finish_reason": "abort"'
    m.append(("finish_reason \\u0061bort escape (valid)", sub_once(body, fr, fr.replace(b'"abort"', b'"\\u0061bort"'))))
    m.append(("dup finish_reason: length then abort (valid)", sub_once(body, fr, b'"finish_reason":"length",' + fr)))
    m.append(("dup finish_reason: abort then length", sub_once(body, fr, fr + b',"finish_reason":"length"')))
    m.append(("escaped key fin\\u0069sh_reason (valid)", sub_once(body, fr, fr.replace(b"finish", b"fin\\u0069sh"))))
    # extra top-level fields with special numbers
    for name, lit in [("1e400", b"1e400"), ("-1e400", b"-1e400"), ("-0", b"-0"), ("-0.0", b"-0.0"),
                      ("NaN", b"NaN"), ("-Infinity", b"-Infinity"), ("1e-400", b"1e-400"),
                      ("int 40 digits", b"9" * 40), ("int 4300 digits", b"9" * 4300),
                      ("int 4301 digits", b"9" * 4301), ("int 5000 digits", b"9" * 5000),
                      ("float 5000-digit mantissa", b"1." + b"3" * 5000),
                      ("nested depth 900", b"[" * 900 + b"]" * 900),
                      ("nested depth 3000", b"[" * 3000 + b"]" * 3000),
                      ("nested depth 9000", b"[" * 9000 + b"]" * 9000),
                      ("nested depth 12000", b"[" * 12000 + b"]" * 12000),
                      ("lone surrogate escape", b'"\\ud800x"'),
                      ("surrogate pair escape", b'"\\ud83d\\ude00"'),
                      ("raw UTF-8 surrogate bytes", b'"\xed\xa0\x80"'),
                      ("invalid UTF-8 0xff in string", b'"ab\xffcd"'),
                      ("overlong UTF-8 C0 80", b'"\xc0\x80"'),
                      ("raw tab in string", b'"a\tb"'),
                      ("DEL 0x7f in string", b'"a\x7fb"'),
                      ("-NaN", b"-NaN"), ("+1", b"+1"), ("01", b"01"), (".5", b".5"), ("1.", b"1."),
                      ("Infinity", b"Infinity"), ("nan lowercase", b"nan"), ("true", b"true")]:
        m.append((f"extra top-level field = {name}", b'{"zz_extra":' + lit + b"," + body[1:]))
    m.append(("UTF-8 BOM prefix", b"\xef\xbb\xbf" + body))
    m.append(("non-ASCII raw byte outside strings (e9 after comma)", b'{"zz":1,\xc3\xa9' + body[1:]))
    m.append(("trailing newline+spaces (valid)", body + b"\n  \r\n"))
    m.append(("trailing garbage 'x'", body + b"x"))
    m.append(("two documents", body + body))
    m.append(("trailing NUL", body + b"\x00"))
    m.append(("form-feed whitespace after first colon", body.replace(b":", b":\x0c", 1)))
    m.append(("truncated at 65536 bytes", body[:65536]))
    m.append(("truncated at 1 MiB boundary", body[:1 << 20]))
    m.append(("truncated: final '}' only removed", body[:-1]))
    m.append(("empty body", b""))
    m.append(("top-level array", b"[" + body + b"]"))
    # choices duplicates
    ck = b'"choices":['
    m.append(("dup choices: [] first then real (py last-wins)", sub_once(body, ck, b'"choices":[],' + ck)))
    m.append(("dup choices: bad choice first then real", sub_once(body, ck, b'"choices":[{"finish_reason":"length"}],' + ck)))
    m.append(("choices with 2 entries", None))  # filled below
    i = body.index(ck) + len(ck)
    # find end of choices array: use json to locate
    s = body.decode("utf-8", "surrogatepass")
    ci = s.index('"choices"')
    ci = s.index('[', ci) + 1
    _, ce = json.JSONDecoder().raw_decode(s, ci)
    m[-1] = ("choices with 2 entries (second {})", (s[:ce] + ",{}" + s[ce:]).encode())
    return m


def compact_mutations(body, n):
    m = []
    slots = 129
    s0 = body.index(b'"compact_logprobs"')

    def span(key):
        kk = b'"' + key + b'":'
        a = body.index(kk, s0) + len(kk)
        while body[a:a + 1] in (b" ",):
            a += 1
        assert body[a:a + 1] == b'"'
        e = body.index(b'"', a + 1)
        return a + 1, e  # string content span

    ta, te = span(b"token_ids")
    la, le = span(b"logprobs")
    ra, re_ = span(b"ranks")
    tok = body[ta:te]
    rk = body[ra:re_]
    assert len(tok) == n * slots * 4 // 3 * 4 // 4 * 1 or True
    # non-canonical trailing bits in ranks
    if rk.endswith(b"=="):
        c = rk[-3:-2]
        alt = b"B" if c == b"A" else (b"A" if c != b"A" else b"B")
        # flip a low bit that canonical encoding would zero
        val = base64.b64decode(rk[-4:])
        tbl = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        idx = tbl.index(rk[-3:-2])
        nc = tbl[idx | 1:(idx | 1) + 1] if idx & 1 == 0 else tbl[idx & ~1:(idx & ~1) + 1]
        newrk = rk[:-3] + nc + b"=="
        assert base64.b64decode(newrk) == base64.b64decode(rk), "trailing bit mutation changed value"
        m.append(("ranks: non-canonical trailing bits (same decoded bytes)", body[:ra] + newrk + body[re_:]))
        m.append(("ranks: padding removed", body[:ra] + rk[:-2] + body[re_:]))
        m.append(("ranks: extra '=' padding", body[:ra] + rk + b"=" + body[re_:]))
    # wrong-but-valid lengths
    m.append(("ranks: last quad removed (wrong length, valid b64)", body[:ra] + rk[:-8] + rk[-4:] + body[re_:]))
    m.append(("token_ids: first 16 chars removed (12 bytes = 3 ints short)", body[:ta] + tok[16:] + body[te:]))
    m.append(("token_ids: one position removed (688 chars = 516 bytes)", body[:ta] + tok[688:] + body[te:]))
    m.append(("token_ids: 64 chars removed mid (NEON block)", body[:ta + 64 * 1000] + body[ta + 64 * 1001:]))
    m.append(("token_ids: 'AAAA' appended (3 bytes extra)", body[:te] + b"AAAA" + body[te:]))
    m.append(("token_ids: one position appended (688 chars)", body[:te] + tok[:688] + body[te:]))
    # swapped rows (positions 0 and 1) in token_ids / logprobs
    m.append(("token_ids: rows 0 and 1 swapped", body[:ta] + tok[688:1376] + tok[:688] + tok[1376:] + body[te:]))
    lp = body[la:le]
    m.append(("logprobs: rows 5 and 6 swapped", body[:la] + lp[:688 * 5] + lp[688 * 6:688 * 7] + lp[688 * 5:688 * 6] + lp[688 * 7:] + body[le:]))
    m.append(("ranks: swap of two int32 quads? (positions 0..2 vs 3..5)",
              body[:ra] + rk[16:32] + rk[:16] + rk[32:] + body[re_:]))
    # escapes inside base64 strings
    m.append(("token_ids: 'A' at bulk offset 6400 as \\u0041 (valid if char is A)", None))
    p = ta + 6400
    ch = body[p:p + 1]
    m[-1] = (f"token_ids: char at bulk offset 6400 as \\u00{ord(ch):02x} escape (valid)",
             body[:p] + b"\\u00%02x" % ord(ch) + body[p + 1:])
    m.append(("token_ids: '=' in middle of bulk", body[:ta + 6400] + b"=" + body[ta + 6401:]))
    m.append(("token_ids: '-' (urlsafe) in bulk", body[:ta + 6400] + b"-" + body[ta + 6401:]))
    m.append(("token_ids: non-ASCII byte in bulk", body[:ta + 6400] + b"\xc3" + body[ta + 6401:]))
    m.append(("token_ids: escaped newline \\n in bulk", body[:ta + 6400] + b"\\n" + body[ta + 6400:]))
    m.append(("token_ids: char in last 64-char tail changed", body[:te - 10] + (b"B" if body[te - 10:te - 9] != b"B" else b"C") + body[te - 9:]))
    m.append(("truncated exactly at a 64-char base64 block boundary in token_ids", body[:ta + 64 * 2000]))
    m.append(("truncated at end of token_ids string (before closing quote)", body[:te]))
    # metadata types/values
    for key, old, new, name in [
        (b"num_positions", None, b"4096.0", "num_positions as float 4096.0"),
        (b"num_slots", None, b"129.0", "num_slots as float 129.0"),
        (b"num_slots", None, b"true", "num_slots as true"),
        (b"dtype_token_ids", None, b'"<i4"', "dtype_token_ids '<i4'"),
        (b"dtype_logprobs", None, b'"float64"', "dtype_logprobs 'float64'"),
        (b"dtype_logprobs", None, b'"f4"', "dtype_logprobs 'f4'"),
        (b"byteorder", None, b'"l\\u0069ttle"', "byteorder escaped (valid)"),
        (b"byteorder", None, b'"big"', "byteorder big"),
    ]:
        kk = b'"' + key + b'":'
        a = body.index(kk, s0) + len(kk)
        while body[a:a + 1] == b" ":
            a += 1
        _, e = json.JSONDecoder().raw_decode(body.decode("latin-1"), a)
        m.append((name, body[:a] + new + body[e:]))
    # duplicate compact keys
    rr = body[ra - len(b'"ranks":"'):ra - 1]
    m.append(("dup ranks key: garbage-same-size first, real last (py last-wins)",
              body[:ra - len(b'"ranks":"')] + b'"ranks":"' + base64.b64encode(b"\x07" * (4 * n)) + b'",' + body[ra - len(b'"ranks":"'):]))
    m.append(("dup byteorder: big then little (valid)", sub_once(body, b'"byteorder":', b'"byteorder":"big","byteorder":', s0)))
    # choice.logprobs
    if b'"logprobs":null' in body:
        m.append(("choice.logprobs [] instead of null", sub_once(body, b'"logprobs":null', b'"logprobs":[]')))
        m.append(("choice.logprobs removed", sub_once(body, b'"logprobs":null,', b'')))
    # token_ids list value variants
    tk = b'"token_ids":['
    a = body.index(tk) + len(tk)
    e = body.index(b",", a)
    first = body[a:e]
    m.append(("token_ids[0] written as float X.0", body[:a] + first + b".0" + body[e:]))
    m.append(("token_ids[0] written as XeO exponent", body[:a] + first + b"e0" + body[e:]))
    m.append(("token_ids[0] with leading '-0'? (-0 replacing value)", body[:a] + b"-0" + body[e:]))
    return m


def openai_mutations(body, n, L):
    s = body.decode("utf-8")
    spans = row_spans(s)
    assert len(spans) == n, len(spans)
    row = lambda i: s[spans[i][0]:spans[i][1]]
    phase_rows = lambda ph: list(range(ph, n, L))
    m = []

    def R(name, repl):
        m.append((name, replace_rows(s, spans, repl).encode()))

    R("swap rows 0 and 1", {0: row(1), 1: row(0)})
    R("swap rows 2000 and 2001", {2000: row(2001), 2001: row(2000)})
    R("swap rows 5 and 5+L (same phase: no-op, valid)", {5: row(5 + L), 5 + L: row(5)})

    def tl_reversed(txt):
        o = json.loads(txt)
        o["top_logprobs"] = list(reversed(o["top_logprobs"]))
        return json.dumps(o, separators=(",", ":"))

    R("top_logprobs reversed in template row 5 only", {5: tl_reversed(row(5))})
    R("top_logprobs reversed in all phase-5 rows (valid)", {p: tl_reversed(row(p)) for p in phase_rows(5)})
    R("top_logprobs reversed in row 5+L only (period break)", {5 + L: tl_reversed(row(5 + L))})

    def addfield(txt, lit):
        return txt[:-1] + ',"zz":' + lit + "}"

    R("extra field in template row 5 only", {5: addfield(row(5), "1")})
    R("extra field in all phase-5 rows (valid)", {p: addfield(row(p), "1") for p in phase_rows(5)})
    R("extra NaN field in all phase-5 rows (py identity: PASS)", {p: addfield(row(p), "NaN") for p in phase_rows(5)})
    R("extra 1e400 field in all phase-5 rows (valid)", {p: addfield(row(p), "1e400") for p in phase_rows(5)})
    R("dup key: token 'zzz' prepended in row 5 (last-wins valid)", {5: '{"token":"zzz",' + row(5)[1:]})
    R("dup key: token 'zzz' prepended in row 5+L (last-wins valid)", {5 + L: '{"token":"zzz",' + row(5 + L)[1:]})
    R("dup key: token 'zzz' appended in row 5", {5: row(5)[:-1] + ',"token":"zzz"}'})
    R("dup key: token 'zzz' appended in row 5+2L", {5 + 2 * L: row(5 + 2 * L)[:-1] + ',"token":"zzz"}'})
    # numeric textual variants in a later row (raw differs, value equal)
    r = row(7 + L)
    o = json.loads(r)
    lpv = repr(o["logprob"])
    R("row 7+L: logprob text with trailing 0 (same float, valid)", {7 + L: r.replace(f'"logprob":{lpv}', f'"logprob":{lpv}0', 1)})
    R("row 7+L: logprob text in exponent form (same float, valid)", {7 + L: r.replace(f'"logprob":{lpv}', f'"logprob":{lpv}e0', 1)})
    R("row 7+L: bytes[0] as float 116.0 (py ==, valid)", {7 + L: r.replace('"bytes":[116', '"bytes":[116.0', 1)})
    R("row 7+L: bytes[0] 116 -> true (py ==: False)", {7 + L: r.replace('"bytes":[116', '"bytes":[true', 1)})
    R("row 7+L: logprob nextafter float (period break)", {7 + L: r.replace(f'"logprob":{lpv}', f'"logprob":{repr(np.nextafter(o["logprob"], 0))}', 1)})
    R("row 7+L: bytes[0] as 116.0000000000001 (diff)", {7 + L: r.replace('"bytes":[116', '"bytes":[116.0000000000001', 1)})
    R("row 7+L: token escaped \\u005f (valid)", {7 + L: r.replace('"token":"token_id', '"token":"token\\u005fid', 1)})
    R("row 7+L: whitespace inside (valid)", {7 + L: r.replace('","logprob"', '" , "logprob"', 1)})
    # template-row semantic variants
    r5 = row(5)
    o5 = json.loads(r5)
    v5 = repr(o5["logprob"])
    R("row 5 (template): sampled logprob 1e400", {5: r5.replace(f'"logprob":{v5}', '"logprob":1e400', 1)})
    R("row 5 (template): sampled logprob NaN", {5: r5.replace(f'"logprob":{v5}', '"logprob":NaN', 1)})
    R("row 5 (template): sampled logprob as -0", {5: r5.replace(f'"logprob":{v5}', '"logprob":-0', 1)})
    R("row 5 (template): sampled logprob +5e-7 rel (within rel 1e-6)",
      {p: row(p).replace(f'"logprob":{v5}', f'"logprob":{repr(o5["logprob"] * (1 + 5e-7))}', 1) for p in phase_rows(5)})
    R("row 5 (template): sampled logprob +2e-6 rel (outside)",
      {p: row(p).replace(f'"logprob":{v5}', f'"logprob":{repr(o5["logprob"] * (1 + 2e-6))}', 1) for p in phase_rows(5)})
    R("row 5 (template): token escaped \\u005f, all phase-5 rows (valid)",
      {p: row(p).replace('"token":"token_id', '"token":"token\\u005fid', 1) for p in phase_rows(5)})
    R("row 5: escaped key \\u006cogprob, all phase rows (valid)",
      {p: row(p).replace('"logprob":', '"\\u006cogprob":', 1) for p in phase_rows(5)})
    # top entry variants in template row
    def entry_mut(txt, f):
        o = json.loads(txt)
        f(o["top_logprobs"])
        return json.dumps(o, separators=(",", ":"))
    R("top entry token int (non-string) in all phase-9 rows",
      {p: entry_mut(row(p), lambda t: t[4].__setitem__("token", 5)) for p in phase_rows(9)})
    R("top entries: one removed, all phase-9 rows",
      {p: entry_mut(row(p), lambda t: t.pop(4)) for p in phase_rows(9)})
    R("top entries: duplicate of entry 4 with WRONG logprob appended (first-wins valid)",
      {p: entry_mut(row(p), lambda t: t.append({"token": t[4]["token"], "logprob": 0.0, "bytes": []})) for p in phase_rows(9)})
    R("top entries: duplicate with WRONG logprob prepended (first-wins -> FAIL)",
      {p: entry_mut(row(p), lambda t: t.insert(0, {"token": t[4]["token"], "logprob": 0.0, "bytes": []})) for p in phase_rows(9)})
    R("top entries: dup appended with non-number logprob 'x' (not evaluated, valid)",
      {p: entry_mut(row(p), lambda t: t.append({"token": t[4]["token"], "logprob": "x"})) for p in phase_rows(9)})
    R("top entries: dup appended missing logprob key (KeyError)",
      {p: entry_mut(row(p), lambda t: t.append({"token": t[4]["token"]})) for p in phase_rows(9)})
    R("top entries: extra key in entry, all phase rows (valid)",
      {p: entry_mut(row(p), lambda t: t[4].__setitem__("zz", None)) for p in phase_rows(9)})
    R("top entries: entry logprob as bool true (not close)",
      {p: entry_mut(row(p), lambda t: t[4].__setitem__("logprob", True)) for p in phase_rows(9)})
    R("row 9 replaced by its list form [row]", {9: "[" + row(9) + "]"})
    R("row 2000 replaced by null (later row)", {2000: "null"})
    R("content: one row appended", {n - 1: row(n - 1) + "," + row(n - 1 - L)})
    # whole-content tricks
    R("content: rows 0..L-1 correct, rest shifted by one period (valid)", {p: row(p - L) for p in range(L, n)})
    tk = '"token_ids":['
    a = s.index(tk) + len(tk)
    e = s.index(",", a)
    m.append(("token_ids[0] written as float X.0", (s[:a] + s[a:e] + ".0" + s[e:]).encode()))
    m.append(("logprobs null instead of object", (s[:s.index('"logprobs":{')] + '"logprobs":null,"zz_lp":{' + s[s.index('"logprobs":{') + len('"logprobs":{'):]).encode()))
    return m


def main(a):
    pool = json.loads(Path(a.token_pool).read_text())
    client1.POOL_IDS = np.asarray([e["id"] for e in pool], dtype=np.int64)
    L = len(pool)
    body = (a.bodies / "body-0000.json").read_bytes()
    muts = common_mutations(body, a.format)
    muts += compact_mutations(body, a.n) if a.format == "compact" else openai_mutations(body, a.n, L)
    rows = []
    for validated in (True, False):
        for name, data in muts:
            if data is None:
                continue
            v1 = cc.v1_verdict(data, 0, validated, a.format, a.n)
            try:
                v3 = cc.v3_verdict(a.client3, data, 0, validated, a.format, a.n, a.token_pool)
            except Exception as e:  # noqa: BLE001
                v3 = {"status": "CRASH", "error": repr(e)[:200]}
            same = v1["status"] == v3["status"] and (
                v1["status"] == "FAIL" or all(v1[k] == v3[k] for k in ("sha256", "bytes", "top", "validated")))
            rows.append({"validated": validated, "mutation": name, "v1": v1["status"], "v3": v3["status"],
                         "agree": same, "v1_error": v1.get("error"), "v3_error": v3.get("error")})
            print(f"{'OK  ' if same else 'DIFF'} val={int(validated)} v1={v1['status']:4} v3={v3['status']:4} {name}"
                  + ("" if same else f"\n      v1: {v1.get('error')}\n      v3: {v3.get('error')}"), flush=True)
    a.out.write_text(json.dumps(rows, indent=1) + "\n")
    print("agree", sum(r["agree"] for r in rows), "/", len(rows), "pool_len", L)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bodies", type=Path, required=True)
    p.add_argument("--format", choices=("openai", "compact"), required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--client3", required=True)
    p.add_argument("--token-pool", required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
