//! Response parsing + validation, equivalent in strength to genopt-client.py
//! (`parse_and_validate_reply`, `validate_compact`, `validate_openai`).
//!
//! Every body: UTF-8 check + full JSON grammar validation of the whole
//! document (a parse that would make Python's json.loads fail fails here),
//! request_id in {mock-%04d, generate-tokens-mock-%04d}, exactly one choice,
//! finish_reason == "abort", len(token_ids) == n and token_ids == expected
//! sampled ids (v1 checks the ids only for validated requests: stronger).
//! Compact: all three base64 arrays decoded strictly with size checks.
//! Validated (index < validate_requests): v1's full per-position semantics.

use crate::json::{py_eq, Mem, V, P, R};
use std::collections::{BTreeSet, HashMap};

pub const TOP_K: usize = 128;

/// float32 bit patterns of genopt-client.py `expected_scores()`, produced by
/// the cluster's numpy (pairwise sum / exp / log exactly as the Python client
/// computes them). `scores_selfcheck` recomputes them in Rust as a sanity check.
pub const SCORE_BITS: [u32; TOP_K] = [
    0xc0539d2c, 0xc055fb61, 0xc0585996, 0xc05ab7cb, 0xc05d1601, 0xc05f7436, 0xc061d26b, 0xc06430a0, 0xc0668ed6,
    0xc068ed0b, 0xc06b4b40, 0xc06da975, 0xc07007ab, 0xc07265e0, 0xc074c415, 0xc077224a, 0xc0798080, 0xc07bdeb5,
    0xc07e3cea, 0xc0804d90, 0xc0817caa, 0xc082abc5, 0xc083dae0, 0xc08509fa, 0xc0863915, 0xc087682f, 0xc088974a,
    0xc089c665, 0xc08af57f, 0xc08c249a, 0xc08d53b5, 0xc08e82cf, 0xc08fb1ea, 0xc090e104, 0xc092101f, 0xc0933f3a,
    0xc0946e54, 0xc0959d6f, 0xc096cc8a, 0xc097fba4, 0xc0992abf, 0xc09a59d9, 0xc09b88f4, 0xc09cb80f, 0xc09de729,
    0xc09f1644, 0xc0a0455f, 0xc0a17479, 0xc0a2a394, 0xc0a3d2ae, 0xc0a501c9, 0xc0a630e4, 0xc0a75ffe, 0xc0a88f19,
    0xc0a9be34, 0xc0aaed4e, 0xc0ac1c69, 0xc0ad4b83, 0xc0ae7a9e, 0xc0afa9b9, 0xc0b0d8d3, 0xc0b207ee, 0xc0b33709,
    0xc0b46623, 0xc0b5953e, 0xc0b6c458, 0xc0b7f373, 0xc0b9228e, 0xc0ba51a8, 0xc0bb80c3, 0xc0bcafde, 0xc0bddef8,
    0xc0bf0e13, 0xc0c03d2d, 0xc0c16c48, 0xc0c29b63, 0xc0c3ca7d, 0xc0c4f998, 0xc0c628b3, 0xc0c757cd, 0xc0c886e8,
    0xc0c9b602, 0xc0cae51d, 0xc0cc1438, 0xc0cd4352, 0xc0ce726d, 0xc0cfa188, 0xc0d0d0a2, 0xc0d1ffbd, 0xc0d32ed7,
    0xc0d45df2, 0xc0d58d0d, 0xc0d6bc27, 0xc0d7eb42, 0xc0d91a5d, 0xc0da4977, 0xc0db7892, 0xc0dca7ac, 0xc0ddd6c7,
    0xc0df05e2, 0xc0e034fc, 0xc0e16417, 0xc0e29331, 0xc0e3c24c, 0xc0e4f167, 0xc0e62081, 0xc0e74f9c, 0xc0e87eb7,
    0xc0e9add1, 0xc0eadcec, 0xc0ec0c06, 0xc0ed3b21, 0xc0ee6a3c, 0xc0ef9956, 0xc0f0c871, 0xc0f1f78c, 0xc0f326a6,
    0xc0f455c1, 0xc0f584db, 0xc0f6b3f6, 0xc0f7e311, 0xc0f9122b, 0xc0fa4146, 0xc0fb7061, 0xc0fc9f7b, 0xc0fdce96,
    0xc0fefdb0, 0xc1001666,
];

/// Recomputes expected_scores() in Rust (numpy pairwise sum for n=128: eight
/// interleaved accumulators). Returns the number of mismatching entries.
pub fn scores_selfcheck() -> usize {
    let s: Vec<f64> = (0..TOP_K).map(|r| -(r as f64) * 0.037).collect();
    let e: Vec<f64> = s.iter().map(|x| x.exp()).collect();
    let mut acc = [0f64; 8];
    for i in (0..TOP_K).step_by(8) {
        for j in 0..8 {
            acc[j] += e[i + j];
        }
    }
    let sum = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    let l = sum.ln();
    (0..TOP_K).filter(|&r| ((s[r] - l) as f32).to_bits() != SCORE_BITS[r]).count()
}

pub struct Cfg {
    pub n: usize,
    pub compact: bool,
    pub pool: Vec<i64>,
    pub pool_tok: Vec<Vec<u8>>, // "token_id:<id>"
    pub scores: [f32; TOP_K],
    /// Test hook (negative control): flips the expectation for one position.
    pub corrupt_expected_position: Option<usize>,
    /// Amendment v3: compact_include_sampled (false: num_slots = k, engine
    /// slots 1..k only, block carries "sampled_slot": false).
    pub include_sampled: bool,
    /// Amendment v3: compact_include_ranks (false: no `ranks` key).
    pub include_ranks: bool,
    /// R3: routed_experts layers (0 = off: routed_experts must be absent/null).
    pub routed_layers: usize,
    /// R3 rows = input_tokens + output_tokens - 1.
    pub routed_rows: usize,
}

impl Cfg {
    pub fn new(n: usize, compact: bool, pool: Vec<i64>) -> Self {
        let pool_tok = pool.iter().map(|id| format!("token_id:{id}").into_bytes()).collect();
        let mut scores = [0f32; TOP_K];
        for r in 0..TOP_K {
            scores[r] = f32::from_bits(SCORE_BITS[r]);
        }
        Cfg {
            n,
            compact,
            pool,
            pool_tok,
            scores,
            corrupt_expected_position: None,
            include_sampled: true,
            include_ranks: true,
            routed_layers: 0,
            routed_rows: 0,
        }
    }

    #[inline]
    fn sampled_index(&self, p: usize) -> usize {
        (p + p % TOP_K) % self.pool.len()
    }

    #[inline]
    pub fn sampled(&self, p: usize) -> i64 {
        let v = self.pool[self.sampled_index(p)];
        if self.corrupt_expected_position == Some(p) {
            v ^ 1
        } else {
            v
        }
    }
}

#[derive(Default)]
pub struct Stats {
    pub top_entries: BTreeSet<usize>,
    pub validated: bool,
}

/// math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
#[inline]
fn isclose(a: f64, b: f64) -> bool {
    if a == b {
        return true;
    }
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    let d = (b - a).abs();
    d <= (1e-6 * b).abs() || d <= (1e-6 * a).abs() || d <= 1e-7
}

pub struct Scratch {
    pub buf: Vec<u8>,
    pub routed: Vec<u8>,
}

impl Scratch {
    pub fn new() -> Self {
        Scratch { buf: Vec::new(), routed: Vec::new() }
    }
}

/// Parses and checks one response body. `index` is the request index.
pub fn check_body<'a>(
    body: &'a [u8],
    index: usize,
    validated: bool,
    cfg: &Cfg,
    st: &mut Stats,
    sc: &mut Scratch,
    mem: &'a Mem<'a>,
) -> R<()> {
    st.validated = validated;
    // UTF-8: the parser consumes every byte; outside strings only ASCII
    // grammar bytes are accepted, and every string containing a non-ASCII
    // byte is UTF-8 validated (== Python decoding the whole document first).
    let mut p = P::with_mem(body, mem);
    let mut request_id: Option<V> = None;
    let mut choices: Option<usize> = None;
    p.object_unique(|p, key| {
        match key {
            b"request_id" => request_id = Some(p.value()?),
            b"choices" => {
                if p.peek() != Some(b'[') {
                    return p.err("choices is not a list");
                }
                let count = p.array(|p, k| if k == 0 { check_choice(p, validated, cfg, st, sc) } else { p.skip_value() })?;
                choices = Some(count);
            }
            _ => p.skip_value()?,
        }
        Ok(())
    })?;
    p.ws();
    if p.i != body.len() {
        return p.err("extra data after document");
    }
    let rid = match request_id {
        Some(V::Str(s)) => s,
        other => return Err(format!("request_id missing or not a string: {other:?}")),
    };
    let a = format!("mock-{index:04}");
    let b = format!("generate-tokens-mock-{index:04}");
    if rid != a.as_bytes() && rid != b.as_bytes() {
        return Err(format!("request_id {:?} != {a} / {b}", String::from_utf8_lossy(&rid)));
    }
    match choices {
        Some(1) => Ok(()),
        other => Err(format!("len(choices) != 1: {other:?}")),
    }
}

#[derive(PartialEq, Debug)]
enum LpKind {
    Missing,
    Null,
    Other,
}

fn check_choice(p: &mut P, validated: bool, cfg: &Cfg, st: &mut Stats, sc: &mut Scratch) -> R<()> {
    let n = cfg.n;
    let mut finish: Option<V> = None;
    let mut token_ids: Option<usize> = None;
    let mut lp_kind = LpKind::Missing;
    let mut content_len: Option<usize> = None;
    let mut compact_seen = false;
    let mut routed_seen = false;
    if p.peek() != Some(b'{') {
        return p.err("choice is not an object");
    }
    p.object_unique(|p, key| {
        match key {
            b"finish_reason" => finish = Some(p.value()?),
            b"token_ids" => {
                if p.peek() != Some(b'[') {
                    return p.err("token_ids is not a list");
                }
                let count = p.array(|p, k| {
                    let (s, e, float) = p.scan_number()?;
                    if float {
                        return p.err("token id is not an integer");
                    }
                    let v: i64 = std::str::from_utf8(&p.b[s..e])
                        .unwrap()
                        .parse()
                        .map_err(|_| format!("token id out of range at {s}"))?;
                    if k < n && v != cfg.sampled(k) {
                        return Err(format!("token_ids[{k}] = {v} != expected sampled {}", cfg.sampled(k)));
                    }
                    Ok(())
                })?;
                token_ids = Some(count);
            }
            b"logprobs" => {
                if p.peek() == Some(b'n') {
                    p.skip_value()?;
                    lp_kind = LpKind::Null;
                } else if cfg.compact {
                    p.skip_value()?;
                    lp_kind = LpKind::Other;
                } else {
                    lp_kind = LpKind::Other;
                    if p.peek() != Some(b'{') {
                        return p.err("logprobs is not an object");
                    }
                    p.object_unique(|p, k| {
                        if k == b"content" {
                            if p.peek() != Some(b'[') {
                                return p.err("logprobs.content is not a list");
                            }
                            let count = if validated { validate_content(p, cfg, st)? } else { p.skip_array_count()? };
                            if count != n {
                                return Err(format!("len(logprobs.content) {count} != {n}"));
                            }
                            content_len = Some(count);
                            Ok(())
                        } else {
                            p.skip_value()
                        }
                    })?;
                }
            }
            b"compact_logprobs" if cfg.compact => {
                check_compact(p, validated, cfg, sc)?;
                compact_seen = true;
            }
            b"routed_experts" => {
                if cfg.routed_layers == 0 {
                    // v1: choice.get("routed_experts") is None
                    if !matches!(p.value()?, V::Null) {
                        return Err("routed_experts present but --routed-experts-layers is 0".into());
                    }
                } else {
                    check_routed(p, validated, cfg, sc)?;
                    routed_seen = true;
                }
            }
            _ => p.skip_value()?,
        }
        Ok(())
    })?;
    match finish {
        Some(V::Str(ref s)) if s == b"abort" => {}
        other => return Err(format!("finish_reason {other:?} != abort")),
    }
    match token_ids {
        Some(c) if c == n => {}
        other => return Err(format!("len(token_ids) {other:?} != {n}")),
    }
    if cfg.routed_layers > 0 && !routed_seen {
        return Err("routed_experts missing".into());
    }
    if cfg.compact {
        if !compact_seen {
            return Err("compact_logprobs missing".into());
        }
        if validated && lp_kind == LpKind::Other {
            return Err("compact: choices[0].logprobs is not None".into());
        }
    } else if content_len != Some(n) {
        return Err(format!("openai: logprobs.content missing ({lp_kind:?})"));
    }
    Ok(())
}

/// One base64 JSON string: strict decode into `buf`, returns decoded length.
fn b64_field(p: &mut P, buf: &mut Vec<u8>) -> R<usize> {
    if p.peek() != Some(b'"') {
        return p.err("compact array is not a string");
    }
    let start = p.i + 1;
    let end = match memchr::memchr(b'"', &p.b[start..]) {
        Some(off) => start + off,
        None => return p.err("unterminated string"),
    };
    let raw = &p.b[start..end];
    let need = raw.len() / 4 * 3 + 3;
    grow_buf(p.mem, buf, need)?;
    match crate::b64::decode(raw, &mut buf[..need]) {
        // Success means every byte is in the base64 alphabet, so the JSON
        // string contained no escapes / control characters and ended here.
        Ok(len) => {
            p.i = end + 1;
            Ok(len)
        }
        Err(raw_err) => {
            // Escaped JSON string (e.g. "\/"): decode the JSON string first, like Python.
            let (s, e, esc) = p.scan_string()?;
            if !esc {
                return Err(format!("base64 decode failed: {raw_err}"));
            }
            // charged for its lifetime: the decoded JSON string
            let mem = p.mem;
            let text_bytes = (e - s) as u64;
            mem.charge(text_bytes)?;
            let r = (|| -> R<usize> {
                let text = p.decode(s, e, esc)?;
                let need = text.len() / 4 * 3 + 3;
                grow_buf(mem, buf, need)?;
                crate::b64::decode(&text, &mut buf[..need]).map_err(|e| format!("base64 decode failed: {e}"))
            })();
            mem.refund(text_bytes);
            r
        }
    }
}

/// Grows a scratch buffer to `need` bytes, charging the growth first; fallible.
fn grow_buf(mem: &Mem, buf: &mut Vec<u8>, need: usize) -> R<()> {
    if buf.len() < need {
        let add = need - buf.len();
        mem.charge(add as u64)?;
        buf.try_reserve_exact(add).map_err(|e| format!("scratch allocation of {need} bytes failed: {e}"))?;
        buf.resize(need, 0);
    }
    Ok(())
}

/// R3: choices[0].routed_experts = base64 of a .npy (uint8, shape (rows, L, 8),
/// C order). Strict header (stricter than np.load): magic, version 1.0/2.0,
/// ASCII dict literal ending in '\n' with exactly descr '|u1', fortran_order
/// False, shape (rows, L, 8); data exactly rows*L*8 bytes. Every request is
/// decoded and header-checked; validated requests check every value.
fn check_routed(p: &mut P, validated: bool, cfg: &Cfg, sc: &mut Scratch) -> R<()> {
    if p.peek() != Some(b'"') {
        return p.err("routed_experts is not a string");
    }
    let len = b64_field(p, &mut sc.routed)?;
    let raw = &sc.routed[..len];
    let (rows, layers) = (cfg.routed_rows, cfg.routed_layers);
    let data = npy_u8_3d(raw, (rows, layers, 8)).map_err(|e| format!("routed_experts npy: {e}"))?;
    if validated {
        for t in 0..rows {
            for l in 0..layers {
                let base = (t * layers + l) * 8;
                for j in 0..8 {
                    let want = ((t * 7 + l * 13 + j * 31) % 256) as u8;
                    if data[base + j] != want {
                        return Err(format!("routed_experts[{t}][{l}][{j}] = {} != {want}", data[base + j]));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Parses a .npy buffer; returns its data section.
pub fn npy_u8_3d(raw: &[u8], shape: (usize, usize, usize)) -> R<&[u8]> {
    if raw.len() < 10 || &raw[..6] != b"\x93NUMPY" {
        return Err("bad magic".into());
    }
    let (hlen, start): (usize, usize) = match (raw[6], raw[7]) {
        (1, 0) => (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10),
        (2, 0) => {
            if raw.len() < 12 {
                return Err("truncated header".into());
            }
            (u32::from_le_bytes(raw[8..12].try_into().unwrap()) as usize, 12)
        }
        (a, b) => return Err(format!("unsupported version {a}.{b}")),
    };
    let end = start.checked_add(hlen).ok_or("header length overflow")?;
    if end > raw.len() {
        return Err("truncated header".into());
    }
    let header = &raw[start..end];
    if !header.is_ascii() || header.last() != Some(&b'\n') {
        return Err("header not ASCII or not newline-terminated".into());
    }
    let h = std::str::from_utf8(header).unwrap();
    let (descr, fortran, dims) = parse_npy_dict(h.trim_end_matches('\n').trim_end_matches(' '))?;
    if descr != "|u1" {
        return Err(format!("descr {descr:?} != '|u1'"));
    }
    if fortran {
        return Err("fortran_order True".into());
    }
    if dims != [shape.0, shape.1, shape.2] {
        return Err(format!("shape {dims:?} != {shape:?}"));
    }
    let want = shape.0.checked_mul(shape.1).and_then(|x| x.checked_mul(shape.2)).ok_or("shape overflow")?;
    let data = &raw[end..];
    if data.len() != want {
        return Err(format!("data {} bytes != {want}", data.len()));
    }
    Ok(data)
}

/// Parses the numpy header dict literal: exactly the keys descr (str),
/// fortran_order (bool) and shape (tuple of non-negative ints), each once.
fn parse_npy_dict(h: &str) -> R<(String, bool, Vec<usize>)> {
    let b = h.as_bytes();
    let mut i = 0;
    let ws = |i: &mut usize| {
        while *i < b.len() && b[*i] == b' ' {
            *i += 1;
        }
    };
    let quoted = |i: &mut usize| -> R<String> {
        let q = *b.get(*i).ok_or("eof")?;
        if q != b'\'' && q != b'"' {
            return Err("expected a quoted string".into());
        }
        let s = *i + 1;
        let e = s + b[s..].iter().position(|&c| c == q).ok_or("unterminated string")?;
        let v = &h[s..e];
        if v.contains('\\') {
            return Err("escapes not supported".into());
        }
        *i = e + 1;
        Ok(v.to_string())
    };
    ws(&mut i);
    if b.get(i) != Some(&b'{') {
        return Err("header is not a dict".into());
    }
    i += 1;
    let (mut descr, mut fortran, mut shape) = (None, None, None);
    loop {
        ws(&mut i);
        if b.get(i) == Some(&b'}') {
            i += 1;
            break;
        }
        let key = quoted(&mut i)?;
        ws(&mut i);
        if b.get(i) != Some(&b':') {
            return Err("expected ':'".into());
        }
        i += 1;
        ws(&mut i);
        match key.as_str() {
            "descr" if descr.is_none() => descr = Some(quoted(&mut i)?),
            "fortran_order" if fortran.is_none() => {
                if h[i..].starts_with("False") {
                    fortran = Some(false);
                    i += 5;
                } else if h[i..].starts_with("True") {
                    fortran = Some(true);
                    i += 4;
                } else {
                    return Err("fortran_order is not a bool".into());
                }
            }
            "shape" if shape.is_none() => {
                if b.get(i) != Some(&b'(') {
                    return Err("shape is not a tuple".into());
                }
                i += 1;
                let mut dims = vec![];
                loop {
                    ws(&mut i);
                    if b.get(i) == Some(&b')') {
                        i += 1;
                        break;
                    }
                    let s = i;
                    while i < b.len() && b[i].is_ascii_digit() {
                        i += 1;
                    }
                    let t = &h[s..i];
                    if t.is_empty() || (t.len() > 1 && t.starts_with('0')) || t.len() > 18 {
                        return Err(format!("bad shape dimension {t:?}"));
                    }
                    dims.push(t.parse::<usize>().unwrap());
                    ws(&mut i);
                    match b.get(i) {
                        Some(b',') => i += 1,
                        Some(b')') => {
                            i += 1;
                            break;
                        }
                        _ => return Err("bad shape tuple".into()),
                    }
                }
                if dims.len() == 1 && !h[..i].trim_end_matches(')').trim_end().ends_with(',') {
                    return Err("(x) is not a tuple".into());
                }
                shape = Some(dims);
            }
            other => return Err(format!("unexpected or duplicate header key {other:?}")),
        }
        ws(&mut i);
        match b.get(i) {
            Some(b',') => i += 1,
            Some(b'}') => {
                i += 1;
                break;
            }
            _ => return Err("expected ',' or '}'".into()),
        }
    }
    ws(&mut i);
    if i != b.len() {
        return Err("trailing data after header dict".into());
    }
    match (descr, fortran, shape) {
        (Some(d), Some(f), Some(s)) => Ok((d, f, s)),
        _ => Err("header must have descr, fortran_order and shape".into()),
    }
}

#[inline]
fn word(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap())
}

fn check_compact(p: &mut P, validated: bool, cfg: &Cfg, sc: &mut Scratch) -> R<()> {
    let n = cfg.n;
    let slots = if cfg.include_sampled { TOP_K + 1 } else { TOP_K };
    let first = if cfg.include_sampled { 1 } else { 0 }; // column of engine slot 1
    let l = cfg.pool.len();
    let mut sampled_slot: Option<V> = None;
    let mut num_positions: Option<V> = None;
    let mut num_slots: Option<V> = None;
    let mut byteorder: Option<V> = None;
    let mut dt_ids: Option<V> = None;
    let mut dt_lps: Option<V> = None;
    let mut seen = [false; 3];
    if p.peek() != Some(b'{') {
        return p.err("compact_logprobs is not an object");
    }
    p.object_unique(|p, key| {
        match key {
            b"num_positions" => num_positions = Some(p.value()?),
            b"num_slots" => num_slots = Some(p.value()?),
            b"sampled_slot" => sampled_slot = Some(p.value()?),
            b"byteorder" => byteorder = Some(p.value()?),
            b"dtype_token_ids" => dt_ids = Some(p.value()?),
            b"dtype_logprobs" => dt_lps = Some(p.value()?),
            b"token_ids" => {
                let len = b64_field(p, &mut sc.buf)?;
                if len != n * slots * 4 {
                    return Err(format!("compact token_ids size {} != {}", len / 4, n * slots));
                }
                if validated {
                    let b = &sc.buf[..len];
                    for pos in 0..n {
                        let row = pos * slots;
                        if cfg.include_sampled && word(b, row) != cfg.sampled(pos) as i32 as u32 {
                            return Err(format!("compact token_ids[{pos}][0] mismatch"));
                        }
                        for r in 0..TOP_K {
                            if word(b, row + first + r) != cfg.pool[(pos + r) % l] as i32 as u32 {
                                return Err(format!("compact token_ids[{pos}][{}] mismatch", first + r));
                            }
                        }
                    }
                }
                seen[0] = true;
            }
            b"logprobs" => {
                let len = b64_field(p, &mut sc.buf)?;
                if len != n * slots * 4 {
                    return Err(format!("compact logprobs size {} != {}", len / 4, n * slots));
                }
                if validated {
                    let b = &sc.buf[..len];
                    let bits: Vec<u32> = cfg.scores.iter().map(|s| s.to_bits()).collect();
                    for pos in 0..n {
                        let row = pos * slots;
                        if cfg.include_sampled && word(b, row) != bits[pos % TOP_K] {
                            return Err(format!("compact logprobs[{pos}][0] bits mismatch"));
                        }
                        for r in 0..TOP_K {
                            if word(b, row + first + r) != bits[r] {
                                return Err(format!("compact logprobs[{pos}][{}] bits mismatch", first + r));
                            }
                        }
                    }
                }
                seen[1] = true;
            }
            b"ranks" => {
                if !cfg.include_ranks {
                    return Err("compact ranks present but compact_include_ranks is false".into());
                }
                let len = b64_field(p, &mut sc.buf)?;
                if len != n * 4 {
                    return Err(format!("compact ranks size {} != {n}", len / 4));
                }
                if validated {
                    for pos in 0..n {
                        if word(&sc.buf, pos) != (pos % TOP_K + 1) as u32 {
                            return Err(format!("compact ranks[{pos}] mismatch"));
                        }
                    }
                }
                seen[2] = true;
            }
            _ => p.skip_value()?,
        }
        Ok(())
    })?;
    if seen != [true, true, cfg.include_ranks] {
        return Err(format!("compact arrays missing: {seen:?} (include_ranks={})", cfg.include_ranks));
    }
    match (&sampled_slot, cfg.include_sampled) {
        (None, true) => {}
        // rs-pr-stack round 15b: the compact block no longer carries the
        // constant "sampled_slot": false (slot 0 is never included).
        (None, false) | (Some(V::Bool(false)), false) => {}
        (other, inc) => return Err(format!("compact sampled_slot {other:?} with include_sampled={inc}")),
    }
    let is_int = |v: &Option<V>, want: usize| matches!(v, Some(V::Int(i)) if *i == want as i128);
    let is_str = |v: &Option<V>, want: &[u8]| matches!(v, Some(V::Str(s)) if s == want);
    if !is_int(&num_positions, n) {
        return Err(format!("compact num_positions {num_positions:?} != {n}"));
    }
    if !is_int(&num_slots, slots) {
        return Err(format!("compact num_slots {num_slots:?} != {slots}"));
    }
    if !is_str(&dt_ids, b"int32") || !is_str(&dt_lps, b"float32") {
        return Err(format!("compact dtypes {dt_ids:?} / {dt_lps:?}"));
    }
    if !is_str(&byteorder, b"little") {
        return Err(format!("compact byteorder {byteorder:?}"));
    }
    Ok(())
}

/// validate_openai(): positions < min(period, n) checked semantically, the
/// rest must equal the template row of the same phase (Python ==).
fn validate_content(p: &mut P, cfg: &Cfg, st: &mut Stats) -> R<usize> {
    let n = cfg.n;
    let l = cfg.pool.len();
    let m = l.min(n);
    // (raw span, parsed row)
    let mut templates: Vec<(usize, usize, V)> = Vec::with_capacity(m);
    let count = p.array(|p, pos| {
        let s = p.i;
        if pos < m {
            let v = p.value()?;
            check_row(&v, pos, cfg, st).map_err(|e| format!("content[{pos}]: {e}"))?;
            templates.push((s, p.i, v));
        } else {
            p.skip_value()?;
            if pos < n {
                let (ts, te, ref tv) = templates[pos % l];
                // identical text => identical json.loads value => Python ==
                // (NaN included: json's NaN is a singleton, see json::py_eq)
                if p.b[s..p.i] != p.b[ts..te] {
                    // temporary tree: charged while it exists, then refunded
                    let used0 = p.mem.used();
                    let v = P::with_mem(&p.b[s..p.i], p.mem).value();
                    let equal: R<bool> = match &v {
                        Ok(v) => Ok(py_eq(v, tv)),
                        Err(e) => Err(e.clone()),
                    };
                    drop(v);
                    p.mem.refund(p.mem.used().saturating_sub(used0));
                    if !equal? {
                        return Err(format!("content[{pos}] != content[{}]", pos % l));
                    }
                }
            }
        }
        Ok(())
    })?;
    Ok(count)
}

fn check_row(row: &V, pos: usize, cfg: &Cfg, st: &mut Stats) -> R<()> {
    let l = cfg.pool.len();
    if !matches!(row, V::Obj(_)) {
        return Err("row is not an object".into());
    }
    let token = row.get(b"token").ok_or("KeyError token")?;
    let want = &cfg.pool_tok[cfg.sampled_index(pos)];
    let want_owned;
    let want: &[u8] = if cfg.corrupt_expected_position == Some(pos) {
        want_owned = format!("token_id:{}", cfg.sampled(pos)).into_bytes();
        &want_owned
    } else {
        want
    };
    match token {
        V::Str(s) if s.as_slice() == want => {}
        other => return Err(format!("token {other:?} != {}", String::from_utf8_lossy(want))),
    }
    let lp = row.get(b"logprob").ok_or("KeyError logprob")?.as_f64().ok_or("logprob not a number")?;
    if !isclose(lp, cfg.scores[pos % TOP_K] as f64) {
        return Err(format!("sampled logprob {lp} !~ {}", cfg.scores[pos % TOP_K]));
    }
    let mut expected: HashMap<&[u8], f64> = HashMap::with_capacity(TOP_K);
    for r in 0..TOP_K {
        expected.insert(cfg.pool_tok[(pos + r) % l].as_slice(), cfg.scores[r] as f64);
    }
    let tops = match row.get(b"top_logprobs") {
        Some(V::Arr(a)) => a,
        other => return Err(format!("top_logprobs not a list: {:?}", other.map(|_| "..."))),
    };
    st.top_entries.insert(tops.len());
    let mut got: HashMap<&[u8], &V> = HashMap::with_capacity(tops.len());
    for entry in tops {
        if !matches!(entry, V::Obj(_)) {
            return Err("top_logprobs entry is not an object".into());
        }
        let t = entry.get(b"token").ok_or("KeyError entry token")?;
        let v = entry.get(b"logprob").ok_or("KeyError entry logprob")?;
        match t {
            V::Str(s) => {
                got.entry(s.as_slice()).or_insert(v);
            }
            V::Arr(_) | V::Obj(_) => return Err("unhashable entry token".into()),
            _ => return Err("non-string entry token (key set differs)".into()),
        }
    }
    if got.len() != expected.len() || !got.keys().all(|k| expected.contains_key(k)) {
        return Err(format!("top_logprobs key set differs ({} keys)", got.len()));
    }
    for (k, v) in &got {
        let x = v.as_f64().ok_or("entry logprob not a number")?;
        if !isclose(x, expected[k]) {
            return Err(format!("entry {} logprob {x} !~ {}", String::from_utf8_lossy(k), expected[k]));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine as _;

    fn cfg(n: usize, compact: bool) -> Cfg {
        Cfg::new(n, compact, (1000..1256).collect()) // len multiple of 128, like the mock pool (1024)
    }

    fn openai_body(c: &Cfg, extra_choice: &str) -> String {
        let n = c.n;
        let l = c.pool.len();
        let rows: Vec<String> = (0..n)
            .map(|p| {
                let tops: Vec<String> = (0..TOP_K)
                    .map(|r| format!("{{\"token\":\"token_id:{}\",\"logprob\":{},\"bytes\":[1,2]}}", c.pool[(p + r) % l], c.scores[r] as f64))
                    .collect();
                format!(
                    "{{\"token\":\"token_id:{}\",\"logprob\":{},\"bytes\":null,\"top_logprobs\":[{}]}}",
                    c.sampled(p),
                    c.scores[p % TOP_K] as f64,
                    tops.join(",")
                )
            })
            .collect();
        let ids: Vec<String> = (0..n).map(|p| c.sampled(p).to_string()).collect();
        format!(
            "{{\"request_id\":\"mock-0000\",\"choices\":[{{\"index\":0,\"logprobs\":{{\"content\":[{}]}},\"finish_reason\":\"abort\",\"token_ids\":[{}]{extra_choice}}}]}}",
            rows.join(","),
            ids.join(",")
        )
    }

    fn compact_body(c: &Cfg, extra_block: &str) -> String {
        let n = c.n;
        let l = c.pool.len();
        let (mut ids, mut lps, mut ranks) = (vec![], vec![], vec![]);
        for p in 0..n {
            ids.extend_from_slice(&(c.sampled(p) as i32).to_le_bytes());
            lps.extend_from_slice(&c.scores[p % TOP_K].to_le_bytes());
            for r in 0..TOP_K {
                ids.extend_from_slice(&(c.pool[(p + r) % l] as i32).to_le_bytes());
                lps.extend_from_slice(&c.scores[r].to_le_bytes());
            }
            ranks.extend_from_slice(&((p % TOP_K + 1) as i32).to_le_bytes());
        }
        let e = base64::engine::general_purpose::STANDARD;
        let tok: Vec<String> = (0..n).map(|p| c.sampled(p).to_string()).collect();
        format!(
            "{{\"request_id\":\"generate-tokens-mock-0000\",\"choices\":[{{\"logprobs\":null,\"finish_reason\":\"abort\",\"token_ids\":[{}],\"compact_logprobs\":{{\"num_positions\":{n},\"num_slots\":129,\"dtype_token_ids\":\"int32\",\"dtype_logprobs\":\"float32\",\"byteorder\":\"little\",\"token_ids\":\"{}\",\"logprobs\":\"{}\",\"ranks\":\"{}\"{extra_block}}}}}]}}",
            tok.join(","),
            e.encode(&ids),
            e.encode(&lps),
            e.encode(&ranks)
        )
    }

    fn npy(header: &str, data: &[u8], v2: bool) -> Vec<u8> {
        let mut h = header.to_string();
        let pre = if v2 { 12 } else { 10 };
        while (pre + h.len() + 1) % 64 != 0 {
            h.push(' ');
        }
        h.push('\n');
        let mut out = b"\x93NUMPY".to_vec();
        if v2 {
            out.extend_from_slice(&[2, 0]);
            out.extend_from_slice(&(h.len() as u32).to_le_bytes());
        } else {
            out.extend_from_slice(&[1, 0]);
            out.extend_from_slice(&(h.len() as u16).to_le_bytes());
        }
        out.extend_from_slice(h.as_bytes());
        out.extend_from_slice(data);
        out
    }

    #[test]
    fn npy_header() {
        let hdr = "{'descr': '|u1', 'fortran_order': False, 'shape': (3, 2, 8), }";
        let data = vec![0u8; 48];
        for v2 in [false, true] {
            assert!(npy_u8_3d(&npy(hdr, &data, v2), (3, 2, 8)).is_ok());
        }
        let bad = [
            "{'descr': '<u1', 'fortran_order': False, 'shape': (3, 2, 8), }",
            "{'descr': '|u1', 'fortran_order': True, 'shape': (3, 2, 8), }",
            "{'descr': '|u1', 'fortran_order': False, 'shape': (3, 2, 9), }",
            "{'descr': '|u1', 'fortran_order': False, 'shape': (3, 2), }",
            "{'descr': '|u1', 'shape': (3, 2, 8), }",
            "{'descr': '|u1', 'fortran_order': False, 'shape': (3, 2, 8), 'x': 1}",
            "{'descr': '|u1', 'descr': '|u1', 'fortran_order': False, 'shape': (3, 2, 8)}",
            "{'descr': '|u1', 'fortran_order': False, 'shape': (03, 2, 8), }",
        ];
        for h in bad {
            assert!(npy_u8_3d(&npy(h, &data, false), (3, 2, 8)).is_err(), "{h}");
        }
        assert!(npy_u8_3d(&npy(hdr, &data[..47], false), (3, 2, 8)).is_err());
        let mut long = data.clone();
        long.push(0);
        assert!(npy_u8_3d(&npy(hdr, &long, false), (3, 2, 8)).is_err());
        let mut v3 = npy(hdr, &data, false);
        v3[6] = 3;
        assert!(npy_u8_3d(&v3, (3, 2, 8)).is_err());
        assert!(npy_u8_3d(&v3[..5], (3, 2, 8)).is_err());
        // double-quoted keys and no trailing comma are valid literal_eval dicts
        assert!(npy_u8_3d(&npy("{\"descr\": \"|u1\", \"fortran_order\": False, \"shape\": (3,2,8)}", &data, false), (3, 2, 8)).is_ok());
    }

    #[test]
    fn compact_switches_and_routed() {
        let e = base64::engine::general_purpose::STANDARD;
        let mut c = cfg(600, true);
        c.include_sampled = false;
        c.include_ranks = false;
        c.routed_layers = 2;
        c.routed_rows = 5;
        let mut vals = vec![];
        for t in 0..5usize {
            for l in 0..2usize {
                for j in 0..8usize {
                    vals.push(((t * 7 + l * 13 + j * 31) % 256) as u8);
                }
            }
        }
        let hdr = "{'descr': '|u1', 'fortran_order': False, 'shape': (5, 2, 8), }";
        let routed = |v: &[u8]| format!(",\"routed_experts\":\"{}\"", e.encode(npy(hdr, v, false)));
        // build a no-sampled/no-ranks block from the full one by re-encoding
        let full = compact_body(&cfg(600, true), "");
        let body = to_no_sampled_no_ranks(&full, 600);
        let ok = fix_tail(body.clone(), &routed(&vals));
        for v in [true, false] {
            assert_eq!(run(&c, &ok, v), Ok(()), "validated={v}");
        }
        let mut bad = vals.clone();
        bad[17] ^= 1;
        let wrong = fix_tail(body.clone(), &routed(&bad));
        assert!(run(&c, &wrong, true).is_err());
        assert_eq!(run(&c, &wrong, false), Ok(())); // values checked only when validated (as v1)
        let missing = fix_tail(body.clone(), "");
        assert!(run(&c, &missing, false).is_err());
        let null = fix_tail(body.clone(), ",\"routed_experts\":null");
        assert!(run(&c, &null, false).is_err());
        // routed off: null/absent ok, present fails
        c.routed_layers = 0;
        assert_eq!(run(&c, &null, false), Ok(()));
        assert_eq!(run(&c, &missing, false), Ok(()));
        assert!(run(&c, &ok, false).is_err());
        // switches: sampled_slot must be false or absent (round 15b), ranks must be absent
        assert!(run(&c, &missing.replace("\"sampled_slot\":false", "\"sampled_slot\":0"), true).is_err());
        assert_eq!(run(&c, &missing.replace(",\"sampled_slot\":false", ""), true), Ok(()));
        c.include_ranks = true;
        assert!(run(&c, &missing, false).is_err());
        c.include_ranks = false;
        c.include_sampled = true;
        assert!(run(&c, &missing, false).is_err());
    }

    /// Rewrites a full compact test body into include_sampled=false / include_ranks=false form.
    fn to_no_sampled_no_ranks(full: &str, n: usize) -> String {
        let e = base64::engine::general_purpose::STANDARD;
        let field = |name: &str| -> (usize, usize) {
            let key = format!("\"{name}\":\"");
            let s = full.rfind(&key).unwrap() + key.len();
            (s, s + full[s..].find('"').unwrap())
        };
        let strip = |name: &str| -> String {
            let (s, t) = field(name);
            let raw = e.decode(&full[s..t]).unwrap();
            let mut out = vec![];
            for p in 0..n {
                out.extend_from_slice(&raw[(p * 129 + 1) * 4..(p * 129 + 129) * 4]);
            }
            e.encode(out)
        };
        let ids = strip("token_ids");
        let lps = strip("logprobs");
        let (rs, _) = field("ranks");
        let block_start = full.find("\"compact_logprobs\":{").unwrap();
        let head = &full[..block_start];
        let ids_tok = &full[full.find("\"token_ids\":[").unwrap()..];
        let _ = (rs, ids_tok);
        format!(
            "{head}\"compact_logprobs\":{{\"num_positions\":{n},\"num_slots\":128,\"sampled_slot\":false,\"dtype_token_ids\":\"int32\",\"dtype_logprobs\":\"float32\",\"byteorder\":\"little\",\"token_ids\":\"{ids}\",\"logprobs\":\"{lps}\"}}}}]}}"
        )
    }

    /// Inserts `extra` as the last member of the choice object.
    fn fix_tail(body: String, extra: &str) -> String {
        let cut = body.len() - "}]}".len();
        assert_eq!(&body[cut..], "}]}");
        format!("{}{extra}}}]}}", &body[..cut])
    }

    fn run(c: &Cfg, body: &str, validated: bool) -> R<()> {
        let mut st = Stats::default();
        let mem = Mem::unlimited();
        check_body(body.as_bytes(), 0, validated, c, &mut st, &mut Scratch::new(), &mem)
    }

    #[test]
    fn parse_memory_is_accounted() {
        let c = cfg(600, false);
        let body = openai_body(&c, "");
        let mut st = Stats::default();
        let big = Mem::new(1 << 30, None);
        assert_eq!(check_body(body.as_bytes(), 0, true, &c, &mut st, &mut Scratch::new(), &big), Ok(()));
        let peak = big.peak();
        assert!(peak > 0);
        let tight = Mem::new(peak / 2, None);
        let e = check_body(body.as_bytes(), 0, true, &c, &mut st, &mut Scratch::new(), &tight).unwrap_err();
        assert!(e.contains("budget"), "{e}");
        // compact: scratch growth is charged too
        let cc = cfg(600, true);
        let cb = compact_body(&cc, "");
        let m = Mem::new(1 << 30, None);
        assert_eq!(check_body(cb.as_bytes(), 0, false, &cc, &mut st, &mut Scratch::new(), &m), Ok(()));
        assert!(m.peak() as usize >= 600 * 129 * 4);
        let tiny = Mem::new(1000, None);
        assert!(check_body(cb.as_bytes(), 0, false, &cc, &mut st, &mut Scratch::new(), &tiny).is_err());
    }

    #[test]
    fn openai_ok_and_duplicates() {
        let c = cfg(600, false); // n > pool (256): periodic rows exercised
        for v in [true, false] {
            assert_eq!(run(&c, &openai_body(&c, ""), v), Ok(()));
            // duplicate checked keys are rejected whatever their order (Python: last wins)
            for extra in [",\"logprobs\":null", ",\"logprobs\":{}", ",\"finish_reason\":\"abort\"", ",\"token_ids\":[]"] {
                assert!(run(&c, &openai_body(&c, extra), v).is_err(), "{extra}");
            }
        }
        let dup_top = openai_body(&c, "").replacen("{\"request_id\":\"mock-0000\",", "{\"request_id\":\"mock-0000\",\"request_id\":\"mock-0000\",", 1);
        assert!(run(&c, &dup_top, false).is_err());
        let dup_lp = openai_body(&c, "").replacen("\"logprobs\":{\"content\"", "\"logprobs\":{\"x\":1,\"x\":2,\"content\"", 1);
        assert!(run(&c, &dup_lp, true).is_err());
        // NaN extras: json NaN is a singleton in Python, so identical rows stay equal
        let mut b = openai_body(&c, "");
        b = b.replace("\"bytes\":null,", "\"bytes\":null,\"x\":NaN,");
        assert_eq!(run(&c, &b, true), Ok(()));
    }

    #[test]
    fn compact_ok_and_duplicates() {
        let c = cfg(600, true);
        for v in [true, false] {
            assert_eq!(run(&c, &compact_body(&c, ""), v), Ok(()));
            assert!(run(&c, &compact_body(&c, ",\"byteorder\":\"little\""), v).is_err());
            assert!(run(&c, &compact_body(&c, ",\"ranks\":\"AAAA\""), v).is_err());
        }
    }
}
