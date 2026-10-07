//! Minimal, strict, zero-copy JSON scanner with Python `json.loads` acceptance
//! semantics (RFC 8259 grammar plus Python's NaN / Infinity / -Infinity
//! literals; strict mode: raw control characters inside strings are errors).
//!
//! Two ways to consume a value:
//!   * `skip_value` fully validates the grammar of a value without building it;
//!   * `value` builds a small generic tree (`V`) for the parts that are
//!     compared semantically (Python equality semantics live in `py_eq`).
//! UTF-8 validity of the whole document is checked by the caller once.

pub type R<T> = Result<T, String>;

/// Maximum container nesting (whole document). CPython 3.12.13's json
/// scanner (the v1/v2 interpreter) accepts at most 9997 nested containers
/// (measured, same at 100 Python frames and in threads); 9990 keeps v3
/// equal-or-stricter with a margin for call-context differences.
pub const MAX_DEPTH: u32 = 9_990;
/// CPython >= 3.11 int max str digits (sys.int_info.default_max_str_digits):
/// json.loads raises ValueError for an integer literal with more digits.
pub const MAX_INT_DIGITS: usize = 4_300;

#[derive(Debug, Clone)]
pub enum V {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    /// Decoded string as WTF-8 bytes (lone surrogates kept), so byte equality
    /// == Python code-point equality.
    Str(Vec<u8>),
    Arr(Vec<V>),
    /// Insertion order kept; lookups are last-wins like Python dicts.
    Obj(Vec<(Vec<u8>, V)>),
}

impl V {
    pub fn get(&self, key: &[u8]) -> Option<&V> {
        match self {
            V::Obj(items) => items.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Python's float(x) for numbers as `math.isclose` would accept them.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            V::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
            V::Int(i) => Some(*i as f64),
            V::Float(f) => Some(*f),
            _ => None,
        }
    }
}

enum Num {
    I(i128),
    F(f64),
}

fn num_of(v: &V) -> Option<Num> {
    match v {
        V::Bool(b) => Some(Num::I(*b as i128)),
        V::Int(i) => Some(Num::I(*i)),
        V::Float(f) => Some(Num::F(*f)),
        _ => None,
    }
}

fn int_eq_float(i: i128, f: f64) -> bool {
    if !f.is_finite() || f.fract() != 0.0 || f.abs() >= 1.7e38 {
        return false;
    }
    (f as i128) == i
}

/// Python `==` for values produced by json.loads. json's NaN literal is a
/// shared singleton (json.decoder NaN constant), and dict/list comparison
/// short-cuts on identity, so a NaN compares equal to another JSON NaN inside
/// containers (the only way v1 compares rows). Bare-float isclose() checks
/// are separate and treat NaN as never close.
pub fn py_eq(a: &V, b: &V) -> bool {
    if let (Some(x), Some(y)) = (num_of(a), num_of(b)) {
        return match (x, y) {
            (Num::I(p), Num::I(q)) => p == q,
            (Num::F(p), Num::F(q)) => p == q || (p.is_nan() && q.is_nan()),
            (Num::I(p), Num::F(q)) | (Num::F(q), Num::I(p)) => int_eq_float(p, q),
        };
    }
    match (a, b) {
        (V::Null, V::Null) => true,
        (V::Str(x), V::Str(y)) => x == y,
        (V::Arr(x), V::Arr(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| py_eq(p, q)),
        (V::Obj(_), V::Obj(_)) => {
            let da = dedup(a);
            let db = dedup(b);
            da.len() == db.len()
                && da.iter().all(|(k, v)| db.get(k).map_or(false, |w| py_eq(v, w)))
        }
        _ => false,
    }
}

fn dedup(v: &V) -> std::collections::HashMap<&[u8], &V> {
    let mut m = std::collections::HashMap::new();
    if let V::Obj(items) = v {
        for (k, x) in items {
            m.insert(k.as_slice(), x); // later keys overwrite: last wins
        }
    }
    m
}

#[inline]
fn has_special(x: u64) -> bool {
    const L: u64 = 0x0101_0101_0101_0101;
    const H: u64 = 0x8080_8080_8080_8080;
    #[inline]
    fn zero(v: u64) -> u64 {
        v.wrapping_sub(L) & !v & H
    }
    let q = x ^ (L * b'"' as u64);
    let s = x ^ (L * b'\\' as u64);
    let lt = x.wrapping_sub(L * 0x20) & !x & H; // some byte < 0x20
    (zero(q) | zero(s) | lt | (x & H)) != 0
}

#[inline]
fn hexval(c: u8) -> Option<u32> {
    match c {
        b'0'..=b'9' => Some((c - b'0') as u32),
        b'a'..=b'f' => Some((c - b'a' + 10) as u32),
        b'A'..=b'F' => Some((c - b'A' + 10) as u32),
        _ => None,
    }
}

/// Per-request parse-memory accounting. Every parse-time allocation that can
/// grow with the input (value trees, decoded strings, base64 scratch, key
/// sets) is charged here *before* it is made. `remaining` starts at the
/// request's scratch reservation; when it runs out, `grow` may take more
/// from the global byte budget without blocking; otherwise the request fails
/// with an error (never an abort, never a deadlock).
pub struct Mem<'a> {
    remaining: std::cell::Cell<u64>,
    grow: Option<&'a dyn Fn(u64) -> bool>,
    reserved: std::cell::Cell<u64>,
    peak: std::cell::Cell<u64>,
}

impl<'a> Mem<'a> {
    pub fn new(reserved: u64, grow: Option<&'a dyn Fn(u64) -> bool>) -> Self {
        Mem {
            remaining: std::cell::Cell::new(reserved),
            grow,
            reserved: std::cell::Cell::new(reserved),
            peak: std::cell::Cell::new(0),
        }
    }
    pub fn unlimited() -> Mem<'static> {
        Mem::new(u64::MAX / 4, None)
    }
    pub fn charge(&self, n: u64) -> R<()> {
        let r = self.remaining.get();
        if r >= n {
            self.remaining.set(r - n);
        } else {
            let need = n - r;
            match self.grow {
                Some(g) if g(need) => {
                    self.reserved.set(self.reserved.get() + need);
                    self.remaining.set(0);
                }
                _ => {
                    return Err(format!(
                        "parse memory budget exhausted: need {n} more bytes, {r} left of the request reservation and the global budget has no room"
                    ))
                }
            }
        }
        let used = self.reserved.get() - self.remaining.get();
        if used > self.peak.get() {
            self.peak.set(used);
        }
        Ok(())
    }
    pub fn refund(&self, n: u64) {
        self.remaining.set(self.remaining.get() + n);
    }
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn remaining(&self) -> u64 {
        self.remaining.get()
    }
    /// Bytes currently charged.
    pub fn used(&self) -> u64 {
        self.reserved.get() - self.remaining.get()
    }
    /// Peak bytes charged (parse scratch high-water mark).
    pub fn peak(&self) -> u64 {
        self.peak.get()
    }
}

/// Keys longer than this are rejected (no real key comes close; bounds key copies).
pub const MAX_KEY_BYTES: usize = 64 << 10;
const NODE: u64 = 2 * std::mem::size_of::<V>() as u64;
const MEMBER: u64 = 2 * std::mem::size_of::<(Vec<u8>, V)>() as u64;

pub struct P<'a> {
    pub b: &'a [u8],
    pub i: usize,
    depth: u32,
    stk: Vec<bool>,
    pub mem: &'a Mem<'a>,
}

thread_local! {
    // one leaked unlimited accountant per thread, for P::new (tests / tools)
    static UNLIMITED: &'static Mem<'static> = Box::leak(Box::new(Mem::unlimited()));
}

impl<'a> P<'a> {
    /// Parser with no memory accounting (tests / offline tools).
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn new(b: &'a [u8]) -> Self {
        let m: &'static Mem<'static> = UNLIMITED.with(|m| *m);
        P { b, i: 0, depth: 0, stk: Vec::new(), mem: m }
    }

    /// Parser whose value trees / decoded strings are charged to `mem`.
    pub fn with_mem(b: &'a [u8], mem: &'a Mem<'a>) -> Self {
        P { b, i: 0, depth: 0, stk: Vec::new(), mem }
    }

    pub fn err<T>(&self, msg: &str) -> R<T> {
        Err(format!("JSON error at byte {}: {}", self.i, msg))
    }

    #[inline]
    pub fn ws(&mut self) {
        while let Some(&c) = self.b.get(self.i) {
            if c == b' ' || c == b'\n' || c == b'\r' || c == b'\t' {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    #[inline]
    pub fn peek(&self) -> Option<u8> {
        self.b.get(self.i).copied()
    }

    #[inline]
    pub fn expect(&mut self, c: u8) -> R<()> {
        if self.peek() == Some(c) {
            self.i += 1;
            Ok(())
        } else {
            self.err(&format!("expected '{}'", c as char))
        }
    }

    fn lit(&mut self, word: &[u8]) -> R<()> {
        if self.b[self.i..].starts_with(word) {
            self.i += word.len();
            Ok(())
        } else {
            self.err("invalid literal")
        }
    }

    fn enter(&mut self) -> R<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return self.err("nesting too deep");
        }
        Ok(())
    }

    /// At an opening quote. Validates the string and returns the raw content
    /// span (without quotes) and whether it contains escapes.
    #[inline]
    pub fn scan_string(&mut self) -> R<(usize, usize, bool)> {
        self.expect(b'"')?;
        let b = self.b;
        let start = self.i;
        let mut i = start;
        let mut esc = false;
        let mut non_ascii = false;
        loop {
            while i + 8 <= b.len() {
                let x = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
                if has_special(x) {
                    break;
                }
                i += 8;
            }
            let Some(&c) = b.get(i) else {
                self.i = i;
                return self.err("unterminated string");
            };
            if c == b'"' {
                if non_ascii && std::str::from_utf8(&b[start..i]).is_err() {
                    self.i = start;
                    return self.err("invalid UTF-8 in string");
                }
                self.i = i + 1;
                return Ok((start, i, esc));
            } else if c >= 0x80 {
                non_ascii = true;
                i += 1;
            } else if c == b'\\' {
                esc = true;
                match b.get(i + 1) {
                    Some(b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => i += 2,
                    Some(b'u') => {
                        if i + 6 > b.len() || !b[i + 2..i + 6].iter().all(|&h| hexval(h).is_some()) {
                            self.i = i;
                            return self.err("invalid \\u escape");
                        }
                        i += 6;
                    }
                    _ => {
                        self.i = i;
                        return self.err("invalid escape");
                    }
                }
            } else if c < 0x20 {
                self.i = i;
                return self.err("invalid control character in string");
            } else {
                i += 1;
            }
        }
    }

    /// Decodes a validated raw string span to WTF-8 bytes.
    /// Decodes a validated raw string span to WTF-8 bytes. The caller has
    /// charged (or bounded) `e - s` bytes; the allocation itself is fallible.
    pub fn decode(&self, s: usize, e: usize, esc: bool) -> R<Vec<u8>> {
        let raw = &self.b[s..e];
        let mut out = Vec::new();
        out.try_reserve_exact(raw.len()).map_err(|er| format!("string allocation of {} bytes failed: {er}", raw.len()))?;
        if !esc {
            out.extend_from_slice(raw);
            return Ok(out);
        }
        let mut i = 0;
        let hex4 = |j: usize| -> u32 { raw[j..j + 4].iter().fold(0, |a, &h| a * 16 + hexval(h).unwrap()) };
        while i < raw.len() {
            let c = raw[i];
            if c != b'\\' {
                out.push(c);
                i += 1;
                continue;
            }
            let d = raw[i + 1];
            i += 2;
            let ch = match d {
                b'"' => '"' as u32,
                b'\\' => '\\' as u32,
                b'/' => '/' as u32,
                b'b' => 8,
                b'f' => 12,
                b'n' => 10,
                b'r' => 13,
                b't' => 9,
                _ => {
                    let mut u = hex4(i);
                    i += 4;
                    // Python combines a high surrogate followed by \uDC00-\uDFFF.
                    if (0xD800..0xDC00).contains(&u) && raw.len() >= i + 6 && raw[i] == b'\\' && raw[i + 1] == b'u' {
                        let lo = hex4(i + 2);
                        if (0xDC00..0xE000).contains(&lo) {
                            u = 0x10000 + ((u - 0xD800) << 10) + (lo - 0xDC00);
                            i += 6;
                        }
                    }
                    u
                }
            };
            push_wtf8(&mut out, ch);
        }
        Ok(out)
    }

    /// Validates a number (or NaN / Infinity / -Infinity); returns span and is_float.
    #[inline]
    pub fn scan_number(&mut self) -> R<(usize, usize, bool)> {
        let b = self.b;
        let s = self.i;
        let mut i = s;
        if b.get(i) == Some(&b'-') {
            i += 1;
            if b[i..].starts_with(b"Infinity") {
                self.i = i + 8;
                return Ok((s, self.i, true));
            }
        }
        match b.get(i) {
            Some(b'0') => i += 1,
            Some(b'1'..=b'9') => {
                i += 1;
                while matches!(b.get(i), Some(b'0'..=b'9')) {
                    i += 1;
                }
            }
            _ => {
                self.i = i;
                return self.err("invalid number");
            }
        }
        let int_end = i;
        let mut float = false;
        if b.get(i) == Some(&b'.') && matches!(b.get(i + 1), Some(b'0'..=b'9')) {
            i += 2;
            while matches!(b.get(i), Some(b'0'..=b'9')) {
                i += 1;
            }
            float = true;
        }
        if matches!(b.get(i), Some(b'e' | b'E')) {
            let mut j = i + 1;
            if matches!(b.get(j), Some(b'+' | b'-')) {
                j += 1;
            }
            if matches!(b.get(j), Some(b'0'..=b'9')) {
                while matches!(b.get(j), Some(b'0'..=b'9')) {
                    j += 1;
                }
                i = j;
                float = true;
            }
        }
        // A '.' or 'e' not followed by digits is left unconsumed, as Python's
        // scanner does; the caller then fails on the unexpected character.
        let digits = int_end - s - (b[s] == b'-') as usize;
        if !float && digits > MAX_INT_DIGITS {
            self.i = s;
            return self.err("integer literal exceeds 4300 digits (Python int limit)");
        }
        self.i = i;
        Ok((s, i, float))
    }

    /// Fully validates one value without building it (iterative, explicit
    /// container stack: this is the hot loop for unvalidated openai bodies).
    pub fn skip_value(&mut self) -> R<()> {
        let mut stack = std::mem::take(&mut self.stk);
        let r = crate::fast::skip(self.b, self.i, self.depth as usize, &mut stack);
        self.stk = stack;
        match r {
            Ok(end) => {
                self.i = end;
                Ok(())
            }
            Err(_) => {
                // Reference validator for the error message. Both must reject.
                let start = self.i;
                match self.skip_value_reference() {
                    Err(e) => Err(e),
                    Ok(()) => {
                        self.i = start;
                        self.err("fast validator rejected a value the reference accepts (bug)")
                    }
                }
            }
        }
    }

    /// Reference (straightforward) validator; same language as fast::skip.
    pub fn skip_value_reference(&mut self) -> R<()> {
        // stack of open containers: true = object, false = array
        let mut stack = std::mem::take(&mut self.stk);
        stack.clear();
        let r = self.skip_value_inner(&mut stack);
        self.stk = stack;
        r
    }

    fn skip_value_inner(&mut self, stack: &mut Vec<bool>) -> R<()> {
        loop {
            // ---- a value is expected at self.i ----
            self.ws();
            match self.peek() {
                Some(b'"') => {
                    self.scan_string()?;
                }
                Some(b'-' | b'0'..=b'9') => {
                    self.scan_number()?;
                }
                Some(b'{') => {
                    if self.depth + stack.len() as u32 + 1 > MAX_DEPTH {
                        return self.err("nesting too deep");
                    }
                    self.i += 1;
                    self.ws();
                    if self.peek() == Some(b'}') {
                        self.i += 1;
                    } else {
                        stack.push(true);
                        self.scan_string()?;
                        self.ws();
                        self.expect(b':')?;
                        continue;
                    }
                }
                Some(b'[') => {
                    if self.depth + stack.len() as u32 + 1 > MAX_DEPTH {
                        return self.err("nesting too deep");
                    }
                    self.i += 1;
                    self.ws();
                    if self.peek() == Some(b']') {
                        self.i += 1;
                    } else {
                        stack.push(false);
                        continue;
                    }
                }
                Some(b't') => self.lit(b"true")?,
                Some(b'f') => self.lit(b"false")?,
                Some(b'n') => self.lit(b"null")?,
                Some(b'N') => self.lit(b"NaN")?,
                Some(b'I') => self.lit(b"Infinity")?,
                _ => return self.err("expected a value"),
            }
            // ---- a value just ended: close containers / move to next member ----
            loop {
                let Some(&is_obj) = stack.last() else {
                    return Ok(());
                };
                self.ws();
                match self.peek() {
                    Some(b',') => {
                        self.i += 1;
                        if is_obj {
                            self.ws();
                            self.scan_string()?;
                            self.ws();
                            self.expect(b':')?;
                        }
                        break;
                    }
                    Some(b'}') if is_obj => {
                        self.i += 1;
                        stack.pop();
                    }
                    Some(b']') if !is_obj => {
                        self.i += 1;
                        stack.pop();
                    }
                    _ => return self.err(if is_obj { "expected ',' or '}'" } else { "expected ',' or ']'" }),
                }
            }
        }
    }

    /// At '['; validates every element and returns the element count.
    pub fn skip_array_count(&mut self) -> R<usize> {
        self.expect(b'[')?;
        self.enter()?;
        let mut n = 0;
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
        } else {
            loop {
                self.skip_value()?;
                n += 1;
                self.ws();
                match self.peek() {
                    Some(b',') => self.i += 1,
                    Some(b']') => {
                        self.i += 1;
                        break;
                    }
                    _ => return self.err("expected ',' or ']'"),
                }
            }
        }
        self.depth -= 1;
        Ok(n)
    }

    /// Iterates object members: calls `f(self, key)` with the parser
    /// positioned at the member value; `f` must consume exactly that value.
    pub fn object<F: FnMut(&mut Self, &[u8]) -> R<()>>(&mut self, mut f: F) -> R<()> {
        self.ws();
        self.expect(b'{')?;
        self.enter()?;
        self.ws();
        if self.peek() == Some(b'}') {
            self.i += 1;
        } else {
            loop {
                self.ws();
                let (s, e, esc) = self.scan_string()?;
                if e - s > MAX_KEY_BYTES {
                    self.i = s;
                    return self.err("object key longer than 64 KiB");
                }
                let key = self.decode(s, e, esc)?;
                self.ws();
                self.expect(b':')?;
                self.ws();
                f(self, &key)?;
                self.ws();
                match self.peek() {
                    Some(b',') => self.i += 1,
                    Some(b'}') => {
                        self.i += 1;
                        break;
                    }
                    _ => return self.err("expected ',' or '}'"),
                }
            }
        }
        self.depth -= 1;
        Ok(())
    }

    /// Like `object`, but a repeated key is an error. Used for every object
    /// whose members v3 checks (top level, choice, logprobs, compact block):
    /// Python keeps only the last duplicate, so rejecting duplicates outright
    /// is equal-or-stricter and removes any first-vs-last ambiguity.
    pub fn object_unique<F: FnMut(&mut Self, &[u8]) -> R<()>>(&mut self, mut f: F) -> R<()> {
        let mut seen: std::collections::HashSet<Vec<u8>> = std::collections::HashSet::new();
        let mut charged = 0u64;
        let r = self.object(|p, k| {
            if seen.contains(k) {
                return p.err(&format!("duplicate key {:?} in a checked object", String::from_utf8_lossy(k)));
            }
            let c = k.len() as u64 + 64;
            p.mem.charge(c)?;
            charged += c;
            seen.insert(k.to_vec());
            f(p, k)
        });
        self.mem.refund(charged);
        r
    }

    /// Iterates array elements: calls `f(self, index)` positioned at the
    /// element (whitespace skipped); returns the count.
    pub fn array<F: FnMut(&mut Self, usize) -> R<()>>(&mut self, mut f: F) -> R<usize> {
        self.ws();
        self.expect(b'[')?;
        self.enter()?;
        let mut n = 0;
        self.ws();
        if self.peek() == Some(b']') {
            self.i += 1;
        } else {
            loop {
                self.ws();
                f(self, n)?;
                n += 1;
                self.ws();
                match self.peek() {
                    Some(b',') => self.i += 1,
                    Some(b']') => {
                        self.i += 1;
                        break;
                    }
                    _ => return self.err("expected ',' or ']'"),
                }
            }
        }
        self.depth -= 1;
        Ok(n)
    }

    pub fn number_value(&mut self) -> R<V> {
        let (s, e, float) = self.scan_number()?;
        let text = std::str::from_utf8(&self.b[s..e]).unwrap();
        if float {
            let f = match text {
                "Infinity" => f64::INFINITY,
                "-Infinity" => f64::NEG_INFINITY,
                _ => text.parse::<f64>().map_err(|e| format!("float parse {text}: {e}"))?,
            };
            Ok(V::Float(f))
        } else {
            match text.parse::<i128>() {
                Ok(i) => Ok(V::Int(i)),
                Err(_) => self.err("integer out of supported range"),
            }
        }
    }

    /// Builds a generic value.
    pub fn value(&mut self) -> R<V> {
        self.ws();
        match self.peek() {
            Some(b'"') => {
                let (s, e, esc) = self.scan_string()?;
                self.mem.charge(NODE + (e - s) as u64)?;
                Ok(V::Str(self.decode(s, e, esc)?))
            }
            Some(b'{') => {
                self.mem.charge(NODE)?;
                let mut items = Vec::new();
                self.object(|p, k| {
                    p.mem.charge(MEMBER + k.len() as u64)?;
                    let v = p.value()?;
                    items.push((k.to_vec(), v));
                    Ok(())
                })?;
                Ok(V::Obj(items))
            }
            Some(b'[') => {
                self.mem.charge(NODE)?;
                let mut items = Vec::new();
                self.array(|p, _| {
                    items.push(p.value()?);
                    Ok(())
                })?;
                Ok(V::Arr(items))
            }
            Some(b't' | b'f' | b'n' | b'N' | b'I' | b'-' | b'0'..=b'9') => {
                self.mem.charge(NODE)?;
                match self.peek() {
                    Some(b't') => self.lit(b"true").map(|_| V::Bool(true)),
                    Some(b'f') => self.lit(b"false").map(|_| V::Bool(false)),
                    Some(b'n') => self.lit(b"null").map(|_| V::Null),
                    Some(b'N') => self.lit(b"NaN").map(|_| V::Float(f64::NAN)),
                    Some(b'I') => self.lit(b"Infinity").map(|_| V::Float(f64::INFINITY)),
                    _ => self.number_value(),
                }
            }
            _ => self.err("expected a value"),
        }
    }
}

fn push_wtf8(out: &mut Vec<u8>, c: u32) {
    if c < 0x80 {
        out.push(c as u8);
    } else if c < 0x800 {
        out.extend_from_slice(&[0xC0 | (c >> 6) as u8, 0x80 | (c & 0x3F) as u8]);
    } else if c < 0x10000 {
        out.extend_from_slice(&[
            0xE0 | (c >> 12) as u8,
            0x80 | ((c >> 6) & 0x3F) as u8,
            0x80 | (c & 0x3F) as u8,
        ]);
    } else {
        out.extend_from_slice(&[
            0xF0 | (c >> 18) as u8,
            0x80 | ((c >> 12) & 0x3F) as u8,
            0x80 | ((c >> 6) & 0x3F) as u8,
            0x80 | (c & 0x3F) as u8,
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(s: &str) -> bool {
        let mut p = P::new(s.as_bytes());
        p.skip_value().is_ok() && {
            p.ws();
            p.i == s.len()
        }
    }

    #[test]
    fn grammar() {
        for good in [
            "{}", "[]", "[1, -0, 0.5, 1e5, 1E+5, -1.5e-3]", "\"a\\u00e9\\n\"", "{\"a\": [true, false, null]}",
            "NaN", "-Infinity", "Infinity", "\"\\ud83d\\ude00\"",
        ] {
            assert!(ok(good), "{good}");
        }
        for bad in [
            "{", "[1,]", "{\"a\":1,}", "01", "1.", "1e", "-", "\"\\x\"", "\"a\tb\"", "tru", "[1 2]", "{1:2}", "+1",
            "\"\\u12\"", ".5", "-NaN",
        ] {
            assert!(!ok(bad), "{bad}");
        }
    }

    #[test]
    fn utf8() {
        let okb = |b: &[u8]| {
            let mut p = P::new(b);
            p.skip_value().is_ok() && p.i == b.len()
        };
        assert!(okb("[\"h\u{e9}llo w\u{f6}rld, long enough for the 8-byte path \u{1F600}\"]".as_bytes()));
        assert!(!okb(b"[\"abcdefghijkl\xff\"]"));
        assert!(!okb(b"[\"\xed\xa0\x80\"]")); // UTF-8-encoded surrogate: Python rejects too
        assert!(!okb(b"[1,\xc3\xa9]"));
        assert!(!okb(b"[\"\xc3\"]"));
    }

    #[test]
    fn memory_budget() {
        let doc = br#"{"a":[1,2,3,{"b":"xxxxxxxxxxxxxxxxxxxx"}],"c":null}"#;
        let m = Mem::new(1 << 20, None);
        let v = P::with_mem(doc, &m).value();
        assert!(v.is_ok());
        let used = (1 << 20) - m.remaining();
        assert!(used > 0 && m.peak() == used);
        let small = Mem::new(used - 1, None);
        assert!(P::with_mem(doc, &small).value().unwrap_err().contains("budget"));
        // growth from the global budget is used when allowed
        let ok = |_n: u64| true;
        let grown = Mem::new(16, Some(&ok));
        assert!(P::with_mem(doc, &grown).value().is_ok());
        let huge_key = format!("{{\"{}\":1}}", "k".repeat(MAX_KEY_BYTES + 1));
        assert!(P::new(huge_key.as_bytes()).value().is_err());
    }

    #[test]
    fn equality() {
        let v = |s: &str| P::new(s.as_bytes()).value().unwrap();
        assert!(py_eq(&v("{\"a\": 1, \"a\": 2}"), &v("{\"a\": 2.0}")));
        assert!(py_eq(&v("[true]"), &v("[1]")));
        assert!(py_eq(&v("NaN"), &v("NaN"))); // json NaN singleton + identity shortcut
        assert!(py_eq(&v("[NaN]"), &v("[NaN]")));
        assert!(!py_eq(&v("\"1\""), &v("1")));
        assert!(py_eq(&v("\"\\u00e9\""), &v("\"\u{e9}\"")));
        assert!(!py_eq(&v("{\"a\": 1}"), &v("{\"a\": 1, \"b\": 1}")));
    }
}
