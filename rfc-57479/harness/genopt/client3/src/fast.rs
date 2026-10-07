//! Hot-loop JSON grammar validator (same language as `json::P::skip_value`:
//! RFC 8259 + Python's NaN/Infinity/-Infinity, strict strings, UTF-8 inside
//! strings). Local index, no Result plumbing per token; returns the end index
//! or Err(position). On Err the caller re-runs the reference skipper to get a
//! message (and the reference result must agree; see tests::agree_fuzz).

use crate::json::{MAX_DEPTH as MAX_DEPTH_U32, MAX_INT_DIGITS};
const MAX_DEPTH: usize = MAX_DEPTH_U32 as usize;

#[inline(always)]
fn at(b: &[u8], i: usize) -> u8 {
    // 0 is never valid where a token byte is required, so it doubles as EOF.
    if i < b.len() {
        unsafe { *b.get_unchecked(i) }
    } else {
        0
    }
}

#[inline(always)]
fn is_ws(c: u8) -> bool {
    c == b' ' || c == b'\n' || c == b'\r' || c == b'\t'
}

#[inline(always)]
fn skip_ws(b: &[u8], mut i: usize) -> usize {
    while is_ws(at(b, i)) {
        i += 1;
    }
    i
}

const L: u64 = 0x0101_0101_0101_0101;
const H: u64 = 0x8080_8080_8080_8080;

#[inline(always)]
fn load8(b: &[u8], i: usize) -> u64 {
    debug_assert!(i + 8 <= b.len());
    u64::from_le_bytes(unsafe { *(b.as_ptr().add(i) as *const [u8; 8]) })
}

/// High bit set in each byte that is '"', '\\', < 0x20 or >= 0x80. The
/// lowest set bit is exact (borrows only propagate above a hit), which is
/// all we use (trailing_zeros on a little-endian load = first such byte).
#[inline(always)]
fn special_mask(x: u64) -> u64 {
    let q = x ^ (L * b'"' as u64);
    let s = x ^ (L * b'\\' as u64);
    let zq = q.wrapping_sub(L) & !q;
    let zs = s.wrapping_sub(L) & !s;
    let lt = x.wrapping_sub(L * 0x20) & !x;
    (zq | zs | lt | x) & H
}

/// High bit set in each byte that is not an ASCII digit (exact per byte).
#[inline(always)]
fn non_digit_mask(x: u64) -> u64 {
    let a = x ^ (L * 0x30);
    let nd = (a & 0xF0F0_F0F0_F0F0_F0F0) | ((a & 0x0F0F_0F0F_0F0F_0F0F).wrapping_add(L * 0x06) & (L * 0x10));
    (nd | nd << 1 | nd << 2 | nd << 3) & H
}

#[inline(always)]
fn hex(c: u8) -> bool {
    c.is_ascii_hexdigit()
}

/// At the opening quote; returns the index after the closing quote.
#[inline(always)]
pub fn string(b: &[u8], i: usize) -> Result<usize, usize> {
    let start = i + 1;
    let mut i = start;
    let mut non_ascii = false;
    loop {
        while i + 8 <= b.len() {
            let m = special_mask(load8(b, i));
            if m != 0 {
                i += (m.trailing_zeros() / 8) as usize;
                break;
            }
            i += 8;
        }
        let c = at(b, i);
        if c == b'"' {
            if non_ascii && std::str::from_utf8(&b[start..i]).is_err() {
                return Err(start);
            }
            return Ok(i + 1);
        } else if c >= 0x80 {
            non_ascii = true;
            i += 1;
        } else if c == b'\\' {
            match at(b, i + 1) {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => i += 2,
                b'u' => {
                    if hex(at(b, i + 2)) && hex(at(b, i + 3)) && hex(at(b, i + 4)) && hex(at(b, i + 5)) {
                        i += 6;
                    } else {
                        return Err(i);
                    }
                }
                _ => return Err(i),
            }
        } else if c < 0x20 {
            return Err(i); // control char or EOF (0)
        } else {
            i += 1;
        }
    }
}

#[inline(always)]
fn digits(b: &[u8], mut i: usize) -> usize {
    // short runs: scalar (well predicted); long runs (float fractions): SWAR
    if !at(b, i).is_ascii_digit() {
        return i;
    }
    i += 1;
    if !at(b, i).is_ascii_digit() {
        return i;
    }
    i += 1;
    while i + 8 <= b.len() {
        let m = non_digit_mask(load8(b, i));
        if m != 0 {
            return i + (m.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while at(b, i).is_ascii_digit() {
        i += 1;
    }
    i
}

/// Inside an array at a value position: consumes a run of plain integers
/// separated by ','. Returns (index, true) if the array's ']' was consumed,
/// else (index of a value the general machine must parse, false). Any
/// non-trivial case (sign, fraction, exponent, whitespace, leading zero
/// followed by a digit, other value types) is handed back unconsumed.
#[inline(always)]
fn int_run(b: &[u8], mut i: usize) -> (usize, bool) {
    loop {
        let s = i;
        let c = at(b, i);
        if c == b'0' {
            i += 1;
        } else if c.wrapping_sub(b'1') < 9 {
            i += 1;
            while at(b, i).is_ascii_digit() {
                i += 1;
            }
        } else {
            return (s, false);
        }
        if i - s > MAX_INT_DIGITS {
            return (s, false); // general path reports the int-digit limit
        }
        match at(b, i) {
            b',' => i += 1,
            b']' => return (i + 1, true),
            _ => return (s, false),
        }
    }
}

/// At '-' or a digit; returns the index after the number.
#[inline(always)]
fn number(b: &[u8], mut i: usize) -> Result<usize, usize> {
    let start = i;
    if at(b, i) == b'-' {
        i += 1;
        if b.len() >= i + 8 && &b[i..i + 8] == b"Infinity" {
            return Ok(i + 8);
        }
    }
    match at(b, i) {
        b'0' => i += 1,
        b'1'..=b'9' => i = digits(b, i + 1),
        _ => return Err(i),
    }
    let int_digits = i - start - (b[start] == b'-') as usize;
    let mut float = false;
    if at(b, i) == b'.' && at(b, i + 1).is_ascii_digit() {
        i = digits(b, i + 2);
        float = true;
    }
    let c = at(b, i);
    if c == b'e' || c == b'E' {
        let mut j = i + 1;
        let s = at(b, j);
        if s == b'+' || s == b'-' {
            j += 1;
        }
        if at(b, j).is_ascii_digit() {
            i = digits(b, j + 1);
            float = true;
        }
    }
    if !float && int_digits > MAX_INT_DIGITS {
        return Err(start);
    }
    Ok(i)
}

#[inline(always)]
fn lit(b: &[u8], i: usize, w: &[u8]) -> Result<usize, usize> {
    if b.len() >= i + w.len() && &b[i..i + w.len()] == w {
        Ok(i + w.len())
    } else {
        Err(i)
    }
}

/// Object member key + ':' starting at i (whitespace allowed before each).
#[inline(always)]
fn key(b: &[u8], i: usize) -> Result<usize, usize> {
    let i = skip_ws(b, i);
    if at(b, i) != b'"' {
        return Err(i);
    }
    let i = skip_ws(b, string(b, i)?);
    if at(b, i) != b':' {
        return Err(i);
    }
    Ok(i + 1)
}

/// Validates one value starting at i (leading whitespace allowed); returns
/// the index just after it.
/// `base` = number of containers already open around this value (the whole
/// document's depth counts against MAX_DEPTH, as in Python).
pub fn skip(b: &[u8], mut i: usize, base: usize, stack: &mut Vec<bool>) -> Result<usize, usize> {
    stack.clear();
    'value: loop {
        i = skip_ws(b, i);
        let c = at(b, i);
        match c {
            b'0'..=b'9' | b'-' => i = number(b, i)?,
            b'"' => i = string(b, i)?,
            b'{' => {
                if base + stack.len() + 1 > MAX_DEPTH {
                    return Err(i);
                }
                let j = skip_ws(b, i + 1);
                if at(b, j) == b'}' {
                    i = j + 1;
                } else {
                    stack.push(true);
                    i = key(b, j)?;
                    continue 'value;
                }
            }
            b'[' => {
                if base + stack.len() + 1 > MAX_DEPTH {
                    return Err(i);
                }
                let j = skip_ws(b, i + 1);
                if at(b, j) == b']' {
                    i = j + 1;
                } else {
                    // Fast path for arrays of plain integers ("bytes": [116,111,...]).
                    let (k, closed) = int_run(b, j);
                    if closed {
                        i = k;
                    } else {
                        stack.push(false);
                        i = k;
                        continue 'value;
                    }
                }
            }
            b't' => i = lit(b, i, b"true")?,
            b'f' => i = lit(b, i, b"false")?,
            b'n' => i = lit(b, i, b"null")?,
            b'N' => i = lit(b, i, b"NaN")?,
            b'I' => i = lit(b, i, b"Infinity")?,
            _ => return Err(i),
        }
        // a value ended at i
        loop {
            let Some(&is_obj) = stack.last() else {
                return Ok(i);
            };
            let mut c = at(b, i);
            if is_ws(c) {
                i = skip_ws(b, i);
                c = at(b, i);
            }
            if c == b',' {
                i = if is_obj { key(b, i + 1)? } else { i + 1 };
                continue 'value;
            } else if (c == b'}' && is_obj) || (c == b']' && !is_obj) {
                i += 1;
                stack.pop();
            } else {
                return Err(i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::P;

    fn reference(b: &[u8]) -> Option<usize> {
        let mut p = P::new(b);
        p.skip_value_reference().ok().map(|_| p.i)
    }

    #[test]
    fn masks_exact_first_byte() {
        // every byte value at every lane, with arbitrary filler before/after
        for fill in [b'a', b'5', 0u8, 0xff, b'"'] {
            for c in 0..=255u8 {
                for lane in 0..8 {
                    let mut w = [b'x'; 8];
                    for k in lane + 1..8 {
                        w[k] = fill;
                    }
                    w[lane] = c;
                    let x = u64::from_le_bytes(w);
                    let sp = c == b'"' || c == b'\\' || c < 0x20 || c >= 0x80;
                    let m = special_mask(x);
                    assert_eq!(m != 0 && (m.trailing_zeros() / 8) as usize == lane, sp, "special c={c} lane={lane}");
                    let mut d = [b'7'; 8];
                    for k in lane + 1..8 {
                        d[k] = fill;
                    }
                    d[lane] = c;
                    let m = non_digit_mask(u64::from_le_bytes(d));
                    let nd = !c.is_ascii_digit();
                    assert_eq!(m != 0 && (m.trailing_zeros() / 8) as usize == lane, nd, "digit c={c} lane={lane}");
                }
            }
        }
    }

    #[test]
    fn python_limits() {
        let mut st = Vec::new();
        let deep = |d: usize, o: &str, c: &str| format!("{}1{}", o.repeat(d), c.repeat(d)).into_bytes();
        for (o, c) in [("[", "]"), ("{\"a\":", "}")] {
            let ok = deep(MAX_DEPTH, o, c);
            let bad = deep(MAX_DEPTH + 1, o, c);
            assert!(skip(&ok, 0, 0, &mut st).is_ok());
            assert!(reference(&ok).is_some());
            assert!(skip(&bad, 0, 0, &mut st).is_err());
            assert!(reference(&bad).is_none());
            // base depth counts too
            assert!(skip(&ok, 0, 1, &mut st).is_err());
        }
        let empty_bad = format!("{}{}", "[".repeat(MAX_DEPTH + 1), "]".repeat(MAX_DEPTH + 1)).into_bytes();
        assert!(skip(&empty_bad, 0, 0, &mut st).is_err() && reference(&empty_bad).is_none());
        for (txt, ok) in [
            ("1".repeat(4300), true),
            ("1".repeat(4301), false),
            (format!("-{}", "1".repeat(4300)), true),
            (format!("-{}", "1".repeat(4301)), false),
            (format!("{}.0", "1".repeat(5000)), true),
            (format!("{}e1", "1".repeat(5000)), true),
            (format!("[{},1]", "1".repeat(4301)), false),
            (format!("[1,{}]", "1".repeat(4300)), true),
        ] {
            let b = txt.as_bytes();
            assert_eq!(skip(b, 0, 0, &mut st).is_ok(), ok, "{}", &txt[..20]);
            assert_eq!(reference(b).is_some(), ok);
        }
    }

    #[test]
    fn agree_fuzz() {
        let seeds: &[&[u8]] = &[
            r#"{"request_id":"mock-0001","choices":[{"index":0,"logprobs":{"content":[{"token":"token_id:5","logprob":-4.85,"bytes":[116,111],"top_logprobs":[{"token":"té\n","logprob":-1.5e-3,"bytes":[]}]}]},"x":[true,false,null,NaN,-Infinity,Infinity,0,-0.0,1E+5,[],{}]}],"s":"héllo 😀 w\"orld"}"#.as_bytes(),
            "{\"a\" : [ 1 , 2 ,\n\t3 ] , \"b\" : { \"c\" : \"\u{e9}\u{1F600}\" } }".as_bytes(),
            b"[[[[[]]]],{},\"\",-1,0.5e-10]",
            b"{\"bytes\":[116,111,107,0,10],\"x\":[[1,2],[3],[0]],\"y\":[0,1.5,2e3,-4,7],\"z\":[10, 20 ,30]}",
            b"[12345678901234567890,-0.12345678901234e-123456789,\"a long string of more than sixteen chars \\n with \\u00e9scapes and more text\",1234567,12345678,123456789]",
        ];
        let mut rng: u64 = 0x1234_5678_9abc_def0;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let alphabet = b"{}[]\",:.-+eE0123456789 \n\\/utfnlasrNIa\x01\xc3\xa9\xff";
        let mut stack = Vec::new();
        let mut checked = 0;
        for seed in seeds {
            assert_eq!(skip(seed, 0, 0, &mut stack).ok(), reference(seed), "seed");
            for _ in 0..40_000 {
                let mut d = seed.to_vec();
                for _ in 0..(1 + next() % 3) {
                    let pos = (next() as usize) % d.len();
                    match next() % 3 {
                        0 => d[pos] = alphabet[(next() as usize) % alphabet.len()],
                        1 => {
                            d.remove(pos);
                        }
                        _ => d.insert(pos, alphabet[(next() as usize) % alphabet.len()]),
                    }
                    if d.is_empty() {
                        d.push(b'1');
                    }
                }
                let f = skip(&d, 0, 0, &mut stack).ok();
                let r = reference(&d);
                assert_eq!(f, r, "disagree on {:?}", String::from_utf8_lossy(&d));
                checked += 1;
            }
        }
        assert!(checked > 100_000);
    }
}
