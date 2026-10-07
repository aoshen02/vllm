//! Strict standard base64 decode: NEON bulk path for 64-char blocks + the
//! `base64` crate (STANDARD: canonical padding, no trailing bits) for the
//! final <=128 chars. Accepts exactly what the crate accepts (tests).

use base64::Engine as _;

/// Decodes `src` into `dst` (needs src.len()/4*3 bytes); returns the length.
pub fn decode(src: &[u8], dst: &mut [u8]) -> Result<usize, String> {
    if src.len() % 4 != 0 {
        return Err("base64 length not a multiple of 4".into());
    }
    // Keep at least one full block (and the padding) for the scalar tail.
    let bulk_blocks = if src.len() > 128 { (src.len() - 65) / 64 } else { 0 };
    let bulk = bulk_blocks * 64;
    #[cfg(target_arch = "aarch64")]
    {
        if bulk > 0 && !unsafe { neon::decode_blocks(&src[..bulk], &mut dst[..bulk / 4 * 3]) } {
            return Err("invalid base64 character".into());
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    let bulk = 0;
    let out = bulk / 4 * 3;
    let n = base64::engine::general_purpose::STANDARD
        .decode_slice(&src[bulk..], &mut dst[out..])
        .map_err(|e| format!("base64: {e}"))?;
    Ok(out + n)
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::*;

    const fn table() -> [u8; 128] {
        let mut t = [0xFFu8; 128];
        let mut i = 0;
        while i < 26 {
            t[b'A' as usize + i] = i as u8;
            t[b'a' as usize + i] = 26 + i as u8;
            i += 1;
        }
        let mut d = 0;
        while d < 10 {
            t[b'0' as usize + d] = 52 + d as u8;
            d += 1;
        }
        t[b'+' as usize] = 62;
        t[b'/' as usize] = 63;
        t
    }
    static TABLE: [u8; 128] = table();

    /// src.len() % 64 == 0; dst.len() == src.len()/4*3. false on any invalid char.
    #[target_feature(enable = "neon")]
    pub unsafe fn decode_blocks(src: &[u8], dst: &mut [u8]) -> bool {
        let lo = vld1q_u8_x4(TABLE.as_ptr());
        let hi = vld1q_u8_x4(TABLE.as_ptr().add(64));
        let k64 = vdupq_n_u8(64);
        let mut err = vdupq_n_u8(0);
        let mut s = src.as_ptr();
        let mut d = dst.as_mut_ptr();
        let end = s.add(src.len());
        while s < end {
            let v = vld4q_u8(s);
            let map = |c: uint8x16_t| -> uint8x16_t { vqtbx4q_u8(vqtbl4q_u8(lo, c), hi, vsubq_u8(c, k64)) };
            let a = map(v.0);
            let b = map(v.1);
            let c = map(v.2);
            let e = map(v.3);
            // invalid table entries are 0xFF; chars >= 128 have the high bit themselves
            err = vorrq_u8(err, vorrq_u8(vorrq_u8(a, b), vorrq_u8(c, e)));
            err = vorrq_u8(err, vorrq_u8(vorrq_u8(v.0, v.1), vorrq_u8(v.2, v.3)));
            let o0 = vorrq_u8(vshlq_n_u8::<2>(a), vshrq_n_u8::<4>(b));
            let o1 = vorrq_u8(vshlq_n_u8::<4>(b), vshrq_n_u8::<2>(c));
            let o2 = vorrq_u8(vshlq_n_u8::<6>(c), e);
            vst3q_u8(d, uint8x16x3_t(o0, o1, o2));
            s = s.add(64);
            d = d.add(48);
        }
        vmaxvq_u8(err) < 0x80
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_crate() {
        let mut x: u64 = 0x2545_F491_4F6C_DD1D;
        let mut rnd = || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let eng = base64::engine::general_purpose::STANDARD;
        for len in (0..2000).chain([4095, 4096, 4097, 100_000]) {
            let raw: Vec<u8> = (0..len).map(|_| rnd() as u8).collect();
            let enc = eng.encode(&raw).into_bytes();
            let mut out = vec![0u8; enc.len() / 4 * 3 + 3];
            let n = decode(&enc, &mut out).unwrap();
            assert_eq!(&out[..n], &raw[..], "len {len}");
            // single corruption anywhere must give the same verdict as the crate
            if !enc.is_empty() {
                for _ in 0..8 {
                    let mut bad = enc.clone();
                    let p = (rnd() as usize) % bad.len();
                    bad[p] = rnd() as u8;
                    let ours = decode(&bad, &mut out).map(|n| out[..n].to_vec()).ok();
                    let theirs = eng.decode(&bad).ok();
                    assert_eq!(ours, theirs, "len {len} pos {p} byte {}", bad[p]);
                }
            }
        }
    }
}
