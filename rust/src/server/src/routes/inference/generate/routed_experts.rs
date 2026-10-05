// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! `choices[0].routed_experts` (R3) for the raw generate route.
//!
//! Python returns `vllm.utils.serial_utils.numpy2base64(routed_experts)`:
//! standard padded base64 of `np.save(array, allow_pickle=False)` where the
//! array is the concatenation (axis 0) of every engine `routed_experts` chunk
//! of the request (the prompt block on the first output, then one row per
//! generated token), or `null` when no chunk arrived.
//!
//! The `.npy` header depends on the final row count, but its *length* does
//! not: numpy pads the first-axis repr to 21 digits of growth space and the
//! whole header to a 64-byte boundary. So rows are base64-encoded as they
//! arrive, phase-shifted by the known header length: the first
//! `(3 - header_len % 3) % 3` data bytes are held back and encoded together
//! with the header at the end, after which the already encoded data
//! continues on a 3-byte boundary. The result is byte-identical to encoding
//! header + data in one go.
//!
//! Like `np.concatenate`, the output is in native (little-endian) byte
//! order: big-endian chunks are written as `<` dtypes with swapped element
//! bytes, even when there is a single chunk.
//!
//! Only the R3 layout `[rows, num_layers, top_k]` is accepted; any other
//! rank fails the request (a server error) instead of building a header.

use std::borrow::Cow;

use vllm_llm::RoutedExperts;

use super::compact::{Base64Segments, EncodedArray, encode_base64_into};

/// numpy `format.GROWTH_AXIS_MAX_DIGITS`.
const GROWTH_AXIS_MAX_DIGITS: usize = 21;
/// numpy `format.ARRAY_ALIGN`.
const ARRAY_ALIGN: usize = 64;
/// `\x93NUMPY` + version (1, 0).
const MAGIC_V1: [u8; 8] = [0x93, b'N', b'U', b'M', b'P', b'Y', 1, 0];

/// The R3 contract: `[rows, num_layers, top_k]`.
const ROUTED_EXPERTS_RANK: usize = 3;

/// The `.npy` v1.0 header numpy 2.x writes for a C-order array of `descr`
/// and `shape` (`numpy.lib.format._write_array_header` / `_wrap_header`).
/// Errors when the header does not fit version 1.0 (numpy would switch to
/// 2.0); unreachable for R3 shapes.
pub(crate) fn npy_header(descr: &str, shape: &[usize]) -> Result<Vec<u8>, String> {
    // repr(tuple): "()", "(5,)", "(5, 61, 8)".
    let shape_repr = match shape {
        [] => "()".to_string(),
        [single] => format!("({single},)"),
        _ => format!(
            "({})",
            shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")
        ),
    };
    let mut header =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_repr}, }}");
    if let Some(first) = shape.first() {
        let digits = first.to_string().len();
        header.push_str(&" ".repeat(GROWTH_AXIS_MAX_DIGITS.saturating_sub(digits)));
    }
    let hlen = header.len() + 1;
    let padlen = ARRAY_ALIGN - ((MAGIC_V1.len() + 2 + hlen) % ARRAY_ALIGN);
    let Ok(total) = u16::try_from(hlen + padlen) else {
        return Err(format!(
            "routed_experts .npy header of {} bytes does not fit version 1.0",
            hlen + padlen
        ));
    };
    let mut out = Vec::with_capacity(MAGIC_V1.len() + 2 + hlen + padlen);
    out.extend_from_slice(&MAGIC_V1);
    out.extend_from_slice(&total.to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    out.resize(out.len() + padlen, b' ');
    out.push(b'\n');
    Ok(out)
}

/// The dtype `np.concatenate` returns for chunks of `dtype`: native
/// (little-endian) byte order.
fn native_descr(dtype: &str) -> Cow<'_, str> {
    match dtype.strip_prefix('>') {
        Some(rest) => Cow::Owned(format!("<{rest}")),
        None => Cow::Borrowed(dtype),
    }
}

/// Reverse the bytes of every `size`-byte element.
fn swap_element_bytes(data: &[u8], size: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());
    for element in data.chunks_exact(size) {
        out.extend(element.iter().rev());
    }
    out
}

/// Incrementally builds the base64 `.npy` of all routed-experts chunks.
#[derive(Debug, Default)]
pub(crate) struct RoutedExpertsEncoder {
    /// Engine dtype string (as sent) and trailing axes, fixed by the first
    /// chunk; the output uses [`native_descr`] of it.
    layout: Option<(String, Vec<usize>)>,
    rows: usize,
    /// Header length (independent of the row count, see module docs).
    header_len: usize,
    /// The first `(3 - header_len % 3) % 3` data bytes.
    held: Vec<u8>,
    data: Base64Segments,
    error: Option<String>,
}

impl RoutedExpertsEncoder {
    /// Whether any chunk arrived (including a rejected one, so its error is
    /// reported even when the flag is off).
    pub(crate) fn has_data(&self) -> bool {
        self.layout.is_some() || self.error.is_some()
    }

    fn held_target(&self) -> usize {
        (3 - self.header_len % 3) % 3
    }

    /// Append one engine chunk (rows along axis 0).
    pub(crate) fn push(&mut self, chunk: RoutedExperts) {
        if self.error.is_some() {
            return;
        }
        if chunk.shape.len() != ROUTED_EXPERTS_RANK {
            // Not formatted: the rank can be huge.
            self.error = Some(format!(
                "routed_experts: expected shape [rows, num_layers, top_k], got {} axes",
                chunk.shape.len()
            ));
            return;
        }
        let Some(size) = RoutedExperts::itemsize(&chunk.dtype) else {
            self.error = Some(format!(
                "routed_experts: unsupported dtype {:?}",
                chunk.dtype
            ));
            return;
        };
        let expected = chunk.shape.iter().try_fold(size, |acc, &axis| acc.checked_mul(axis));
        if expected != Some(chunk.data.len()) {
            self.error = Some(format!(
                "routed_experts: byte length {} does not match shape {:?} dtype {}",
                chunk.data.len(),
                chunk.shape,
                chunk.dtype
            ));
            return;
        }
        let tail = &chunk.shape[1..];
        match &self.layout {
            None => match npy_header(&native_descr(&chunk.dtype), &chunk.shape) {
                Ok(header) => {
                    self.header_len = header.len();
                    self.layout = Some((chunk.dtype.clone(), tail.to_vec()));
                }
                Err(error) => {
                    self.error = Some(error);
                    return;
                }
            },
            Some((dtype, layout_tail)) => {
                if *dtype != chunk.dtype || layout_tail.as_slice() != tail {
                    self.error = Some(format!(
                        "routed_experts chunks changed layout: {dtype} {layout_tail:?} then {} {:?}",
                        chunk.dtype, chunk.shape
                    ));
                    return;
                }
            }
        }
        self.rows += chunk.rows();
        let swapped;
        let mut data = if chunk.dtype.starts_with('>') && size > 1 {
            swapped = swap_element_bytes(&chunk.data, size);
            swapped.as_slice()
        } else {
            chunk.data.as_ref()
        };
        let missing = self.held_target() - self.held.len();
        if missing > 0 {
            let take = missing.min(data.len());
            self.held.extend_from_slice(&data[..take]);
            data = &data[take..];
        }
        self.data.push(data);
    }

    /// The finished base64 `.npy`, `None` when no chunk arrived.
    pub(crate) fn finish(self) -> Result<Option<EncodedArray>, String> {
        if let Some(error) = self.error {
            return Err(error);
        }
        let Some((dtype, tail)) = &self.layout else {
            return Ok(None);
        };
        let shape: Vec<usize> = std::iter::once(self.rows).chain(tail.iter().copied()).collect();
        let mut head = npy_header(&native_descr(dtype), &shape)?;
        if head.len() != self.header_len {
            // Only possible with a row count beyond 21 digits.
            return Err(format!(
                "routed_experts .npy header length changed ({} -> {})",
                self.header_len,
                head.len()
            ));
        }
        head.extend_from_slice(&self.held);
        let mut first = Vec::with_capacity(head.len().div_ceil(3) * 4);
        encode_base64_into(&head, &mut first);
        let rest = self.data.finish();
        let mut segments = Vec::with_capacity(rest.segments.len() + 1);
        let len = first.len() + rest.len;
        segments.push(bytes::Bytes::from(first));
        segments.extend(rest.segments);
        Ok(Some(EncodedArray { segments, len }))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use base64::Engine as _;
    use base64::engine::general_purpose::STANDARD;
    use sha2::{Digest, Sha256};

    use super::*;

    /// Element `i` of the reference arrays: `(7 i + 3) % m` stored in the
    /// dtype's little-endian bytes (`m` = 251 / 65521 / 2147483629).
    pub(crate) fn reference_bytes(descr: &str, count: usize) -> Vec<u8> {
        let size = RoutedExperts::itemsize(descr).unwrap();
        let modulus: u64 = match size {
            1 => 251,
            2 => 65_521,
            _ => 2_147_483_629,
        };
        let mut out = Vec::with_capacity(count * size);
        for i in 0..count as u64 {
            let value = (i * 7 + 3) % modulus;
            out.extend_from_slice(&value.to_le_bytes()[..size]);
        }
        out
    }

    fn chunk(descr: &str, shape: &[usize], data: &[u8]) -> RoutedExperts {
        RoutedExperts {
            dtype: descr.to_string(),
            shape: shape.to_vec(),
            data: bytes::Bytes::copy_from_slice(data),
        }
    }

    /// Vectors produced by numpy 2.2.6 (`/home/inf-aoshen/.venv`):
    /// `base64.b64encode(np.save(arr))` for `arr` = reference values reshaped
    /// to `shape` with dtype `descr`: (descr, shape, sha256 of the base64
    /// text, base64 length, first 120 base64 characters).
    const NUMPY_VECTORS: &[(&str, &[usize], &str, usize, &str)] = &[
        (
            "|u1",
            &[1, 4, 8],
            "dbedd44e4556020ca02cd584146de020c5c6120cce414f9730fade98156696ba",
            216,
            "k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDEsIDQsIDgpLCB9ICAgICAgICAgICAgICAgICAg",
        ),
        (
            "|u1",
            &[22, 61, 8],
            "48646c763781d1484d5bbda470514a4b879ecf37d5206a4077bfe3f577907594",
            14488,
            "k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDIyLCA2MSwgOCksIH0gICAgICAgICAgICAgICAg",
        ),
        (
            "|u1",
            &[0, 61, 8],
            "2367878c37678c27a07330bf538129baf39111afca3e64d41ca50be0c844faaa",
            172,
            "k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDAsIDYxLCA4KSwgfSAgICAgICAgICAgICAgICAg",
        ),
        (
            "<u2",
            &[7, 4, 8],
            "22fcf1234a66faec92b5d48c2bae0f7c9c87adfb3cee1a25a8159fcfa4ef7d6a",
            768,
            "k05VTVBZAQB2AHsnZGVzY3InOiAnPHUyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDcsIDQsIDgpLCB9ICAgICAgICAgICAgICAgICAg",
        ),
        (
            "<i4",
            &[3, 2],
            "b41ac2da2962f164b4a46543c86586274e7da3b9533a833f43eccfc8b88ddfa4",
            204,
            "k05VTVBZAQB2AHsnZGVzY3InOiAnPGk0JywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDMsIDIpLCB9ICAgICAgICAgICAgICAgICAgICAg",
        ),
        (
            "|u1",
            &[262143, 4, 8],
            "505f345b45275d1044d97059e5d09d26c2652182dee323ab7c05860e0cf668de",
            11184940,
            "k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDI2MjE0MywgNCwgOCksIH0gICAgICAgICAgICAg",
        ),
        (
            "<u2",
            &[16385, 61, 8],
            "c2e2ae5cf0fb76e32cb646ca2cf309c7d1240460483b347cddf3872cd6d5f854",
            21322520,
            "k05VTVBZAQB2AHsnZGVzY3InOiAnPHUyJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDE2Mzg1LCA2MSwgOCksIH0gICAgICAgICAgICAg",
        ),
        (
            "|u1",
            &[5],
            "685bd4ee3e28b4061c27ef89f49520345a2fb0f8d890c982c1fcb19eaf9886ce",
            180,
            "k05VTVBZAQB2AHsnZGVzY3InOiAnfHUxJywgJ2ZvcnRyYW5fb3JkZXInOiBGYWxzZSwgJ3NoYXBlJzogKDUsKSwgfSAgICAgICAgICAgICAgICAgICAgICAg",
        ),
    ];

    #[test]
    fn npy_header_matches_numpy_layout() {
        let npy_header = |descr: &str, shape: &[usize]| npy_header(descr, shape).unwrap();
        let header = npy_header("|u1", &[16384, 61, 8]);
        assert_eq!(header.len(), 128);
        assert_eq!(&header[..8], &MAGIC_V1);
        assert_eq!(u16::from_le_bytes([header[8], header[9]]), 118);
        assert!(header.ends_with(b" \n"));
        let text = String::from_utf8_lossy(&header[10..]);
        assert!(
            text.starts_with("{'descr': '|u1', 'fortran_order': False, 'shape': (16384, 61, 8), }")
        );
        assert_eq!(npy_header("|u1", &[5]).len(), 128);
        assert!(String::from_utf8_lossy(&npy_header("|u1", &[5])).contains("'shape': (5,), }"));
        // Header length does not depend on the row count's digits.
        assert_eq!(
            npy_header("<u2", &[1, 61, 8]).len(),
            npy_header("<u2", &[262_143, 61, 8]).len()
        );
    }

    #[test]
    fn incremental_npy_base64_matches_numpy_vectors_for_any_chunking() {
        for &(descr, shape, sha256, b64_len, b64_prefix) in NUMPY_VECTORS {
            let count: usize = shape.iter().product();
            let data = reference_bytes(descr, count);
            let row_elems: usize = shape[1..].iter().product();
            let row_bytes = row_elems * RoutedExperts::itemsize(descr).unwrap();
            if shape.len() != 3 {
                // Header layout check for other ranks (one-shot); the encoder
                // only accepts the R3 layout.
                let mut whole = npy_header(descr, shape).unwrap();
                whole.extend_from_slice(&data);
                let text = STANDARD.encode(&whole);
                assert_eq!(text.len(), b64_len);
                assert_eq!(format!("{:x}", Sha256::digest(text.as_bytes())), sha256);
                let mut encoder = RoutedExpertsEncoder::default();
                encoder.push(chunk(descr, shape, &data));
                assert!(encoder.finish().is_err(), "{descr} {shape:?}");
                continue;
            }
            // Chunkings: one chunk, single rows, and uneven multi-row chunks
            // (the engine's prompt block then per-token rows).
            for rows_per_chunk in [shape[0].max(1), 1024, 7, 3, 2, 1] {
                let mut encoder = RoutedExpertsEncoder::default();
                if shape[0] == 0 {
                    encoder.push(chunk(descr, shape, &data));
                }
                for rows in data.chunks(row_bytes * rows_per_chunk) {
                    let mut chunk_shape = shape.to_vec();
                    chunk_shape[0] = rows.len() / row_bytes;
                    encoder.push(chunk(descr, &chunk_shape, rows));
                }
                let text = encoder.finish().unwrap().unwrap().to_base64_string();
                assert_eq!(
                    text.len(),
                    b64_len,
                    "{descr} {shape:?} rows/chunk {rows_per_chunk}"
                );
                assert_eq!(&text[..b64_prefix.len()], b64_prefix);
                let digest = format!("{:x}", Sha256::digest(text.as_bytes()));
                assert_eq!(
                    digest, sha256,
                    "{descr} {shape:?} rows/chunk {rows_per_chunk}"
                );
                if rows_per_chunk == 1024 && count > 1_000_000 {
                    break;
                }
            }
        }
    }

    #[test]
    fn incremental_npy_equals_one_shot_encoding() {
        for (descr, tail, rows_per_chunk) in [
            ("|u1", vec![4, 8], vec![16_384, 1, 1, 1024, 5]),
            ("<u2", vec![61, 8], vec![3, 1, 2]),
            ("|u1", vec![61, 8], vec![1]),
            ("<i4", vec![2, 3], vec![1, 1, 1, 1]),
            ("<u8", vec![3, 2], vec![2, 1]),
        ] {
            let row_elems: usize = tail.iter().product();
            let size = RoutedExperts::itemsize(descr).unwrap();
            let total_rows: usize = rows_per_chunk.iter().sum();
            let data = reference_bytes(descr, total_rows * row_elems);
            let mut encoder = RoutedExpertsEncoder::default();
            let mut offset = 0;
            for rows in &rows_per_chunk {
                let len = rows * row_elems * size;
                let shape: Vec<usize> =
                    std::iter::once(*rows).chain(tail.iter().copied()).collect();
                encoder.push(chunk(descr, &shape, &data[offset..offset + len]));
                offset += len;
            }
            let shape: Vec<usize> =
                std::iter::once(total_rows).chain(tail.iter().copied()).collect();
            let mut whole = npy_header(descr, &shape).unwrap();
            whole.extend_from_slice(&data);
            assert_eq!(
                encoder.finish().unwrap().unwrap().to_base64_string(),
                STANDARD.encode(&whole),
                "{descr} {tail:?} {rows_per_chunk:?}"
            );
        }
    }

    #[test]
    fn no_chunk_is_null_and_layout_changes_fail() {
        assert!(RoutedExpertsEncoder::default().finish().unwrap().is_none());
        let mut encoder = RoutedExpertsEncoder::default();
        encoder.push(chunk("|u1", &[1, 4, 8], &[0; 32]));
        encoder.push(chunk("|u1", &[1, 5, 8], &[0; 40]));
        assert!(encoder.finish().is_err());
        let mut encoder = RoutedExpertsEncoder::default();
        encoder.push(chunk("|u1", &[1, 4, 8], &[0; 32]));
        encoder.push(chunk("<u2", &[1, 4, 8], &[0; 64]));
        assert!(encoder.finish().is_err());
    }

    /// numpy 2.2.6, `scripts/r6-be-vectors.py`: `numpy2base64(np.concatenate(
    /// chunks))` for big-endian chunks of shape (5, 3, 4) split [2, 1, 2]
    /// (identical for a single chunk): (kind, output descr, sha256 of the
    /// base64 text, base64 length).
    const BIG_ENDIAN_VECTORS: &[(&str, &str, &str, usize)] = &[
        (
            "u2",
            "<u2",
            "8ddcde2f0ef9b73cb59e9ec938c51161c28397a80be45451c3b035b2653ed481",
            332,
        ),
        (
            "u4",
            "<u4",
            "cbc600960cb9e726773862e01441e120dc3420c120937ba8cf98fa04201723b0",
            492,
        ),
        (
            "u8",
            "<u8",
            "94a828f48dd6340064a35d5e839f829e495aae7900aaa427a240b778d7ef5f94",
            812,
        ),
        (
            "i2",
            "<i2",
            "9a77b43cffe9719dd0e12aed01e5b738564f87b310ee829c0ce7e274a63f6bd1",
            332,
        ),
        (
            "i4",
            "<i4",
            "723813963b10d285aef9d6ae420f873a4621636d85148f16600855c63dda2a5e",
            492,
        ),
        (
            "i8",
            "<i8",
            "eb653749d55a40fff6551a4bdfe9434f1caecf3daa150636017886250562f046",
            812,
        ),
    ];

    #[test]
    fn big_endian_chunks_are_written_native_like_numpy_concatenate() {
        for &(kind, out_descr, sha256, b64_len) in BIG_ENDIAN_VECTORS {
            let size = RoutedExperts::itemsize(&format!("<{kind}")).unwrap();
            let le = reference_bytes(&format!("<{kind}"), 5 * 3 * 4);
            let be = swap_element_bytes(&le, size);
            let descr = format!(">{kind}");
            let row = 3 * 4 * size;
            for split in [vec![5], vec![2, 1, 2], vec![1, 1, 1, 1, 1]] {
                let mut encoder = RoutedExpertsEncoder::default();
                let mut offset = 0;
                for rows in &split {
                    encoder.push(chunk(
                        &descr,
                        &[*rows, 3, 4],
                        &be[offset..offset + rows * row],
                    ));
                    offset += rows * row;
                }
                let text = encoder.finish().unwrap().unwrap().to_base64_string();
                assert_eq!(text.len(), b64_len, "{descr} {split:?}");
                assert_eq!(
                    format!("{:x}", Sha256::digest(text.as_bytes())),
                    sha256,
                    "{descr} {split:?}"
                );
                // Same bytes as the little-endian input encoded directly.
                let mut whole = npy_header(out_descr, &[5, 3, 4]).unwrap();
                whole.extend_from_slice(&le);
                assert_eq!(text, STANDARD.encode(&whole), "{descr} {split:?}");
            }
        }
    }

    /// Wire-accepted metadata outside the R3 contract (e.g. 22,000 axes of
    /// 1 with one byte) fails the request instead of panicking.
    #[test]
    fn non_r3_shapes_fail_without_panicking() {
        let huge: Vec<usize> = vec![1; 22_000];
        assert!(npy_header("|u1", &huge).is_err());
        for shape in [huge, vec![1], vec![1, 8], vec![1, 1, 1, 1]] {
            let mut encoder = RoutedExpertsEncoder::default();
            encoder.push(chunk("|u1", &shape, &[3]));
            let error = encoder.finish().unwrap_err();
            assert!(error.len() < 200, "{error}");
            // Later valid chunks do not clear the error.
            let mut encoder = RoutedExpertsEncoder::default();
            encoder.push(chunk("|u1", &[1, 1, 1], &[3]));
            encoder.push(chunk("|u1", &shape, &[3]));
            assert!(encoder.finish().is_err());
        }
        // Length and dtype checks.
        let mut encoder = RoutedExpertsEncoder::default();
        encoder.push(chunk("|u1", &[2, 1, 1], &[3]));
        assert!(encoder.finish().is_err());
        let mut encoder = RoutedExpertsEncoder::default();
        encoder.push(chunk("<f4", &[1, 1, 1], &[0; 4]));
        assert!(encoder.finish().is_err());
    }
}
