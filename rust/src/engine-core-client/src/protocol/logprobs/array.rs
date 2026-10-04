// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use crate::error::{Error, Result, ext_value_decode};
use crate::protocol::tensor::{ShapeExt as _, WireArrayData, WireNdArray};

/// `"{prefix}.{name}"` for error messages, formatted only when an error is
/// actually reported (the success path decodes thousands of arrays per
/// second and should not allocate field-name strings).
#[derive(Debug, Clone, Copy)]
pub(super) struct FieldName<'a> {
    pub prefix: &'a str,
    pub name: &'a str,
}

impl std::fmt::Display for FieldName<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.prefix, self.name)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ScalarType {
    I32,
    I64,
    F32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Endianness {
    Little,
    Big,
    Native,
}

pub(super) fn decode_array1_u32<Frame>(
    value: WireNdArray,
    field: &FieldName<'_>,
    frames: &[Frame],
) -> Result<Vec<u32>>
where
    Frame: AsRef<[u8]>,
{
    let (shape, bytes, scalar, endianness) =
        decode_array_metadata(&value, field, frames, &[ScalarType::I32, ScalarType::I64])?;
    if shape.len() != 1 {
        return Err(decode_error(
            field,
            &format!("expected rank-1 array, got rank {}", shape.len()),
        ));
    }

    decode_u32_values(bytes, scalar, endianness, field)
}

/// Validate dtype/shape/length and return the array's raw bytes, borrowed
/// from the inline raw view or directly from the aux frame (no copy).
pub(super) fn decode_array_metadata<'a, Frame>(
    value: &'a WireNdArray,
    field: &FieldName<'_>,
    frames: &'a [Frame],
    expected_scalars: &[ScalarType],
) -> Result<(&'a [usize], &'a [u8], ScalarType, Endianness)>
where
    Frame: AsRef<[u8]>,
{
    let WireNdArray { dtype, shape, data } = value;
    let (scalar, endianness) = parse_dtype(dtype, field)?;
    if !expected_scalars.contains(&scalar) {
        return Err(decode_error(
            field,
            &format!("expected dtype in {:?}, got {}", expected_scalars, dtype),
        ));
    }

    let bytes = resolve_array_bytes(data, field, frames)?;
    validate_byte_length(shape.as_slice(), bytes.len(), field, scalar)?;
    Ok((shape.as_slice(), bytes, scalar, endianness))
}

pub(super) fn parse_dtype(dtype: &str, field: &FieldName<'_>) -> Result<(ScalarType, Endianness)> {
    let (endianness, body) = match dtype.as_bytes().first().copied() {
        Some(b'<') => (Endianness::Little, &dtype[1..]),
        Some(b'>') => (Endianness::Big, &dtype[1..]),
        Some(b'=') => (Endianness::Native, &dtype[1..]),
        Some(b'|') => (Endianness::Native, &dtype[1..]),
        _ => (Endianness::Native, dtype),
    };

    let scalar = match body {
        "i4" | "int32" => ScalarType::I32,
        "i8" | "int64" => ScalarType::I64,
        "f4" | "float32" => ScalarType::F32,
        _ => {
            return Err(decode_error(
                field,
                &format!("unsupported dtype string {dtype:?}"),
            ));
        }
    };
    Ok((scalar, endianness))
}

pub(super) fn resolve_array_bytes<'a, Frame>(
    value: &'a WireArrayData,
    field: &FieldName<'_>,
    frames: &'a [Frame],
) -> Result<&'a [u8]>
where
    Frame: AsRef<[u8]>,
{
    match value {
        WireArrayData::RawView(bytes) => Ok(bytes.as_ref()),
        WireArrayData::AuxIndex(index) => {
            let frame = frames.get(*index).ok_or_else(|| {
                decode_error(
                    field,
                    &format!(
                        "aux frame index {index} out of range for {} frames",
                        frames.len()
                    ),
                )
            })?;
            Ok(frame.as_ref())
        }
    }
}

pub(super) fn validate_byte_length(
    shape: &[usize],
    byte_len: usize,
    field: &FieldName<'_>,
    scalar: ScalarType,
) -> Result<()> {
    let element_count = shape
        .checked_numel()
        .ok_or_else(|| decode_error(field, "shape element count overflowed usize"))?;
    let element_size = match scalar {
        ScalarType::I32 | ScalarType::F32 => 4,
        ScalarType::I64 => 8,
    };
    let expected = element_count
        .checked_mul(element_size)
        .ok_or_else(|| decode_error(field, "byte length overflowed usize"))?;
    if expected != byte_len {
        return Err(decode_error(
            field,
            &format!("byte length mismatch: expected {expected}, got {byte_len}"),
        ));
    }
    Ok(())
}

/// Decode i32/i64 words straight into `u32` (token ids / ranks) in one pass,
/// rejecting values that do not fit, without an intermediate signed vector.
fn decode_u32_values(
    bytes: &[u8],
    scalar: ScalarType,
    endianness: Endianness,
    field: &FieldName<'_>,
) -> Result<Vec<u32>> {
    fn convert<const N: usize>(
        bytes: &[u8],
        field: &FieldName<'_>,
        word: impl Fn([u8; N]) -> i64,
    ) -> Result<Vec<u32>> {
        if !bytes.len().is_multiple_of(N) {
            return Err(decode_error(
                field,
                &format!("byte length {} is not divisible by {N}", bytes.len()),
            ));
        }
        let mut values = Vec::with_capacity(bytes.len() / N);
        for chunk in bytes.chunks_exact(N) {
            let value = word(chunk.try_into().expect("chunks_exact yields N bytes"));
            let Ok(value) = u32::try_from(value) else {
                return Err(convert_error(value, field));
            };
            values.push(value);
        }
        Ok(values)
    }
    match (scalar, endianness) {
        (ScalarType::I32, Endianness::Little) => {
            convert(bytes, field, |w| i32::from_le_bytes(w) as i64)
        }
        (ScalarType::I32, Endianness::Big) => {
            convert(bytes, field, |w| i32::from_be_bytes(w) as i64)
        }
        (ScalarType::I32, Endianness::Native) => {
            convert(bytes, field, |w| i32::from_ne_bytes(w) as i64)
        }
        (ScalarType::I64, Endianness::Little) => convert(bytes, field, i64::from_le_bytes),
        (ScalarType::I64, Endianness::Big) => convert(bytes, field, i64::from_be_bytes),
        (ScalarType::I64, Endianness::Native) => convert(bytes, field, i64::from_ne_bytes),
        (ScalarType::F32, _) => unreachable!("scalar validation should reject f32"),
    }
}

fn convert_error(value: impl std::fmt::Display, field: &FieldName<'_>) -> Error {
    decode_error(
        field,
        &format!("expected non-negative token id/rank that fits in u32, got {value}"),
    )
}

pub(super) fn decode_error(field: &FieldName<'_>, reason: &str) -> Error {
    ext_value_decode!("{field}: {reason}")
}

/// A validated rank-2 integer or float array, borrowed from its raw view or
/// aux frame.
#[derive(Debug, Clone, Copy)]
pub(super) struct ArrayView2<'a> {
    pub rows: usize,
    pub cols: usize,
    pub bytes: &'a [u8],
    pub scalar: ScalarType,
    pub endianness: Endianness,
}

impl ArrayView2<'_> {
    fn word_size(&self) -> usize {
        match self.scalar {
            ScalarType::I32 | ScalarType::F32 => 4,
            ScalarType::I64 => 8,
        }
    }

    /// The raw bytes of row `row`.
    pub fn row(&self, row: usize) -> &[u8] {
        let width = self.cols * self.word_size();
        &self.bytes[row * width..(row + 1) * width]
    }
}

/// Validate a rank-2 array (dtype, rank, byte length) without decoding it.
pub(super) fn view_array2<'a, Frame>(
    value: &'a WireNdArray,
    field: &FieldName<'_>,
    frames: &'a [Frame],
    expected_scalars: &[ScalarType],
) -> Result<ArrayView2<'a>>
where
    Frame: AsRef<[u8]>,
{
    let (shape, bytes, scalar, endianness) =
        decode_array_metadata(value, field, frames, expected_scalars)?;
    if shape.len() != 2 {
        return Err(decode_error(
            field,
            &format!("expected rank-2 array, got rank {}", shape.len()),
        ));
    }
    Ok(ArrayView2 {
        rows: shape[0],
        cols: shape[1],
        bytes,
        scalar,
        endianness,
    })
}

/// Check that every integer word fits `u32` (same error as decoding it),
/// without materializing the values.
pub(super) fn validate_u32_words(view: &ArrayView2<'_>, field: &FieldName<'_>) -> Result<()> {
    fn check<const N: usize>(
        bytes: &[u8],
        field: &FieldName<'_>,
        word: impl Fn([u8; N]) -> i64,
    ) -> Result<()> {
        for chunk in bytes.chunks_exact(N) {
            let value = word(chunk.try_into().expect("chunks_exact yields N bytes"));
            if u32::try_from(value).is_err() {
                return Err(convert_error(value, field));
            }
        }
        Ok(())
    }
    let bytes = view.bytes;
    match (view.scalar, view.endianness) {
        (ScalarType::I32, Endianness::Little) => {
            check(bytes, field, |w| i32::from_le_bytes(w) as i64)
        }
        (ScalarType::I32, Endianness::Big) => check(bytes, field, |w| i32::from_be_bytes(w) as i64),
        (ScalarType::I32, Endianness::Native) => {
            check(bytes, field, |w| i32::from_ne_bytes(w) as i64)
        }
        (ScalarType::I64, Endianness::Little) => check(bytes, field, i64::from_le_bytes),
        (ScalarType::I64, Endianness::Big) => check(bytes, field, i64::from_be_bytes),
        (ScalarType::I64, Endianness::Native) => check(bytes, field, i64::from_ne_bytes),
        (ScalarType::F32, _) => unreachable!("scalar validation should reject f32"),
    }
}

/// Append one token id per word of a validated integer row (values already
/// checked by [`validate_u32_words`]).
pub(super) fn push_token_ids(
    row: &[u8],
    scalar: ScalarType,
    endianness: Endianness,
    mut push: impl FnMut(u32),
) {
    fn each<const N: usize>(row: &[u8], word: impl Fn([u8; N]) -> u32, push: &mut impl FnMut(u32)) {
        for chunk in row.chunks_exact(N) {
            push(word(chunk.try_into().expect("chunks_exact yields N bytes")));
        }
    }
    match (scalar, endianness) {
        (ScalarType::I32, Endianness::Little) => {
            each(row, |w| i32::from_le_bytes(w) as u32, &mut push)
        }
        (ScalarType::I32, Endianness::Big) => {
            each(row, |w| i32::from_be_bytes(w) as u32, &mut push)
        }
        (ScalarType::I32, Endianness::Native) => {
            each(row, |w| i32::from_ne_bytes(w) as u32, &mut push)
        }
        (ScalarType::I64, Endianness::Little) => {
            each(row, |w| i64::from_le_bytes(w) as u32, &mut push)
        }
        (ScalarType::I64, Endianness::Big) => {
            each(row, |w| i64::from_be_bytes(w) as u32, &mut push)
        }
        (ScalarType::I64, Endianness::Native) => {
            each(row, |w| i64::from_ne_bytes(w) as u32, &mut push)
        }
        (ScalarType::F32, _) => unreachable!("scalar validation should reject f32"),
    }
}

/// Visit each f32 of a validated float row.
pub(super) fn for_each_f32(row: &[u8], endianness: Endianness, mut visit: impl FnMut(usize, f32)) {
    let read: fn([u8; 4]) -> f32 = match endianness {
        Endianness::Little => f32::from_le_bytes,
        Endianness::Big => f32::from_be_bytes,
        Endianness::Native => f32::from_ne_bytes,
    };
    for (index, chunk) in row.chunks_exact(4).enumerate() {
        visit(
            index,
            read(chunk.try_into().expect("chunks_exact yields 4 bytes")),
        );
    }
}
