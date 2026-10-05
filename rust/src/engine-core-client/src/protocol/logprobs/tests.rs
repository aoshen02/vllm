// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::BTreeSet;

use bytes::Bytes;
use rmpv::Value;

use super::{Logprobs, PositionLogprobs, TokenLogprob};
use crate::protocol::output::{EngineCoreFinishReason, decode_engine_core_outputs};

fn encode_value(value: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, value).unwrap();
    out
}

fn output_wire_with_custom_fields(
    new_logprobs: Option<Value>,
    prompt_logprobs: Option<Value>,
) -> Value {
    Value::Array(vec![
        Value::from(0),
        Value::Array(vec![Value::Array(vec![
            Value::from("req-1"),
            Value::Array(vec![Value::from(7), Value::from(8)]),
            new_logprobs.unwrap_or(Value::Nil),
            prompt_logprobs.unwrap_or(Value::Nil),
            Value::Nil,
            Value::from(EngineCoreFinishReason::Length as u8),
        ])]),
        Value::Nil,
        Value::from(0.0),
        Value::Nil,
        Value::Array(vec![Value::from("req-1")]),
    ])
}

fn ndarray_value(dtype: &str, shape: &[usize], data: Value) -> Value {
    Value::Array(vec![
        Value::from(dtype),
        Value::Array(shape.iter().copied().map(Value::from).collect()),
        data,
    ])
}

fn inline_logprobs_value() -> Value {
    let ids = Value::Ext(
        3,
        vec![
            1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
            0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
        ],
    );
    let probs = Value::Ext(
        3,
        vec![
            0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64,
        ],
    );
    let ranks = Value::Ext(3, vec![1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
    Value::Array(vec![
        ndarray_value("<i8", &[2, 3], ids),
        ndarray_value("<f4", &[2, 3], probs),
        ndarray_value("<i8", &[2], ranks),
        Value::Nil,
    ])
}

fn inline_prompt_logprobs_value() -> Value {
    let ids = Value::Ext(
        3,
        vec![
            10, 0, 0, 0, 0, 0, 0, 0, 11, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0,
            0, 0, 0, 0, 14, 0, 0, 0, 0, 0, 0, 0, 15, 0, 0, 0, 0, 0, 0, 0,
        ],
    );
    let probs = Value::Ext(
        3,
        vec![
            0, 0, 32, 65, 0, 0, 48, 65, 0, 0, 64, 65, 0, 0, 80, 65, 0, 0, 96, 65, 0, 0, 112, 65,
        ],
    );
    let ranks = Value::Ext(3, vec![3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0]);
    Value::Array(vec![
        ndarray_value("int64", &[2, 3], ids),
        ndarray_value("float32", &[2, 3], probs),
        ndarray_value("int64", &[2], ranks),
        Value::Nil,
        Value::Nil,
    ])
}

fn expected_sample_logprobs() -> Logprobs {
    Logprobs {
        positions: vec![
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 1,
                        logprob: 1.0,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: 2,
                        logprob: 2.0,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: 3,
                        logprob: 3.0,
                        rank: 2,
                    },
                ],
            },
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 4,
                        logprob: 4.0,
                        rank: 2,
                    },
                    TokenLogprob {
                        token_id: 5,
                        logprob: 5.0,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: 6,
                        logprob: 6.0,
                        rank: 2,
                    },
                ],
            },
        ],
    }
}

fn expected_prompt_logprobs() -> Logprobs {
    Logprobs {
        positions: vec![
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 10,
                        logprob: 10.0,
                        rank: 3,
                    },
                    TokenLogprob {
                        token_id: 11,
                        logprob: 11.0,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: 12,
                        logprob: 12.0,
                        rank: 2,
                    },
                ],
            },
            PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 13,
                        logprob: 13.0,
                        rank: 4,
                    },
                    TokenLogprob {
                        token_id: 14,
                        logprob: 14.0,
                        rank: 1,
                    },
                    TokenLogprob {
                        token_id: 15,
                        logprob: 15.0,
                        rank: 2,
                    },
                ],
            },
        ],
    }
}

#[test]
fn decodes_inline_new_logprobs() {
    let frames = vec![Bytes::from(encode_value(&output_wire_with_custom_fields(
        Some(inline_logprobs_value()),
        None,
    )))];
    let decoded = decode_engine_core_outputs(&frames).unwrap().into_request_batch().unwrap();

    let logprobs = decoded.outputs[0].new_logprobs.clone().unwrap().into_direct().unwrap();
    assert_eq!(logprobs, expected_sample_logprobs());
    assert_eq!(
        decoded.finished_requests,
        Some(BTreeSet::from(["req-1".to_string()]))
    );
}

#[test]
fn decodes_multipart_new_logprobs() {
    let frames = vec![
        Bytes::from(encode_value(&output_wire_with_custom_fields(
            Some(Value::Array(vec![
                ndarray_value("<i8", &[2, 3], Value::from(1)),
                ndarray_value("<f4", &[2, 3], Value::from(2)),
                ndarray_value("<i8", &[2], Value::from(3)),
                Value::Nil,
            ])),
            None,
        ))),
        Bytes::from_static(&[
            1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0,
            0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0,
        ]),
        Bytes::from_static(&[
            0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64, 0, 0, 160, 64, 0, 0, 192, 64,
        ]),
        Bytes::from_static(&[1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]),
    ];
    let decoded = decode_engine_core_outputs(&frames).unwrap().into_request_batch().unwrap();

    let logprobs = decoded.outputs[0].new_logprobs.clone().unwrap().into_direct().unwrap();
    assert_eq!(logprobs, expected_sample_logprobs());
}

#[test]
fn decodes_inline_prompt_logprobs() {
    let frames = vec![Bytes::from(encode_value(&output_wire_with_custom_fields(
        None,
        Some(inline_prompt_logprobs_value()),
    )))];
    let decoded = decode_engine_core_outputs(&frames).unwrap().into_request_batch().unwrap();

    let logprobs = decoded.outputs[0]
        .new_prompt_logprobs_tensors
        .clone()
        .unwrap()
        .into_direct()
        .unwrap();
    assert_eq!(logprobs, expected_prompt_logprobs());
}

#[test]
fn rejects_non_none_cu_num_generated_tokens_tensor() {
    let Value::Array(mut fields) = inline_prompt_logprobs_value() else {
        panic!("inline_prompt_logprobs_value must be an array");
    };
    fields[4] = Value::from(42);

    let frames = vec![Bytes::from(encode_value(&output_wire_with_custom_fields(
        None,
        Some(Value::Array(fields)),
    )))];

    let error = decode_engine_core_outputs(&frames).unwrap_err();
    let crate::error::Error::ExtValueDecode { message } = &error else {
        panic!("expected ValueDecodeExt");
    };
    assert_eq!(
        message,
        "new_prompt_logprobs_tensors.cu_num_generated_tokens_tensor: \
         expected None for per-request engine-core logprobs payload"
    );
}

#[test]
fn decodes_big_endian_payloads() {
    let frames = vec![Bytes::from(encode_value(&output_wire_with_custom_fields(
        Some(Value::Array(vec![
            ndarray_value(">i4", &[1, 2], Value::Ext(3, vec![0, 0, 0, 1, 0, 0, 0, 2])),
            ndarray_value(
                ">f4",
                &[1, 2],
                Value::Ext(3, vec![63, 128, 0, 0, 64, 0, 0, 0]),
            ),
            ndarray_value(">i4", &[1], Value::Ext(3, vec![0, 0, 0, 3])),
            Value::Nil,
        ])),
        None,
    )))];
    let decoded = decode_engine_core_outputs(&frames).unwrap().into_request_batch().unwrap();
    let logprobs = decoded.outputs[0].new_logprobs.clone().unwrap().into_direct().unwrap();
    assert_eq!(
        logprobs,
        Logprobs {
            positions: vec![PositionLogprobs {
                entries: vec![
                    TokenLogprob {
                        token_id: 1,
                        logprob: 1.0,
                        rank: 3,
                    },
                    TokenLogprob {
                        token_id: 2,
                        logprob: 2.0,
                        rank: 1,
                    },
                ],
            }],
        }
    );
}

#[test]
fn rejects_non_none_cu_num_generated_tokens() {
    let frames = vec![Bytes::from(encode_value(&output_wire_with_custom_fields(
        Some(Value::Array(vec![
            ndarray_value("<i8", &[1, 1], Value::Ext(3, vec![1, 0, 0, 0, 0, 0, 0, 0])),
            ndarray_value("<f4", &[1, 1], Value::Ext(3, vec![0, 0, 128, 63])),
            ndarray_value("<i8", &[1], Value::Ext(3, vec![1, 0, 0, 0, 0, 0, 0, 0])),
            Value::Array(vec![Value::from(0usize), Value::from(1usize)]),
        ])),
        None,
    )))];

    let error = decode_engine_core_outputs(&frames).unwrap_err();
    let crate::error::Error::ExtValueDecode { message } = &error else {
        panic!("expected ValueDecodeExt");
    };
    assert_eq!(
        message,
        "new_logprobs.cu_num_generated_tokens: expected None for per-request engine-core logprobs payload, got [0, 1]"
    );
    assert_eq!(
        error.to_string(),
        "messagepack ext value decode failed: new_logprobs.cu_num_generated_tokens: expected None for per-request engine-core logprobs payload, got [0, 1]"
    );
}

#[test]
fn decodes_zero_row_logprobs_as_empty() {
    for shape in [[0usize, 0], [0, 3]] {
        let frames = vec![Bytes::from(encode_value(&output_wire_with_custom_fields(
            None,
            Some(Value::Array(vec![
                ndarray_value("<i8", &shape, Value::Ext(3, Vec::new())),
                ndarray_value("<f4", &shape, Value::Ext(3, Vec::new())),
                ndarray_value("<i8", &[0], Value::Ext(3, Vec::new())),
                Value::Nil,
            ])),
        )))];
        let decoded = decode_engine_core_outputs(&frames).unwrap().into_request_batch().unwrap();
        let logprobs = decoded.outputs[0]
            .new_prompt_logprobs_tensors
            .clone()
            .unwrap()
            .into_direct()
            .unwrap();
        assert!(logprobs.is_empty());
    }
}

#[test]
fn rejects_zero_column_logprobs_with_rows() {
    let ranks = Value::Ext(3, vec![1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]);
    let frames = vec![Bytes::from(encode_value(&output_wire_with_custom_fields(
        Some(Value::Array(vec![
            ndarray_value("<i8", &[2, 0], Value::Ext(3, Vec::new())),
            ndarray_value("<f4", &[2, 0], Value::Ext(3, Vec::new())),
            ndarray_value("<i8", &[2], ranks),
            Value::Nil,
        ])),
        None,
    )))];

    let error = decode_engine_core_outputs(&frames).unwrap_err();
    let crate::error::Error::ExtValueDecode { message } = &error else {
        panic!("expected ExtValueDecode");
    };
    assert_eq!(
        message,
        "new_logprobs: zero-column logprobs payload with 2 rows"
    );
}

/// The decoder before the round-4 rewrite (Cursor element reads, aux frame
/// copied, intermediate vectors, per-row assembly), reproduced as the
/// reference for the randomized equivalence test below.
mod reference {
    use std::io::Cursor;

    use byteorder::{BigEndian, LittleEndian, NativeEndian, ReadBytesExt};

    use crate::error::{Error, Result, bail_ext_value_decode, ext_value_decode};
    use crate::protocol::logprobs::{Logprobs, PositionLogprobs, TokenLogprob};
    use crate::protocol::tensor::{ShapeExt as _, WireArrayData, WireNdArray};

    fn err(field: &str, reason: &str) -> Error {
        ext_value_decode!("{field}: {reason}")
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum Scalar {
        I32,
        I64,
        F32,
    }

    #[derive(Clone, Copy, PartialEq)]
    enum Endian {
        Little,
        Big,
        Native,
    }

    fn dtype(dtype: &str, field: &str) -> Result<(Scalar, Endian)> {
        let (endian, body) = match dtype.as_bytes().first().copied() {
            Some(b'<') => (Endian::Little, &dtype[1..]),
            Some(b'>') => (Endian::Big, &dtype[1..]),
            Some(b'=') | Some(b'|') => (Endian::Native, &dtype[1..]),
            _ => (Endian::Native, dtype),
        };
        let scalar = match body {
            "i4" | "int32" => Scalar::I32,
            "i8" | "int64" => Scalar::I64,
            "f4" | "float32" => Scalar::F32,
            _ => return Err(err(field, &format!("unsupported dtype string {dtype:?}"))),
        };
        Ok((scalar, endian))
    }

    fn metadata<F: AsRef<[u8]>>(
        value: WireNdArray,
        field: &str,
        frames: &[F],
        expected: &[Scalar],
    ) -> Result<(Vec<usize>, Vec<u8>, Scalar, Endian)> {
        let WireNdArray {
            dtype: name,
            shape,
            data,
        } = value;
        let (scalar, endian) = dtype(&name, field)?;
        if !expected.contains(&scalar) {
            return Err(err(
                field,
                &format!("expected dtype in {expected:?}, got {name}"),
            ));
        }
        let bytes = match data {
            WireArrayData::RawView(bytes) => bytes.to_vec(),
            WireArrayData::AuxIndex(index) => frames
                .get(index)
                .ok_or_else(|| {
                    err(
                        field,
                        &format!(
                            "aux frame index {index} out of range for {} frames",
                            frames.len()
                        ),
                    )
                })?
                .as_ref()
                .to_vec(),
        };
        let count = shape
            .checked_numel()
            .ok_or_else(|| err(field, "shape element count overflowed usize"))?;
        let size = if scalar == Scalar::I64 { 8 } else { 4 };
        let expected_len = count
            .checked_mul(size)
            .ok_or_else(|| err(field, "byte length overflowed usize"))?;
        if expected_len != bytes.len() {
            return Err(err(
                field,
                &format!(
                    "byte length mismatch: expected {expected_len}, got {}",
                    bytes.len()
                ),
            ));
        }
        Ok((shape, bytes, scalar, endian))
    }

    fn ints(bytes: &[u8], scalar: Scalar, endian: Endian, field: &str) -> Result<Vec<u32>> {
        let mut cursor = Cursor::new(bytes);
        let mut out = Vec::new();
        while (cursor.position() as usize) < bytes.len() {
            let value: i64 = match (scalar, endian) {
                (Scalar::I32, Endian::Little) => cursor.read_i32::<LittleEndian>().unwrap() as i64,
                (Scalar::I32, Endian::Big) => cursor.read_i32::<BigEndian>().unwrap() as i64,
                (Scalar::I32, Endian::Native) => cursor.read_i32::<NativeEndian>().unwrap() as i64,
                (_, Endian::Little) => cursor.read_i64::<LittleEndian>().unwrap(),
                (_, Endian::Big) => cursor.read_i64::<BigEndian>().unwrap(),
                (_, Endian::Native) => cursor.read_i64::<NativeEndian>().unwrap(),
            };
            out.push(u32::try_from(value).map_err(|_| {
                err(
                    field,
                    &format!("expected non-negative token id/rank that fits in u32, got {value}"),
                )
            })?);
        }
        Ok(out)
    }

    fn floats(bytes: &[u8], endian: Endian) -> Vec<f32> {
        let mut cursor = Cursor::new(bytes);
        let mut out = Vec::new();
        while (cursor.position() as usize) < bytes.len() {
            out.push(match endian {
                Endian::Little => cursor.read_f32::<LittleEndian>().unwrap(),
                Endian::Big => cursor.read_f32::<BigEndian>().unwrap(),
                Endian::Native => cursor.read_f32::<NativeEndian>().unwrap(),
            });
        }
        out
    }

    pub(super) fn resolve<F: AsRef<[u8]>>(
        ids: WireNdArray,
        lps: WireNdArray,
        ranks: WireNdArray,
        frames: &[F],
        prefix: &str,
    ) -> Result<Logprobs> {
        let f_ids = format!("{prefix}.logprob_token_ids");
        let (shape, bytes, scalar, endian) =
            metadata(ids, &f_ids, frames, &[Scalar::I32, Scalar::I64])?;
        if shape.len() != 2 {
            return Err(err(
                &f_ids,
                &format!("expected rank-2 array, got rank {}", shape.len()),
            ));
        }
        let (rows, cols) = (shape[0], shape[1]);
        let id_values = ints(&bytes, scalar, endian, &f_ids)?;
        let f_lps = format!("{prefix}.logprobs");
        let (lshape, lbytes, _, lendian) = metadata(lps, &f_lps, frames, &[Scalar::F32])?;
        if lshape.len() != 2 {
            return Err(err(
                &f_lps,
                &format!("expected rank-2 array, got rank {}", lshape.len()),
            ));
        }
        let lp_values = floats(&lbytes, lendian);
        let f_ranks = format!("{prefix}.token_ranks");
        let (rshape, rbytes, rscalar, rendian) =
            metadata(ranks, &f_ranks, frames, &[Scalar::I32, Scalar::I64])?;
        if rshape.len() != 1 {
            return Err(err(
                &f_ranks,
                &format!("expected rank-1 array, got rank {}", rshape.len()),
            ));
        }
        let rank_values = ints(&rbytes, rscalar, rendian, &f_ranks)?;
        if rows != lshape[0] || cols != lshape[1] {
            bail_ext_value_decode!(
                "{prefix}: row shape mismatch between token ids ({}, {}) and logprobs ({}, {})",
                rows,
                cols,
                lshape[0],
                lshape[1]
            );
        }
        if rows != rank_values.len() {
            bail_ext_value_decode!(
                "{prefix}: token_ranks length {} does not match row count {}",
                rank_values.len(),
                rows
            );
        }
        if rows == 0 {
            return Ok(Logprobs {
                positions: Vec::new(),
            });
        }
        if cols == 0 {
            bail_ext_value_decode!("{prefix}: zero-column logprobs payload with {} rows", rows);
        }
        let mut positions = Vec::new();
        for ((id_row, lp_row), sampled_rank) in
            id_values.chunks(cols).zip(lp_values.chunks(cols)).zip(rank_values)
        {
            if sampled_rank == 0 {
                bail_ext_value_decode!("token_ranks must be >= 1 for decoded engine-core logprobs");
            }
            positions.push(PositionLogprobs {
                entries: id_row
                    .iter()
                    .zip(lp_row)
                    .enumerate()
                    .map(|(index, (&token_id, &logprob))| TokenLogprob {
                        token_id,
                        logprob,
                        rank: if index == 0 {
                            sampled_rank
                        } else {
                            index as u32
                        },
                    })
                    .collect(),
            });
        }
        Ok(Logprobs { positions })
    }
}

#[test]
fn rewritten_decoder_matches_reference_on_random_payloads() {
    use super::WireLogprobs;
    use crate::protocol::tensor::{WireArrayData, WireNdArray};

    let state = std::cell::Cell::new(0x2545_f491_4f6c_dd1d_u64);
    let next = || {
        let mut s = state.get();
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        state.set(s);
        s
    };
    let int_dtypes = [
        "<i8", ">i8", "=i8", "i8", "int64", "<i4", ">i4", "|i4", "int32", "<f4", "<u8",
    ];
    let float_dtypes = ["<f4", ">f4", "=f4", "float32", "<i4"];
    let mut outcomes = (0, 0);
    for case in 0..20_000 {
        let rows = (next() % 5) as usize;
        let cols = (next() % 6) as usize;
        let id_dtype = int_dtypes[(next() % int_dtypes.len() as u64) as usize];
        let lp_dtype = float_dtypes[(next() % float_dtypes.len() as u64) as usize];
        let rank_dtype = int_dtypes[(next() % 7) as usize];
        let mut frames: Vec<Bytes> = vec![Bytes::from_static(b"header")];
        let mut array = |dtype: &str, shape: Vec<usize>, count: usize, size: usize, ids: bool| {
            // Occasionally corrupt the byte length.
            let count = if next() % 40 == 0 { count + 1 } else { count };
            let mut raw = Vec::with_capacity(count * size);
            for _ in 0..count {
                let value = next();
                let word: u64 = if ids {
                    // ids/ranks: mostly valid, sometimes negative, > u32 or 0.
                    match value % 25 {
                        0 => u64::MAX - (value >> 40),
                        1 => (u32::MAX as u64) + 1 + (value >> 50),
                        2 => 0,
                        _ => (value >> 20) % 200_000 + 1,
                    }
                } else {
                    value
                };
                raw.extend_from_slice(&word.to_le_bytes()[..size]);
            }
            let data = if next() % 2 == 0 {
                WireArrayData::RawView(Bytes::from(raw))
            } else {
                frames.push(Bytes::from(raw));
                // Occasionally point past the last frame.
                let skew = if next() % 30 == 0 { 9 } else { 0 };
                WireArrayData::AuxIndex(frames.len() - 1 + skew)
            };
            WireNdArray {
                dtype: dtype.to_string(),
                shape,
                data,
            }
        };
        let size_of = |dtype: &str| {
            if dtype.ends_with('8') || dtype == "int64" {
                8
            } else {
                4
            }
        };
        let id_shape = if next() % 30 == 0 {
            vec![rows * cols]
        } else {
            vec![rows, cols]
        };
        let lp_rows = if next() % 30 == 0 { rows + 1 } else { rows };
        let rank_len = if next() % 30 == 0 { rows + 1 } else { rows };
        let ids = array(id_dtype, id_shape, rows * cols, size_of(id_dtype), true);
        let lps = array(lp_dtype, vec![lp_rows, cols], lp_rows * cols, 4, false);
        let ranks = array(
            rank_dtype,
            vec![rank_len],
            rank_len,
            size_of(rank_dtype),
            true,
        );

        let expected = reference::resolve(
            ids.clone(),
            lps.clone(),
            ranks.clone(),
            &frames,
            "new_logprobs",
        );
        let actual = WireLogprobs {
            logprob_token_ids: ids,
            logprobs: lps,
            token_ranks: ranks,
            cu_num_generated_tokens: None,
            cu_num_generated_tokens_tensor: None,
        }
        .resolve(&frames, "new_logprobs");
        match (&expected, &actual) {
            (Ok(expected), Ok(actual)) => {
                outcomes.0 += 1;
                assert_eq!(
                    expected.positions.len(),
                    actual.positions.len(),
                    "case {case}"
                );
                for (e, a) in expected.positions.iter().zip(&actual.positions) {
                    assert_eq!(e.entries.len(), a.entries.len(), "case {case}");
                    for (e, a) in e.entries.iter().zip(&a.entries) {
                        assert_eq!(
                            (e.token_id, e.logprob.to_bits(), e.rank),
                            (a.token_id, a.logprob.to_bits(), a.rank),
                            "case {case}"
                        );
                    }
                }
            }
            (Err(expected), Err(actual)) => {
                outcomes.1 += 1;
                assert_eq!(expected.to_string(), actual.to_string(), "case {case}");
            }
            _ => panic!("case {case}: reference {expected:?} vs rewritten {actual:?}"),
        }
    }
    // Both successes and every kind of error are exercised substantially.
    assert!(outcomes.0 > 3000 && outcomes.1 > 3000, "{outcomes:?}");
}

/// Round 8: row-count and width mismatches between token ids, logprobs and
/// ranks (including counts *smaller* than the ranks count, which the
/// randomized test only perturbs upward) are decode errors with exactly the
/// base decoder's message, both directly and through the full wire path.
#[test]
fn mismatched_row_counts_and_widths_are_decode_errors_like_base() {
    use super::WireLogprobs;
    use crate::protocol::tensor::{WireArrayData, WireNdArray};

    let int_array = |shape: &[usize]| {
        let count: usize = shape.iter().product();
        let raw: Vec<u8> = (0..count as i64).flat_map(|v| (v + 1).to_le_bytes()).collect();
        WireNdArray {
            dtype: "<i8".to_string(),
            shape: shape.to_vec(),
            data: WireArrayData::RawView(Bytes::from(raw)),
        }
    };
    let float_array = |shape: &[usize]| {
        let count: usize = shape.iter().product();
        let raw: Vec<u8> = (0..count).flat_map(|v| (-(v as f32)).to_le_bytes()).collect();
        WireNdArray {
            dtype: "<f4".to_string(),
            shape: shape.to_vec(),
            data: WireArrayData::RawView(Bytes::from(raw)),
        }
    };
    let wire = |array: WireNdArray| {
        let WireArrayData::RawView(raw) = &array.data else {
            unreachable!()
        };
        ndarray_value(&array.dtype, &array.shape, Value::Ext(3, raw.to_vec()))
    };
    // (ids shape, logprobs shape, ranks len, expected message)
    let cases: [(&[usize], &[usize], usize, &str); 7] = [
        (
            &[1, 3],
            &[2, 3],
            2,
            "row shape mismatch between token ids (1, 3) and logprobs (2, 3)",
        ),
        (
            &[2, 3],
            &[1, 3],
            2,
            "row shape mismatch between token ids (2, 3) and logprobs (1, 3)",
        ),
        (
            &[1, 3],
            &[1, 3],
            2,
            "token_ranks length 2 does not match row count 1",
        ),
        (
            &[2, 3],
            &[2, 3],
            1,
            "token_ranks length 1 does not match row count 2",
        ),
        (
            &[2, 3],
            &[2, 2],
            2,
            "row shape mismatch between token ids (2, 3) and logprobs (2, 2)",
        ),
        (
            &[2, 2],
            &[2, 3],
            2,
            "row shape mismatch between token ids (2, 2) and logprobs (2, 3)",
        ),
        (
            &[0, 3],
            &[0, 3],
            1,
            "token_ranks length 1 does not match row count 0",
        ),
    ];
    let no_frames: Vec<Bytes> = Vec::new();
    for (ids, lps, ranks, reason) in cases {
        let expected = format!("new_logprobs: {reason}");
        let reference = reference::resolve(
            int_array(ids),
            float_array(lps),
            int_array(&[ranks]),
            &no_frames,
            "new_logprobs",
        )
        .expect_err("reference rejects");
        let crate::error::Error::ExtValueDecode { message } = &reference else {
            panic!("expected ExtValueDecode, got {reference:?}");
        };
        assert_eq!(message, &expected);
        let actual = WireLogprobs {
            logprob_token_ids: int_array(ids),
            logprobs: float_array(lps),
            token_ranks: int_array(&[ranks]),
            cu_num_generated_tokens: None,
            cu_num_generated_tokens_tensor: None,
        }
        .resolve(&no_frames, "new_logprobs")
        .expect_err("rewritten decoder rejects");
        assert_eq!(actual.to_string(), reference.to_string());

        // Full engine-core output decode path.
        let frames = vec![Bytes::from(encode_value(&output_wire_with_custom_fields(
            Some(Value::Array(vec![
                wire(int_array(ids)),
                wire(float_array(lps)),
                wire(int_array(&[ranks])),
                Value::Nil,
            ])),
            None,
        )))];
        let error = decode_engine_core_outputs(&frames).expect_err("decode rejects");
        let crate::error::Error::ExtValueDecode { message } = &error else {
            panic!("expected ExtValueDecode, got {error:?}");
        };
        assert_eq!(message, &expected);
    }
}
