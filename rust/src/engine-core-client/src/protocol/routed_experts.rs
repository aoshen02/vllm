// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Per-output expert routing decisions (`EngineCoreOutput.routed_experts`).
//!
//! Python engine-core attaches a numpy array of shape
//! `[rows, num_layers, top_k]` (dtype `uint8` when the model has at most 256
//! experts, else `uint16`) when `enable_return_routed_experts` is on: the
//! prompt block on the first output, then one row per generated token.

use bytes::Bytes;
use enum_as_inner::EnumAsInner;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use rmpv::Value;

use crate::error::Result;
use crate::protocol::tensor::{CUSTOM_TYPE_RAW_VIEW, ShapeExt as _, WireArrayData, WireNdArray};

/// Decoded routed experts of one engine output: the raw C-order bytes of an
/// integer array, in the byte order of `dtype` as sent by the engine.
#[derive(Debug, Clone, PartialEq)]
pub struct RoutedExperts {
    /// numpy dtype string as sent by the engine (`dtype.str`), e.g. `"|u1"`
    /// or `"<u2"`.
    pub dtype: String,
    /// Array shape; the first axis counts rows (tokens).
    pub shape: Vec<usize>,
    /// Raw element bytes, `shape.product() * itemsize` long.
    pub data: Bytes,
}

impl RoutedExperts {
    /// Size in bytes of one element of `dtype`, for the integer dtypes the
    /// engine emits (`|u1`, `|i1`, `<u2`, `<i2`, `<u4`, `<i4`, `<u8`, `<i8`
    /// and their big-endian forms); `None` otherwise.
    pub fn itemsize(dtype: &str) -> Option<usize> {
        match dtype {
            "|u1" | "|i1" => Some(1),
            "<u2" | "<i2" | ">u2" | ">i2" => Some(2),
            "<u4" | "<i4" | ">u4" | ">i4" => Some(4),
            "<u8" | "<i8" | ">u8" | ">i8" => Some(8),
            _ => None,
        }
    }

    /// Number of rows (first axis).
    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }
}

/// Wire form until aux frames are resolved, then the decoded value.
///
/// A malformed value (not an ndarray, unsupported dtype, length mismatch,
/// missing aux frame) never fails the decode of the whole output batch:
/// it becomes [`Self::Invalid`] and only fails its own request.
#[derive(Debug, Clone, PartialEq, EnumAsInner)]
pub enum MaybeWireRoutedExperts {
    /// Still referencing an inline raw view or an aux frame.
    Wire(Box<WireNdArray>),
    /// Resolved array.
    Direct(RoutedExperts),
    /// A value that is not a supported routed-experts array (the reason).
    Invalid(String),
}

impl<'de> Deserialize<'de> for MaybeWireRoutedExperts {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Ok(Self::from_value(Value::deserialize(deserializer)?))
    }
}

/// msgpack type name for error messages (never the value: it can be huge).
fn kind(value: &Value) -> &'static str {
    match value {
        Value::Nil => "nil",
        Value::Boolean(_) => "bool",
        Value::Integer(_) => "integer",
        Value::F32(_) | Value::F64(_) => "float",
        Value::String(_) => "string",
        Value::Binary(_) => "binary",
        Value::Array(_) => "array",
        Value::Map(_) => "map",
        Value::Ext(..) => "ext",
    }
}

impl MaybeWireRoutedExperts {
    /// Parse the ndarray tuple `(dtype, shape, data)`; anything else is
    /// [`Self::Invalid`].
    fn from_value(value: Value) -> Self {
        let invalid = |reason: String| Self::Invalid(format!("routed_experts: {reason}"));
        let Value::Array(items) = value else {
            return invalid(format!("expected an ndarray tuple, got {}", kind(&value)));
        };
        let Ok([dtype, shape, data]) = <[Value; 3]>::try_from(items) else {
            return invalid("expected an ndarray tuple (dtype, shape, data)".to_string());
        };
        let Value::String(dtype) = dtype else {
            return invalid(format!("expected a dtype string, got {}", kind(&dtype)));
        };
        let Some(dtype) = dtype.into_str() else {
            return invalid("dtype is not valid UTF-8".to_string());
        };
        let Value::Array(axes) = shape else {
            return invalid(format!("expected a shape array, got {}", kind(&shape)));
        };
        let Some(shape) = axes
            .iter()
            .map(|axis| axis.as_u64().and_then(|axis| usize::try_from(axis).ok()))
            .collect::<Option<Vec<usize>>>()
        else {
            return invalid("shape axes must be non-negative integers".to_string());
        };
        let data = match data {
            Value::Ext(tag, bytes) if tag == CUSTOM_TYPE_RAW_VIEW => {
                WireArrayData::RawView(Bytes::from(bytes))
            }
            Value::Integer(index) => match index.as_u64().and_then(|i| usize::try_from(i).ok()) {
                Some(index) => WireArrayData::AuxIndex(index),
                None => return invalid("aux frame index must be non-negative".to_string()),
            },
            other => {
                return invalid(format!(
                    "expected raw-view ext or aux frame index, got {}",
                    kind(&other)
                ));
            }
        };
        Self::Wire(Box::new(WireNdArray { dtype, shape, data }))
    }
}

impl Serialize for MaybeWireRoutedExperts {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Wire(value) => value.serialize(serializer),
            Self::Invalid(_) => serializer.serialize_unit(),
            Self::Direct(value) => WireNdArray {
                dtype: value.dtype.clone(),
                shape: value.shape.clone(),
                data: WireArrayData::RawView(value.data.clone()),
            }
            .serialize(serializer),
        }
    }
}

impl MaybeWireRoutedExperts {
    /// Resolve aux-frame references and validate dtype, shape and length.
    /// Never fails the batch: a bad value becomes [`Self::Invalid`].
    pub(super) fn resolve<Frame>(self, frames: &[Frame]) -> Result<Self>
    where
        Frame: AsRef<[u8]>,
    {
        let wire = match self {
            Self::Wire(wire) => *wire,
            resolved => return Ok(resolved),
        };
        let invalid = |reason: String| Ok(Self::Invalid(format!("routed_experts: {reason}")));
        let WireNdArray { dtype, shape, data } = wire;
        let Some(itemsize) = RoutedExperts::itemsize(&dtype) else {
            return invalid(format!("unsupported dtype {dtype:?}"));
        };
        if shape.is_empty() {
            return invalid("expected an array with at least one axis".to_string());
        }
        let data = match data {
            WireArrayData::RawView(bytes) => bytes,
            WireArrayData::AuxIndex(index) => match frames.get(index) {
                Some(frame) => Bytes::copy_from_slice(frame.as_ref()),
                None => {
                    return invalid(format!(
                        "aux frame index {index} out of range for {} frames",
                        frames.len()
                    ));
                }
            },
        };
        let expected = shape.checked_numel().and_then(|count| count.checked_mul(itemsize));
        if expected != Some(data.len()) {
            return invalid(format!(
                "byte length {} does not match {} axes of dtype {dtype}",
                data.len(),
                shape.len()
            ));
        }
        Ok(Self::Direct(RoutedExperts { dtype, shape, data }))
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use rmpv::Value;

    use super::*;
    use crate::protocol::output::{EngineCoreOutputs, decode_engine_core_outputs};

    /// One-output message whose `routed_experts` (tuple field 12) is `array`.
    fn message(array: Value, aux: Vec<Bytes>) -> Vec<Bytes> {
        let mut output = vec![Value::from("req-1"), Value::Array(vec![Value::from(7)])];
        output.extend(std::iter::repeat_n(Value::Nil, 10));
        output.push(array);
        let header = Value::Array(vec![
            Value::from(0),
            Value::Array(vec![Value::Array(output)]),
            Value::Nil,
            Value::from(0.0),
        ]);
        let mut first = Vec::new();
        rmpv::encode::write_value(&mut first, &header).unwrap();
        std::iter::once(Bytes::from(first)).chain(aux).collect()
    }

    fn ndarray(dtype: &str, shape: &[usize], data: Value) -> Value {
        Value::Array(vec![
            Value::from(dtype),
            Value::Array(shape.iter().copied().map(Value::from).collect()),
            data,
        ])
    }

    fn decode_value(frames: &[Bytes]) -> Option<MaybeWireRoutedExperts> {
        let EngineCoreOutputs::RequestBatch(batch) =
            decode_engine_core_outputs(frames).expect("batch decodes")
        else {
            panic!("expected a request batch");
        };
        batch.outputs.into_iter().next().unwrap().routed_experts
    }

    fn decode(frames: &[Bytes]) -> crate::error::Result<Option<RoutedExperts>> {
        Ok(decode_value(frames).map(|value| value.into_direct().unwrap()))
    }

    #[test]
    fn decodes_aux_frame_and_inline_routed_experts() {
        let rows: Vec<u8> = (0..2 * 3 * 4).collect();
        let aux = message(
            ndarray("|u1", &[2, 3, 4], Value::from(1)),
            vec![Bytes::from(rows.clone())],
        );
        let decoded = decode(&aux).unwrap().unwrap();
        assert_eq!(decoded.dtype, "|u1");
        assert_eq!(decoded.shape, vec![2, 3, 4]);
        assert_eq!(decoded.data.as_ref(), rows.as_slice());

        let wide: Vec<u8> = (0..2 * 2 * 2).collect();
        let inline = message(
            ndarray("<u2", &[1, 2, 2], Value::Ext(3, wide.clone())),
            vec![],
        );
        let decoded = decode(&inline).unwrap().unwrap();
        assert_eq!(decoded.dtype, "<u2");
        assert_eq!(decoded.data.as_ref(), wide.as_slice());

        assert!(decode(&message(Value::Nil, vec![])).unwrap().is_none());
    }

    /// A bad value only marks its own output invalid; the batch decodes.
    #[test]
    fn bad_routed_experts_payloads_are_request_local() {
        // Wrong byte length, unsupported dtypes, missing aux frame, no axes,
        // not an ndarray at all, bad shape and data fields.
        let cases = [
            message(
                ndarray("<f2", &[1, 1, 1], Value::Ext(3, vec![0; 2])),
                vec![],
            ),
            message(ndarray("|b1", &[1, 1, 1], Value::Ext(3, vec![1])), vec![]),
            message(Value::from("not an array"), vec![]),
            message(Value::Array(vec![Value::from("|u1")]), vec![]),
            message(
                Value::Array(vec![
                    Value::from("|u1"),
                    Value::from(3),
                    Value::Ext(3, vec![0]),
                ]),
                vec![],
            ),
            message(
                Value::Array(vec![
                    Value::from("|u1"),
                    Value::Array(vec![Value::from(-1)]),
                    Value::Ext(3, vec![0]),
                ]),
                vec![],
            ),
            message(ndarray("|u1", &[1], Value::Ext(7, vec![0])), vec![]),
            message(ndarray("|u1", &[1], Value::from("x")), vec![]),
            message(
                ndarray("|u1", &[2, 3, 4], Value::Ext(3, vec![0; 23])),
                vec![],
            ),
            message(
                ndarray("<f4", &[1, 1, 1], Value::Ext(3, vec![0; 4])),
                vec![],
            ),
            message(ndarray("|u1", &[1, 2, 2], Value::from(3)), vec![]),
            message(ndarray("|u1", &[], Value::Ext(3, vec![0])), vec![]),
        ];
        for frames in cases {
            let value = decode_value(&frames).expect("routed_experts present");
            let reason = value.into_invalid().expect("invalid value");
            assert!(reason.starts_with("routed_experts: "), "{reason}");
        }
    }
}
