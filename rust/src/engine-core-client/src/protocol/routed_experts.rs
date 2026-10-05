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

use crate::error::{Error, Result, bail_ext_value_decode};
use crate::protocol::tensor::{ShapeExt as _, WireArrayData, WireNdArray};

/// Decoded routed experts of one engine output: the raw C-order bytes of a
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
#[derive(Debug, Clone, PartialEq, EnumAsInner)]
pub enum MaybeWireRoutedExperts {
    /// Still referencing an inline raw view or an aux frame.
    Wire(Box<WireNdArray>),
    /// Resolved array.
    Direct(RoutedExperts),
}

impl<'de> Deserialize<'de> for MaybeWireRoutedExperts {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        WireNdArray::deserialize(deserializer).map(|value| Self::Wire(Box::new(value)))
    }
}

impl Serialize for MaybeWireRoutedExperts {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Wire(value) => value.serialize(serializer),
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
    pub(super) fn resolve<Frame>(self, frames: &[Frame]) -> Result<Self>
    where
        Frame: AsRef<[u8]>,
    {
        let wire = match self {
            Self::Direct(value) => return Ok(Self::Direct(value)),
            Self::Wire(wire) => *wire,
        };
        let WireNdArray { dtype, shape, data } = wire;
        let Some(itemsize) = RoutedExperts::itemsize(&dtype) else {
            bail_ext_value_decode!("routed_experts: unsupported dtype {dtype:?}");
        };
        if shape.is_empty() {
            bail_ext_value_decode!("routed_experts: expected an array with at least one axis");
        }
        let data = match data {
            WireArrayData::RawView(bytes) => bytes,
            WireArrayData::AuxIndex(index) => match frames.get(index) {
                Some(frame) => Bytes::copy_from_slice(frame.as_ref()),
                None => bail_ext_value_decode!(
                    "routed_experts: aux frame index {index} out of range for {} frames",
                    frames.len()
                ),
            },
        };
        let expected = shape.checked_numel().and_then(|count| count.checked_mul(itemsize));
        if expected != Some(data.len()) {
            bail_ext_value_decode!(
                "routed_experts: byte length mismatch for shape {shape:?} dtype {dtype}: got {}",
                data.len()
            );
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

    fn decode(frames: &[Bytes]) -> crate::error::Result<Option<RoutedExperts>> {
        let EngineCoreOutputs::RequestBatch(batch) = decode_engine_core_outputs(frames)? else {
            panic!("expected a request batch");
        };
        Ok(batch
            .outputs
            .into_iter()
            .next()
            .unwrap()
            .routed_experts
            .map(|value| value.into_direct().unwrap()))
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

    #[test]
    fn rejects_bad_routed_experts_payloads() {
        // Wrong byte length, unsupported dtype, missing aux frame, no axes.
        let cases = [
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
            assert!(decode(&frames).is_err());
        }
    }
}
