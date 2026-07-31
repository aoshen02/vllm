// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use enum_as_inner::EnumAsInner;
use serde::{Deserialize, Deserializer, Serialize};
use serde_tuple::{Deserialize_tuple, Serialize_tuple};

use crate::error::{Error, Result, bail_ext_value_decode};
use crate::protocol::logprobs::array::{decode_array1_u32, decode_array1_u64};
use crate::protocol::tensor::WireNdArray;

/// Token IDs in the sampling distribution for each generated token.
///
/// `offsets` uses CSR layout: candidate IDs for generated token `i` are in
/// `token_ids[offsets[i]..offsets[i + 1]]`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SamplingMask {
    pub token_ids: Vec<u32>,
    pub offsets: Vec<u64>,
}

/// Output wrapper that starts in the Python wire format and is resolved into
/// [`SamplingMask`] before callers receive the engine-core output.
#[derive(Clone, PartialEq, Debug, EnumAsInner)]
pub enum MaybeWireSamplingMask {
    Wire(Box<WireSamplingMask>),
    Direct(SamplingMask),
}

impl<'de> Deserialize<'de> for MaybeWireSamplingMask {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        WireSamplingMask::deserialize(deserializer).map(|value| Self::Wire(Box::new(value)))
    }
}

impl Serialize for MaybeWireSamplingMask {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Wire(value) => value.serialize(serializer),
            Self::Direct(_) => Err(serde::ser::Error::custom(
                "resolved sampling masks cannot be serialized",
            )),
        }
    }
}

impl MaybeWireSamplingMask {
    pub(super) fn resolve<Frame>(self, frames: &[Frame], field_prefix: &str) -> Result<Self>
    where
        Frame: AsRef<[u8]>,
    {
        match self {
            Self::Direct(value) => Ok(Self::Direct(value)),
            Self::Wire(value) => value.resolve(frames, field_prefix).map(Self::Direct),
        }
    }
}

/// Python wire representation of `SamplingMaskLists` before ndarray payloads
/// are resolved.
#[derive(Debug, Clone, PartialEq, Serialize_tuple, Deserialize_tuple)]
pub struct WireSamplingMask {
    pub token_ids: WireNdArray,
    pub offsets: WireNdArray,
    #[serde(default)]
    pub cu_num_generated_tokens: Option<Vec<usize>>,
}

impl WireSamplingMask {
    fn resolve<Frame>(self, frames: &[Frame], field_prefix: &str) -> Result<SamplingMask>
    where
        Frame: AsRef<[u8]>,
    {
        if let Some(indices) = self.cu_num_generated_tokens {
            bail_ext_value_decode!(
                "{field_prefix}.cu_num_generated_tokens: \
                 expected None for per-request engine-core sampling mask, got {indices:?}"
            );
        }

        let value = SamplingMask {
            token_ids: decode_array1_u32(
                self.token_ids,
                &format!("{field_prefix}.token_ids"),
                frames,
            )?,
            offsets: decode_array1_u64(self.offsets, &format!("{field_prefix}.offsets"), frames)?,
        };
        validate(&value, field_prefix)?;
        Ok(value)
    }
}

pub(crate) fn validate(value: &SamplingMask, field_prefix: &str) -> Result<()> {
    let Some((&first, rest)) = value.offsets.split_first() else {
        bail_ext_value_decode!("{field_prefix}.offsets: expected at least one offset");
    };
    if first != 0 {
        bail_ext_value_decode!("{field_prefix}.offsets: expected first offset 0, got {first}");
    }
    if rest.windows(2).any(|window| window[1] < window[0]) {
        bail_ext_value_decode!("{field_prefix}.offsets: offsets must be non-decreasing");
    }
    let expected_last = u64::try_from(value.token_ids.len()).expect("usize always fits in u64");
    if value.offsets.last().copied() != Some(expected_last) {
        bail_ext_value_decode!(
            "{field_prefix}.offsets: expected final offset {expected_last}, got {:?}",
            value.offsets.last()
        );
    }
    Ok(())
}
