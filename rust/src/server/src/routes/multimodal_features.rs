use std::collections::HashMap;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde::{Deserialize, Serialize};
use thiserror_ext::AsReport as _;
use vllm_engine_core_client::protocol::dtype::TensorDtype;
use vllm_engine_core_client::protocol::multimodal::{
    MmFeatureSpec, MmFeatures, MmField, MmKwargsItem, MmModality, PlaceholderRange,
};
use vllm_engine_core_client::protocol::tensor::{WireArrayData, WireTensor};

use crate::error::ApiError;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlaceholderRangeInfo {
    pub offset: usize,
    pub length: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub is_embed: Option<Vec<bool>>,
}

#[serde_with::skip_serializing_none]
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MultiModalFeatures {
    pub mm_hashes: HashMap<MmModality, Vec<String>>,
    pub mm_placeholders: HashMap<MmModality, Vec<PlaceholderRangeInfo>>,
    pub kwargs_data: Option<HashMap<MmModality, Vec<Option<String>>>>,
    pub mm_metadata: Option<HashMap<MmModality, Vec<Option<String>>>>,
}

impl MultiModalFeatures {
    pub fn from_features(features: MmFeatures) -> Result<Self, ApiError> {
        let mut kwargs_data = HashMap::new();
        let mut mm_metadata = HashMap::new();
        let mut result = Self {
            mm_hashes: HashMap::new(),
            mm_placeholders: HashMap::new(),
            kwargs_data: None,
            mm_metadata: None,
        };
        for feature in features {
            let is_embed = feature
                .mm_position
                .is_embed
                .map(|mask| {
                    mask.validate_inline()
                        .map_err(|error| ApiError::invalid_request(error, Some("features")))?;
                    if mask.dtype != TensorDtype::Bool || mask.shape != [feature.mm_position.length]
                    {
                        return Err(ApiError::invalid_request(
                            "is_embed must match the placeholder range",
                            Some("features"),
                        ));
                    }
                    match mask.data {
                        WireArrayData::RawView(bytes) => {
                            Ok(bytes.iter().map(|byte| *byte != 0).collect())
                        }
                        WireArrayData::AuxIndex(_) => unreachable!(),
                    }
                })
                .transpose()?;
            let modality = feature.modality;
            result.mm_hashes.entry(modality).or_default().push(feature.identifier);
            result.mm_placeholders.entry(modality).or_default().push(PlaceholderRangeInfo {
                offset: feature.mm_position.offset,
                length: feature.mm_position.length,
                is_embed,
            });
            let metadata: MmKwargsItem = feature
                .data
                .as_ref()
                .into_iter()
                .flat_map(|item| item.iter())
                .filter(|(_, elem)| match &elem.field {
                    MmField::Batched(field) => field.keep_on_cpu,
                    MmField::Flat(field) => field.keep_on_cpu,
                    MmField::Shared(field) => field.keep_on_cpu,
                })
                .map(|(key, elem)| (key.clone(), elem.clone()))
                .collect();
            kwargs_data
                .entry(modality)
                .or_insert_with(Vec::new)
                .push(feature.data.as_ref().map(encode_kwargs).transpose()?);
            mm_metadata
                .entry(modality)
                .or_insert_with(Vec::new)
                .push((!metadata.is_empty()).then(|| encode_kwargs(&metadata)).transpose()?);
        }
        result.kwargs_data = Some(kwargs_data);
        if mm_metadata.values().flatten().any(Option::is_some) {
            result.mm_metadata = Some(mm_metadata);
        }
        Ok(result)
    }

    pub fn into_features(self, has_ec_transfer: bool) -> Result<MmFeatures, ApiError> {
        let mut features = Vec::new();
        if self.mm_hashes.len() != self.mm_placeholders.len()
            || self
                .kwargs_data
                .as_ref()
                .is_some_and(|items| items.len() != self.mm_hashes.len())
            || self
                .mm_metadata
                .as_ref()
                .is_some_and(|items| items.len() != self.mm_hashes.len())
        {
            return Err(ApiError::invalid_request(
                "multimodal fields must use the same modalities",
                Some("features"),
            ));
        }
        for (modality, hashes) in self.mm_hashes {
            let positions = self.mm_placeholders.get(&modality).ok_or_else(|| {
                ApiError::invalid_request("missing multimodal placeholders", Some("features"))
            })?;
            let kwargs = self.kwargs_data.as_ref().and_then(|items| items.get(&modality));
            let metadata = self.mm_metadata.as_ref().and_then(|items| items.get(&modality));
            if positions.len() != hashes.len()
                || kwargs.is_some_and(|items| items.len() != hashes.len())
                || metadata.is_some_and(|items| items.len() != hashes.len())
                || (self.kwargs_data.is_some() && kwargs.is_none())
                || (self.mm_metadata.is_some() && metadata.is_none())
            {
                return Err(ApiError::invalid_request(
                    "multimodal fields must have the same item counts",
                    Some("features"),
                ));
            }
            for (index, hash) in hashes.into_iter().enumerate() {
                let raw = kwargs.and_then(|items| items[index].as_deref());
                let extra = metadata.and_then(|items| items[index].as_deref());
                if raw.is_none() && extra.is_some() && !has_ec_transfer {
                    return Err(ApiError::invalid_request(
                        "metadata-only multimodal items require ec_transfer_params",
                        Some("features"),
                    ));
                }
                let decode = |data: &str| -> Result<MmKwargsItem, ApiError> {
                    let bytes = STANDARD.decode(data).map_err(|error| {
                        ApiError::invalid_request(error.to_report_string(), Some("features"))
                    })?;
                    rmp_serde::from_slice(&bytes).map_err(|error| {
                        ApiError::invalid_request(error.to_report_string(), Some("features"))
                    })
                };
                let mut data = raw.map(decode).transpose()?;
                if let Some(extra) = extra {
                    data.get_or_insert_with(Default::default).extend(decode(extra)?);
                }
                let position = &positions[index];
                features.push(MmFeatureSpec {
                    data,
                    modality,
                    identifier: hash.clone(),
                    mm_hash: Some(hash),
                    mm_position: PlaceholderRange {
                        offset: position.offset,
                        length: position.length,
                        is_embed: position
                            .is_embed
                            .as_ref()
                            .map(|mask| {
                                WireTensor::from_bool(vec![mask.len()], mask.clone()).map_err(
                                    |error| ApiError::invalid_request(error, Some("features")),
                                )
                            })
                            .transpose()?,
                    },
                });
            }
        }
        Ok(features)
    }
}

fn encode_kwargs(item: &MmKwargsItem) -> Result<String, ApiError> {
    rmp_serde::to_vec_named(item)
        .map(|bytes| STANDARD.encode(bytes))
        .map_err(|error| ApiError::invalid_request(error.to_report_string(), Some("features")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use vllm_engine_core_client::protocol::multimodal::{
        MmBatchedField, MmFieldElem, MmKwargValue,
    };

    #[test]
    fn render_features_round_trip_tensors_masks_and_cpu_metadata() {
        let feature = MmFeatureSpec {
            data: Some(MmKwargsItem::from([
                (
                    "pixels".into(),
                    MmFieldElem {
                        data: Some(MmKwargValue::Tensor(
                            WireTensor::from_f32(vec![2], vec![1., 2.]).unwrap(),
                        )),
                        field: MmField::Batched(MmBatchedField { keep_on_cpu: false }),
                    },
                ),
                (
                    "grid".into(),
                    MmFieldElem {
                        data: Some(MmKwargValue::Int(2)),
                        field: MmField::Batched(MmBatchedField { keep_on_cpu: true }),
                    },
                ),
            ])),
            modality: MmModality::Image,
            identifier: "image-hash".into(),
            mm_hash: Some("image-hash".into()),
            mm_position: PlaceholderRange {
                offset: 3,
                length: 2,
                is_embed: Some(WireTensor::from_bool(vec![2], vec![true, false]).unwrap()),
            },
        };
        let payload = MultiModalFeatures::from_features(vec![feature.clone()]).unwrap();
        assert_eq!(
            payload.clone().into_features(false).unwrap(),
            vec![feature.clone()]
        );
        let payload = MultiModalFeatures {
            kwargs_data: None,
            ..payload
        };
        assert!(payload.clone().into_features(false).is_err());
        let metadata = payload.into_features(true).unwrap();
        assert_eq!(metadata[0].mm_position, feature.mm_position);
        assert_eq!(
            metadata[0].data.as_ref().unwrap().keys().collect::<Vec<_>>(),
            vec!["grid"]
        );
    }

    #[test]
    fn render_features_preserve_explicit_cache_identity() {
        let feature = MmFeatureSpec {
            data: None,
            modality: MmModality::Image,
            identifier: "user-image-uuid".into(),
            mm_hash: Some("processor-content-hash".into()),
            mm_position: PlaceholderRange {
                offset: 0,
                length: 2,
                is_embed: None,
            },
        };
        let payload = MultiModalFeatures::from_features(vec![feature]).unwrap();
        let features = payload.into_features(false).unwrap();
        assert_eq!(features[0].identifier, "user-image-uuid");
    }
}
