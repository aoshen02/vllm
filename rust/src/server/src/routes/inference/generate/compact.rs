// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Opt-in `logprobs_format: "compact"` encoding for raw generate output
//! logprobs.
//!
//! The compact block carries the engine `LogprobsLists` rows unchanged: for
//! each scored position, slot 0 is the sampled token and slots `1..S` are the
//! engine top-k candidates in engine order. Token ids and ranks are
//! little-endian `int32`, logprobs are the raw engine `float32` bits (no
//! `-9999` clamp; non-finite values preserved bit-for-bit). Each array is
//! standard-alphabet, padded base64.
//!
//! Encoding is incremental: every engine step is packed and base64-encoded as
//! it arrives, so a request that is aborted after a long generation only has
//! to concatenate already-encoded segments when it renders its response.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use serde::Serialize;
use vllm_llm::{Logprobs, LogprobsAccumulator};

/// Wire value of the request `logprobs_format` field.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) enum LogprobsFormat {
    /// Today's OpenAI chat-style `logprobs.content` list (the default).
    #[default]
    OpenAi,
    /// Packed base64 arrays in `choices[i].compact_logprobs`.
    Compact,
}

impl LogprobsFormat {
    /// Parse the request field (`None` = absent); returns `None` for an
    /// unsupported value, including an explicit `null` or a non-string
    /// (matching the Python frontend's `Literal["openai", "compact"]`).
    pub(crate) fn parse(value: Option<&serde_json::Value>) -> Option<Self> {
        match value {
            None => Some(Self::OpenAi),
            Some(serde_json::Value::String(value)) => match value.as_str() {
                "openai" => Some(Self::OpenAi),
                "compact" => Some(Self::Compact),
                _ => None,
            },
            Some(_) => None,
        }
    }
}

/// Encoded segments are closed once they reach roughly this many bytes. Large
/// engine steps produce one exactly-sized segment each.
const SEGMENT_TARGET_BYTES: usize = 1 << 20;

/// Incremental padded base64 encoder whose output is kept as immutable
/// segments that can be handed to the HTTP body without copying.
#[derive(Debug, Default)]
struct Base64Segments {
    segments: Vec<Bytes>,
    current: String,
    carry: [u8; 2],
    carry_len: usize,
    encoded_len: usize,
}

impl Base64Segments {
    fn push(&mut self, mut data: &[u8]) {
        if self.carry_len > 0 {
            let need = 3 - self.carry_len;
            if data.len() < need {
                self.carry[self.carry_len..self.carry_len + data.len()].copy_from_slice(data);
                self.carry_len += data.len();
                return;
            }
            let mut group = [0_u8; 3];
            group[..self.carry_len].copy_from_slice(&self.carry[..self.carry_len]);
            group[self.carry_len..].copy_from_slice(&data[..need]);
            self.carry_len = 0;
            self.encode(&group);
            data = &data[need..];
        }
        let whole = data.len() - data.len() % 3;
        self.encode(&data[..whole]);
        let rest = &data[whole..];
        self.carry[..rest.len()].copy_from_slice(rest);
        self.carry_len = rest.len();
    }

    /// Encode `data` and append it. Only the final call (from `finish`) may
    /// pass a length that is not a multiple of three.
    fn encode(&mut self, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let encoded = data.len().div_ceil(3) * 4;
        if !self.current.is_empty() && self.current.len() + encoded > SEGMENT_TARGET_BYTES {
            self.close_segment();
        }
        if self.current.is_empty() {
            self.current.reserve_exact(encoded);
        }
        STANDARD.encode_string(data, &mut self.current);
        self.encoded_len += encoded;
    }

    fn close_segment(&mut self) {
        let mut current = std::mem::take(&mut self.current);
        if current.capacity() > current.len() + current.len() / 4 {
            current.shrink_to_fit();
        }
        self.segments.push(Bytes::from(current));
    }

    fn finish(mut self) -> EncodedArray {
        if self.carry_len > 0 {
            let carry = self.carry;
            let len = self.carry_len;
            self.carry_len = 0;
            self.encode(&carry[..len]);
        }
        if !self.current.is_empty() {
            self.close_segment();
        }
        EncodedArray {
            segments: self.segments,
            len: self.encoded_len,
        }
    }
}

/// One base64-encoded array split into immutable segments.
#[derive(Debug, Clone, Default)]
pub(crate) struct EncodedArray {
    pub segments: Vec<Bytes>,
    /// Total encoded length in bytes.
    pub len: usize,
}

impl EncodedArray {
    /// Concatenate all segments into one string.
    pub(crate) fn to_base64_string(&self) -> String {
        let mut out = String::with_capacity(self.len);
        for segment in &self.segments {
            // Segments are produced by the base64 encoder and are ASCII.
            out.push_str(std::str::from_utf8(segment).expect("base64 segment is ASCII"));
        }
        out
    }
}

/// Accumulates sample logprobs directly into the compact wire encoding.
#[derive(Debug, Default)]
pub(crate) struct CompactLogprobsAccumulator {
    /// The request's row width `k + 1` (`0` when unknown, i.e. `logprobs`
    /// absent or `-1`). When known it is always the reported `num_slots`:
    /// wider engine rows (padded to the batch-wide max top-k) are truncated
    /// to it, like the Python frontend, and narrower rows are an error.
    requested_slots: usize,
    num_slots: Option<usize>,
    num_positions: usize,
    /// Whether the engine attached a logprobs payload to any step.
    saw_payload: bool,
    token_ids: Base64Segments,
    logprobs: Base64Segments,
    ranks: Base64Segments,
    scratch_token_ids: Vec<u8>,
    scratch_logprobs: Vec<u8>,
    scratch_ranks: Vec<u8>,
    error: Option<String>,
}

impl CompactLogprobsAccumulator {
    pub(crate) fn new(requested_slots: usize) -> Self {
        Self {
            requested_slots,
            ..Default::default()
        }
    }

    /// Whether any step carried a logprobs payload.
    pub(crate) fn saw_payload(&self) -> bool {
        self.saw_payload
    }

    fn fail(&mut self, message: String) {
        if self.error.is_none() {
            self.error = Some(message);
        }
    }

    /// Finish encoding and return the compact block.
    pub(crate) fn finish(self) -> Result<CompactLogprobs, String> {
        if let Some(error) = self.error {
            return Err(error);
        }
        Ok(CompactLogprobs {
            num_positions: self.num_positions,
            num_slots: self.num_slots.unwrap_or(self.requested_slots),
            token_ids: self.token_ids.finish(),
            logprobs: self.logprobs.finish(),
            ranks: self.ranks.finish(),
        })
    }
}

impl LogprobsAccumulator for CompactLogprobsAccumulator {
    fn extend(&mut self, step: Logprobs) {
        self.saw_payload = true;
        if self.error.is_some() || step.positions.is_empty() {
            return;
        }
        // S = k + 1 is fixed by the request when known; only an unknown k
        // (`-1`) takes the engine row width.
        let width = step.positions[0].entries.len();
        let slots = *self.num_slots.get_or_insert(match self.requested_slots {
            0 => width,
            requested => requested,
        });
        if slots == 0 {
            self.fail("raw generate logprobs position unexpectedly had no token candidates".into());
            return;
        }

        let rows = step.positions.len();
        self.scratch_token_ids.clear();
        self.scratch_logprobs.clear();
        self.scratch_ranks.clear();
        self.scratch_token_ids.reserve(rows * slots * 4);
        self.scratch_logprobs.reserve(rows * slots * 4);
        self.scratch_ranks.reserve(rows * 4);

        for position in &step.positions {
            if position.entries.len() < slots {
                self.fail(format!(
                    "raw generate logprobs row has {} candidates, expected at least {slots}",
                    position.entries.len()
                ));
                return;
            }
            let sampled_rank = position.entries[0].rank;
            if sampled_rank > i32::MAX as u32 {
                self.fail(format!("sampled rank {sampled_rank} does not fit int32"));
                return;
            }
            self.scratch_ranks.extend_from_slice(&sampled_rank.to_le_bytes());
            for entry in &position.entries[..slots] {
                if entry.token_id > i32::MAX as u32 {
                    self.fail(format!("token id {} does not fit int32", entry.token_id));
                    return;
                }
                self.scratch_token_ids.extend_from_slice(&entry.token_id.to_le_bytes());
                self.scratch_logprobs.extend_from_slice(&entry.logprob.to_bits().to_le_bytes());
            }
        }

        self.token_ids.push(&self.scratch_token_ids);
        self.logprobs.push(&self.scratch_logprobs);
        self.ranks.push(&self.scratch_ranks);
        self.num_positions += rows;
    }

    fn num_positions(&self) -> usize {
        self.num_positions
    }
}

/// Finished compact block for one choice.
#[derive(Debug, Clone)]
pub(crate) struct CompactLogprobs {
    pub num_positions: usize,
    pub num_slots: usize,
    pub token_ids: EncodedArray,
    pub logprobs: EncodedArray,
    pub ranks: EncodedArray,
}

pub(crate) const DTYPE_TOKEN_IDS: &str = "int32";
pub(crate) const DTYPE_LOGPROBS: &str = "float32";
pub(crate) const BYTEORDER: &str = "little";

/// Owned, serde-serializable form of [`CompactLogprobs`] for small payloads
/// (streaming chunks). Field order matches the non-streaming renderer.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct CompactLogprobsJson {
    pub num_positions: usize,
    pub num_slots: usize,
    pub dtype_token_ids: &'static str,
    pub dtype_logprobs: &'static str,
    pub byteorder: &'static str,
    pub token_ids: String,
    pub logprobs: String,
    pub ranks: String,
}

impl From<&CompactLogprobs> for CompactLogprobsJson {
    fn from(value: &CompactLogprobs) -> Self {
        Self {
            num_positions: value.num_positions,
            num_slots: value.num_slots,
            dtype_token_ids: DTYPE_TOKEN_IDS,
            dtype_logprobs: DTYPE_LOGPROBS,
            byteorder: BYTEORDER,
            token_ids: value.token_ids.to_base64_string(),
            logprobs: value.logprobs.to_base64_string(),
            ranks: value.ranks.to_base64_string(),
        }
    }
}

/// Encode one step's logprobs (e.g. one streaming chunk) as a compact block.
pub(crate) fn encode_compact(
    logprobs: Logprobs,
    requested_slots: usize,
) -> Result<CompactLogprobsJson, String> {
    let mut accumulator = CompactLogprobsAccumulator::new(requested_slots);
    accumulator.extend(logprobs);
    accumulator.finish().map(|block| CompactLogprobsJson::from(&block))
}

#[cfg(test)]
pub(crate) mod tests {
    use vllm_llm::{PositionLogprobs, TokenLogprob};

    use super::*;

    /// Decoded compact arrays: (token_ids, logprob bits, ranks).
    pub(crate) fn decode_compact(block: &serde_json::Value) -> (Vec<i32>, Vec<u32>, Vec<i32>) {
        let decode = |key: &str| {
            STANDARD
                .decode(block[key].as_str().expect("base64 string"))
                .expect("valid base64")
        };
        let ints = |bytes: Vec<u8>| {
            assert_eq!(bytes.len() % 4, 0);
            bytes
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let bits = decode("logprobs")
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (ints(decode("token_ids")), bits, ints(decode("ranks")))
    }

    pub(crate) fn position(entries: &[(u32, f32, u32)]) -> PositionLogprobs {
        PositionLogprobs {
            entries: entries
                .iter()
                .map(|&(token_id, logprob, rank)| TokenLogprob {
                    token_id,
                    logprob,
                    rank,
                })
                .collect(),
        }
    }

    #[test]
    fn parse_logprobs_format() {
        assert_eq!(LogprobsFormat::parse(None), Some(LogprobsFormat::OpenAi));
        assert_eq!(
            LogprobsFormat::parse(Some(&serde_json::json!("openai"))),
            Some(LogprobsFormat::OpenAi)
        );
        assert_eq!(
            LogprobsFormat::parse(Some(&serde_json::json!("compact"))),
            Some(LogprobsFormat::Compact)
        );
        assert_eq!(
            LogprobsFormat::parse(Some(&serde_json::json!("Compact"))),
            None
        );
        assert_eq!(LogprobsFormat::parse(Some(&serde_json::json!(""))), None);
        assert_eq!(LogprobsFormat::parse(Some(&serde_json::Value::Null)), None);
        assert_eq!(LogprobsFormat::parse(Some(&serde_json::json!(1))), None);
    }

    #[test]
    fn compact_truncates_padded_rows_to_requested_width() {
        // Engine rows padded to a wider batch-wide top-k (4 slots) for a
        // request that asked for k = 1 (2 slots).
        let mut accumulator = CompactLogprobsAccumulator::new(2);
        accumulator.extend(Logprobs {
            positions: vec![
                position(&[(5, -0.5, 2), (4, -0.25, 1), (5, -0.5, 2), (6, -1.0, 3)]),
                position(&[(7, -0.1, 1), (7, -0.1, 1), (8, -2.0, 2), (9, -3.0, 3)]),
            ],
        });
        let block = CompactLogprobsJson::from(&accumulator.finish().expect("finish"));
        let value = serde_json::to_value(&block).unwrap();
        assert_eq!(value["num_slots"], 2);
        let (token_ids, bits, ranks) = decode_compact(&value);
        assert_eq!(token_ids, vec![5, 4, 7, 7]);
        assert_eq!(
            bits,
            [-0.5_f32, -0.25, -0.1, -0.1].map(f32::to_bits).to_vec()
        );
        assert_eq!(ranks, vec![2, 1]);

        // Unknown requested width (`logprobs: -1`): keep the engine width.
        let mut accumulator = CompactLogprobsAccumulator::new(0);
        accumulator.extend(Logprobs {
            positions: vec![position(&[(5, -0.5, 2), (4, -0.25, 1), (5, -0.5, 2)])],
        });
        assert_eq!(accumulator.finish().unwrap().num_slots, 3);
    }

    #[test]
    fn incremental_base64_matches_one_shot_for_all_split_points() {
        let data: Vec<u8> = (0..=255_u8).cycle().take(1000).collect();
        for chunk in 1..=11 {
            let mut segments = Base64Segments::default();
            for piece in data.chunks(chunk) {
                segments.push(piece);
            }
            let encoded = segments.finish();
            let expected = STANDARD.encode(&data);
            assert_eq!(encoded.len, expected.len());
            assert_eq!(encoded.to_base64_string(), expected, "chunk={chunk}");
        }
        for len in 0..7 {
            let mut segments = Base64Segments::default();
            segments.push(&data[..len]);
            assert_eq!(
                segments.finish().to_base64_string(),
                STANDARD.encode(&data[..len])
            );
        }
    }

    #[test]
    fn steps_coalesce_into_bounded_exact_segments() {
        // 300 KB raw -> 400 KB encoded: two steps fit one 1 MiB segment.
        let step = vec![7_u8; 3 * 100_000];
        let mut segments = Base64Segments::default();
        for _ in 0..4 {
            segments.push(&step);
        }
        let encoded = segments.finish();
        assert_eq!(encoded.segments.len(), 2);
        assert!(encoded.segments.iter().all(|s| s.len() == 800_000));
        assert_eq!(encoded.len, 1_600_000);

        // A step larger than the target becomes its own segment.
        let big = vec![7_u8; 3 * 300_000];
        let mut segments = Base64Segments::default();
        segments.push(&step);
        segments.push(&big);
        segments.push(&big);
        let encoded = segments.finish();
        let lens: Vec<_> = encoded.segments.iter().map(|s| s.len()).collect();
        assert_eq!(lens, [400_000, 1_200_000, 1_200_000]);
    }

    #[test]
    fn compact_round_trip_preserves_engine_rows_and_raw_bits() {
        let nan_payload = f32::from_bits(0x7fc0_1234);
        let rows = [
            position(&[(5, -0.25, 3), (9, -0.1, 1), (7, -0.2, 2), (5, -0.25, 3)]),
            position(&[
                (11, f32::NEG_INFINITY, 40),
                (1, -0.0, 1),
                (2, -1e30, 2),
                (3, nan_payload, 3),
            ]),
        ];
        let mut accumulator = CompactLogprobsAccumulator::new(4);
        accumulator.extend(Logprobs {
            positions: rows[..1].to_vec(),
        });
        accumulator.extend(Logprobs {
            positions: rows[1..].to_vec(),
        });
        assert_eq!(accumulator.num_positions(), 2);
        let block = CompactLogprobsJson::from(&accumulator.finish().expect("finish"));
        let value = serde_json::to_value(&block).unwrap();
        assert_eq!(value["num_positions"], 2);
        assert_eq!(value["num_slots"], 4);
        assert_eq!(value["dtype_token_ids"], "int32");
        assert_eq!(value["dtype_logprobs"], "float32");
        assert_eq!(value["byteorder"], "little");

        let (token_ids, bits, ranks) = decode_compact(&value);
        let expected_ids: Vec<i32> =
            rows.iter().flat_map(|p| p.entries.iter().map(|e| e.token_id as i32)).collect();
        let expected_bits: Vec<u32> = rows
            .iter()
            .flat_map(|p| p.entries.iter().map(|e| e.logprob.to_bits()))
            .collect();
        assert_eq!(token_ids, expected_ids);
        assert_eq!(bits, expected_bits);
        // Sampled slot first; ranks are the sampled token's engine rank.
        assert_eq!(token_ids[0], 5);
        assert_eq!(ranks, vec![3, 40]);
        // No clamp: -inf stays -inf, NaN payload and -0.0 sign preserved.
        assert_eq!(f32::from_bits(bits[4]), f32::NEG_INFINITY);
        assert_eq!(bits[5], (-0.0_f32).to_bits());
        assert_eq!(bits[7], 0x7fc0_1234);
    }

    #[test]
    fn compact_empty_uses_requested_slots() {
        let block = CompactLogprobsAccumulator::new(129).finish().expect("finish");
        let value = serde_json::to_value(CompactLogprobsJson::from(&block)).unwrap();
        assert_eq!(value["num_positions"], 0);
        assert_eq!(value["num_slots"], 129);
        assert_eq!(value["token_ids"], "");
        assert_eq!(value["logprobs"], "");
        assert_eq!(value["ranks"], "");
    }

    #[test]
    fn compact_rejects_narrower_row() {
        let mut accumulator = CompactLogprobsAccumulator::new(2);
        accumulator.extend(Logprobs {
            positions: vec![position(&[(1, -0.1, 1), (1, -0.1, 1)])],
        });
        accumulator.extend(Logprobs {
            positions: vec![position(&[(1, -0.1, 1)])],
        });
        assert!(accumulator.finish().is_err());
    }

    #[test]
    fn compact_narrow_first_row_does_not_shrink_requested_width() {
        // k = 2 -> S = 3 is fixed by the request; a narrow first row must not
        // set num_slots = 2 and truncate the later, valid rows.
        let mut accumulator = CompactLogprobsAccumulator::new(3);
        accumulator.extend(Logprobs {
            positions: vec![position(&[(1, -0.1, 1), (1, -0.1, 1)])],
        });
        accumulator.extend(Logprobs {
            positions: vec![position(&[(2, -0.1, 1), (2, -0.1, 1), (3, -0.2, 2)])],
        });
        assert!(accumulator.finish().is_err());

        // Valid rows keep S = k + 1 exactly.
        let mut accumulator = CompactLogprobsAccumulator::new(3);
        accumulator.extend(Logprobs {
            positions: vec![position(&[(2, -0.1, 1), (2, -0.1, 1), (3, -0.2, 2)])],
        });
        assert_eq!(accumulator.finish().unwrap().num_slots, 3);
    }
}
