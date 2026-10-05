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

/// Maximum size of one encoded segment (a multiple of 4). Steps are split
/// across segments (keeping the base64 carry), so no segment exceeds it.
const SEGMENT_TARGET_BYTES: usize = 1 << 20;

/// Upper bound on the bytes packed per scratch batch (per array).
const SCRATCH_TARGET_BYTES: usize = 1 << 20;

/// Standard base64 alphabet (`base64::engine::general_purpose::STANDARD`).
const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// The two output characters for every 12-bit input value, so each 3-byte
/// group is encoded with two table lookups (8 KiB, stays in L1).
static BASE64_PAIRS: [[u8; 2]; 4096] = {
    let mut table = [[0_u8; 2]; 4096];
    let mut index = 0;
    while index < 4096 {
        table[index] = [BASE64_ALPHABET[index >> 6], BASE64_ALPHABET[index & 63]];
        index += 1;
    }
    table
};

/// Append the standard padded base64 encoding of `input` to `out`; byte-
/// identical to `STANDARD.encode`, about twice as fast as the scalar `base64`
/// engine for whole 3-byte groups (padding only for a final partial group).
pub(super) fn encode_base64_into(input: &[u8], out: &mut Vec<u8>) {
    let whole = input.len() - input.len() % 3;
    let start = out.len();
    out.resize(start + whole / 3 * 4, 0);
    for (src, dst) in input[..whole].chunks_exact(3).zip(out[start..].chunks_exact_mut(4)) {
        let group = (src[0] as usize) << 16 | (src[1] as usize) << 8 | src[2] as usize;
        dst[..2].copy_from_slice(&BASE64_PAIRS[group >> 12]);
        dst[2..].copy_from_slice(&BASE64_PAIRS[group & 0xfff]);
    }
    let tail = &input[whole..];
    if !tail.is_empty() {
        out.extend_from_slice(STANDARD.encode(tail).as_bytes());
    }
}

/// Incremental padded base64 encoder whose output is kept as immutable
/// segments that can be handed to the HTTP body without copying.
#[derive(Debug, Default)]
pub(super) struct Base64Segments {
    segments: Vec<Bytes>,
    current: Vec<u8>,
    carry: [u8; 2],
    carry_len: usize,
    encoded_len: usize,
}

impl Base64Segments {
    pub(super) fn push(&mut self, mut data: &[u8]) {
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

    /// Encode `data` and append it, splitting it so that no segment exceeds
    /// `SEGMENT_TARGET_BYTES`. Only the final call (from `finish`) may pass a
    /// length that is not a multiple of three.
    fn encode(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            // Start a new segment when the (first segment-sized piece of the)
            // data does not fit, so a typical engine step becomes one
            // exactly-sized allocation instead of being spread over a
            // growing buffer.
            let wanted = (data.len().div_ceil(3) * 4).min(SEGMENT_TARGET_BYTES);
            if !self.current.is_empty() && self.current.len() + wanted > SEGMENT_TARGET_BYTES {
                self.close_segment();
            }
            // Raw bytes (a multiple of 3) that still fit the open segment.
            let room = (SEGMENT_TARGET_BYTES - self.current.len()) / 4 * 3;
            let (piece, rest) = data.split_at(data.len().min(room));
            let encoded = piece.len().div_ceil(3) * 4;
            if self.current.is_empty() {
                self.current.reserve_exact(encoded);
            } else if self.current.capacity() - self.current.len() < encoded {
                // Coalescing small steps: grow geometrically but never past
                // the segment size (plain `String` growth would overshoot).
                let grow =
                    encoded.max(self.current.len()).min(SEGMENT_TARGET_BYTES - self.current.len());
                self.current.reserve_exact(grow);
            }
            encode_base64_into(piece, &mut self.current);
            self.encoded_len += encoded;
            data = rest;
        }
    }

    fn close_segment(&mut self) {
        let mut current = std::mem::take(&mut self.current);
        if current.capacity() > current.len() + current.len() / 4 {
            current.shrink_to_fit();
        }
        self.segments.push(Bytes::from(current));
    }

    pub(super) fn finish(mut self) -> EncodedArray {
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
    /// The request's row width `S = k + 1` (`k` = `logprobs`, else
    /// `len(logprob_token_ids)`); `0` when the request asked for no logprobs,
    /// in which case payloads are ignored. Wider engine rows (padded to the
    /// batch-wide max) are truncated to it, like the Python frontend;
    /// narrower rows are an error. (`logprobs: -1` is rejected for compact
    /// at validation, so the width never comes from engine data.)
    requested_slots: usize,
    /// SPEC v3 `compact_include_sampled: false`: drop engine slot 0 (the
    /// sampled token), keeping only the k top-k slots.
    skip_sampled: bool,
    /// SPEC v3 `compact_include_ranks: false`: omit `ranks`.
    skip_ranks: bool,
    num_positions: usize,
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

    /// Apply the SPEC v3 field switches (`compact_include_sampled`,
    /// `compact_include_ranks`); both default to included.
    pub(crate) fn with_switches(mut self, include_sampled: bool, include_ranks: bool) -> Self {
        self.skip_sampled = !include_sampled;
        self.skip_ranks = !include_ranks;
        self
    }

    /// Engine slots emitted per position.
    fn emitted_slots(&self) -> std::ops::Range<usize> {
        let first = usize::from(self.skip_sampled).min(self.requested_slots);
        first..self.requested_slots
    }

    /// Retained capacity of the packing scratch buffers.
    #[cfg(test)]
    fn scratch_capacity(&self) -> usize {
        self.scratch_token_ids
            .capacity()
            .max(self.scratch_logprobs.capacity())
            .max(self.scratch_ranks.capacity())
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
        let num_slots = self.emitted_slots().len();
        Ok(CompactLogprobs {
            num_positions: self.num_positions,
            num_slots,
            sampled_slot: !self.skip_sampled,
            token_ids: self.token_ids.finish(),
            logprobs: self.logprobs.finish(),
            ranks: (!self.skip_ranks).then(|| self.ranks.finish()),
        })
    }

    fn flush_candidates(&mut self) {
        self.token_ids.push(&self.scratch_token_ids);
        self.logprobs.push(&self.scratch_logprobs);
        self.scratch_token_ids.clear();
        self.scratch_logprobs.clear();
    }

    fn flush_ranks(&mut self) {
        self.ranks.push(&self.scratch_ranks);
        self.scratch_ranks.clear();
    }
}

impl LogprobsAccumulator for CompactLogprobsAccumulator {
    fn observe_output(&mut self, new_tokens: usize, logprob_positions: Option<usize>) {
        // Every generated token needs exactly one position, step by step (a
        // zero-token output, e.g. an abort before any token, needs none).
        let positions = logprob_positions.unwrap_or(0);
        if self.requested_slots > 0 && positions != new_tokens {
            self.fail(format!(
                "raw generate output carried {positions} logprob positions for {new_tokens} new tokens"
            ));
        }
    }

    fn extend(&mut self, step: Logprobs) {
        let slots = self.requested_slots;
        if slots == 0 || self.error.is_some() || step.positions.is_empty() {
            return;
        }
        // Validate every row before reserving anything.
        if let Some(narrow) = step.positions.iter().find(|p| p.entries.len() < slots) {
            let width = narrow.entries.len();
            self.fail(format!(
                "raw generate logprobs row has {width} candidates, expected at least {slots}"
            ));
            return;
        }

        // Pack candidates incrementally, flushing to the encoders whenever a
        // scratch buffer reaches SCRATCH_TARGET_BYTES, independently of row
        // boundaries: retained scratch stays bounded for any row width.
        let rows = step.positions.len();
        let emitted = self.emitted_slots();
        let candidate_bytes = rows.saturating_mul(emitted.len()).saturating_mul(4);
        let wanted = candidate_bytes.min(SCRATCH_TARGET_BYTES);
        self.scratch_token_ids
            .reserve_exact(wanted.saturating_sub(self.scratch_token_ids.len()));
        self.scratch_logprobs
            .reserve_exact(wanted.saturating_sub(self.scratch_logprobs.len()));
        if !self.skip_ranks {
            self.scratch_ranks.reserve_exact(
                (rows * 4).min(SCRATCH_TARGET_BYTES).saturating_sub(self.scratch_ranks.len()),
            );
        }

        for position in &step.positions {
            if !self.skip_ranks {
                let sampled_rank = position.entries[0].rank;
                if sampled_rank > i32::MAX as u32 {
                    self.fail(format!("sampled rank {sampled_rank} does not fit int32"));
                    return;
                }
                self.scratch_ranks.extend_from_slice(&sampled_rank.to_le_bytes());
                if self.scratch_ranks.len() >= SCRATCH_TARGET_BYTES {
                    self.flush_ranks();
                }
            }
            let mut row = &position.entries[emitted.clone()];
            while !row.is_empty() {
                // Copy as many candidates as fit before the next flush, in
                // one tight loop over pre-sized scratch.
                let room = (SCRATCH_TARGET_BYTES - self.scratch_token_ids.len()) / 4;
                let (chunk, rest) = row.split_at(row.len().min(room.max(1)));
                if let Some(entry) = chunk.iter().find(|e| e.token_id > i32::MAX as u32) {
                    self.fail(format!("token id {} does not fit int32", entry.token_id));
                    return;
                }
                let start = self.scratch_token_ids.len();
                self.scratch_token_ids.resize(start + chunk.len() * 4, 0);
                self.scratch_logprobs.resize(start + chunk.len() * 4, 0);
                for ((ids, logprobs), entry) in self.scratch_token_ids[start..]
                    .chunks_exact_mut(4)
                    .zip(self.scratch_logprobs[start..].chunks_exact_mut(4))
                    .zip(chunk)
                {
                    ids.copy_from_slice(&entry.token_id.to_le_bytes());
                    logprobs.copy_from_slice(&entry.logprob.to_bits().to_le_bytes());
                }
                if self.scratch_token_ids.len() >= SCRATCH_TARGET_BYTES {
                    self.flush_candidates();
                }
                row = rest;
            }
        }
        self.flush_candidates();
        self.flush_ranks();
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
    /// `false` when slot 0 (the sampled token) was dropped; rendered as
    /// `"sampled_slot": false`, omitted when `true`.
    pub sampled_slot: bool,
    pub token_ids: EncodedArray,
    pub logprobs: EncodedArray,
    /// `None` when ranks were not requested (key omitted).
    pub ranks: Option<EncodedArray>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sampled_slot: Option<bool>,
    pub token_ids: String,
    pub logprobs: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranks: Option<String>,
}

impl From<&CompactLogprobs> for CompactLogprobsJson {
    fn from(value: &CompactLogprobs) -> Self {
        Self {
            num_positions: value.num_positions,
            num_slots: value.num_slots,
            sampled_slot: (!value.sampled_slot).then_some(false),
            dtype_token_ids: DTYPE_TOKEN_IDS,
            dtype_logprobs: DTYPE_LOGPROBS,
            byteorder: BYTEORDER,
            token_ids: value.token_ids.to_base64_string(),
            logprobs: value.logprobs.to_base64_string(),
            ranks: value.ranks.as_ref().map(EncodedArray::to_base64_string),
        }
    }
}

/// Encode one step's logprobs (e.g. one streaming chunk) as a compact block.
pub(crate) fn encode_compact(
    logprobs: Logprobs,
    new_tokens: usize,
    requested_slots: usize,
    include_sampled: bool,
    include_ranks: bool,
) -> Result<CompactLogprobsJson, String> {
    let mut accumulator = CompactLogprobsAccumulator::new(requested_slots)
        .with_switches(include_sampled, include_ranks);
    accumulator.observe_output(new_tokens, Some(logprobs.len()));
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

        // A request that asked for no logprobs (S = 0) ignores payloads.
        let mut accumulator = CompactLogprobsAccumulator::new(0);
        accumulator.extend(Logprobs {
            positions: vec![position(&[(5, -0.5, 2), (4, -0.25, 1), (5, -0.5, 2)])],
        });
        assert_eq!(accumulator.num_positions(), 0);
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
    fn segments_never_exceed_target_and_match_one_shot() {
        let data: Vec<u8> = (0..7_000_003_u32).map(|i| (i * 31 + 7) as u8).collect();
        // Mixed push sizes: tiny (carry across segment boundaries), steps
        // smaller than a segment, and steps several segments long.
        for sizes in [
            vec![1_usize, 2, 300_000, 5, 3_000_001, 700_000, 1],
            vec![3_500_000],
            vec![1_048_575, 1, 1_048_576 / 4 * 3, 2],
        ] {
            let mut segments = Base64Segments::default();
            let mut offset = 0;
            for size in sizes.iter().cycle() {
                if offset >= data.len() {
                    break;
                }
                let end = (offset + size).min(data.len());
                segments.push(&data[offset..end]);
                offset = end;
            }
            let encoded = segments.finish();
            assert!(
                encoded
                    .segments
                    .iter()
                    .all(|s| !s.is_empty() && s.len() <= SEGMENT_TARGET_BYTES),
                "segment lens {:?}",
                encoded.segments.iter().map(|s| s.len()).collect::<Vec<_>>()
            );
            let expected = STANDARD.encode(&data);
            assert_eq!(encoded.len, expected.len());
            assert_eq!(encoded.to_base64_string(), expected);
        }
    }

    #[test]
    fn wide_steps_keep_scratch_and_segments_bounded() {
        // 256 positions x 4097 slots: 4 MiB per packed array in one step.
        let slots = 4097_u32;
        let positions: Vec<_> = (0..256_u32)
            .map(|row| PositionLogprobs {
                entries: (0..slots)
                    .map(|slot| TokenLogprob {
                        token_id: row * 7 + slot,
                        logprob: -(slot as f32) * 1e-3,
                        rank: slot.max(1),
                    })
                    .collect(),
            })
            .collect();
        let mut accumulator = CompactLogprobsAccumulator::new(slots as usize);
        accumulator.extend(Logprobs {
            positions: positions.clone(),
        });
        assert!(
            accumulator.scratch_capacity() <= 2 * SCRATCH_TARGET_BYTES,
            "scratch capacity {}",
            accumulator.scratch_capacity()
        );
        // Full segments carry no spare capacity.
        assert!(
            accumulator.token_ids.current.capacity() <= SEGMENT_TARGET_BYTES,
            "open segment capacity {}",
            accumulator.token_ids.current.capacity()
        );
        let block = accumulator.finish().unwrap();
        for array in [
            &block.token_ids,
            &block.logprobs,
            block.ranks.as_ref().unwrap(),
        ] {
            assert!(array.segments.iter().all(|s| s.len() <= SEGMENT_TARGET_BYTES));
        }
        let value = serde_json::to_value(CompactLogprobsJson::from(&block)).unwrap();
        let (token_ids, bits, _) = decode_compact(&value);
        let expected_ids: Vec<i32> = positions
            .iter()
            .flat_map(|p| p.entries.iter().map(|e| e.token_id as i32))
            .collect();
        let expected_bits: Vec<u32> = positions
            .iter()
            .flat_map(|p| p.entries.iter().map(|e| e.logprob.to_bits()))
            .collect();
        assert_eq!(token_ids, expected_ids);
        assert_eq!(bits, expected_bits);
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

    fn wide_row(slots: u32) -> PositionLogprobs {
        PositionLogprobs {
            entries: (0..slots)
                .map(|slot| TokenLogprob {
                    token_id: slot,
                    logprob: -(slot as f32),
                    rank: slot.max(1),
                })
                .collect(),
        }
    }

    #[test]
    fn table_base64_matches_standard_engine() {
        // Every 3-byte group value (2^24 groups), plus all tail lengths.
        let mut every_group = Vec::with_capacity(3 << 24);
        for value in 0_u32..1 << 24 {
            every_group.extend_from_slice(&value.to_be_bytes()[1..]);
        }
        let mut out = Vec::new();
        encode_base64_into(&every_group, &mut out);
        assert!(out == STANDARD.encode(&every_group).into_bytes());
        for len in 0..8 {
            let input = &every_group[1000..1000 + len];
            let mut out = b"prefix".to_vec();
            encode_base64_into(input, &mut out);
            assert_eq!(
                out,
                [b"prefix".as_slice(), STANDARD.encode(input).as_bytes()].concat()
            );
        }
    }

    #[test]
    fn single_row_wider_than_scratch_target_keeps_scratch_bounded() {
        let slots = 300_001_u32;
        let mut accumulator = CompactLogprobsAccumulator::new(slots as usize);
        accumulator.extend(Logprobs {
            positions: vec![wide_row(slots)],
        });
        assert!(
            accumulator.scratch_capacity() <= SCRATCH_TARGET_BYTES,
            "scratch capacity {}",
            accumulator.scratch_capacity()
        );
        let block = accumulator.finish().unwrap();
        let value = serde_json::to_value(CompactLogprobsJson::from(&block)).unwrap();
        let (token_ids, bits, ranks) = decode_compact(&value);
        assert_eq!(token_ids, (0..slots as i32).collect::<Vec<_>>());
        assert_eq!(
            bits,
            (0..slots).map(|s| (-(s as f32)).to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(ranks, vec![1]);
    }

    #[test]
    fn growing_steps_keep_retained_scratch_within_target() {
        // Same S across steps of increasing row counts: capped reservations
        // must not double the previous capacity past the target.
        let slots = 200_001_u32;
        let mut accumulator = CompactLogprobsAccumulator::new(slots as usize);
        for rows in [1, 2, 3, 5] {
            accumulator.extend(Logprobs {
                positions: vec![wide_row(slots); rows],
            });
            assert!(
                accumulator.scratch_capacity() <= SCRATCH_TARGET_BYTES,
                "after {rows} rows: scratch capacity {}",
                accumulator.scratch_capacity()
            );
        }
        // Ranks scratch too: many narrow rows, then more.
        let mut accumulator = CompactLogprobsAccumulator::new(1);
        for rows in [100_000, 200_000, 300_000] {
            accumulator.extend(Logprobs {
                positions: vec![wide_row(1); rows],
            });
            assert!(
                accumulator.scratch_capacity() <= SCRATCH_TARGET_BYTES,
                "after {rows} narrow rows: scratch capacity {}",
                accumulator.scratch_capacity()
            );
        }
        assert_eq!(accumulator.finish().unwrap().num_positions, 600_000);
    }

    #[test]
    fn huge_requested_width_does_not_reserve_before_checking_rows() {
        // A requested width far above the actual rows must fail cleanly
        // without reserving `rows * requested * 4` bytes first.
        let mut accumulator = CompactLogprobsAccumulator::new(1 << 31);
        accumulator.extend(Logprobs {
            positions: vec![wide_row(3)],
        });
        assert!(
            accumulator.scratch_capacity() <= SCRATCH_TARGET_BYTES,
            "scratch capacity {}",
            accumulator.scratch_capacity()
        );
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
