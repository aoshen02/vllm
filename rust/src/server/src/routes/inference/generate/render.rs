// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Direct JSON rendering for non-streaming raw generate responses.
//!
//! The non-streaming response can hold hundreds of thousands of positions with
//! 100+ candidates each. Instead of building a `ChatLogProbs` tree (two heap
//! strings and one byte vector per candidate) and serializing it into one
//! contiguous buffer, the default (`openai`) format is written straight from
//! the engine logprobs in bounded chunks as the HTTP body is polled, and the
//! compact format hands its pre-encoded base64 segments to the body without
//! copying. The produced bytes are identical to serializing the previous
//! `GenerateResponse` value with `serde_json`.

use std::collections::{HashMap, VecDeque};
use std::convert::Infallible;
use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::Body;
use axum::http::{HeaderValue, header};
use axum::response::{IntoResponse, Response};
use bytes::Bytes;
use http_body::{Frame, SizeHint};
use serde::Serialize;
use serde_json::Value;
use vllm_llm::PositionLogprobs;

use super::compact::{BYTEORDER, CompactLogprobs, DTYPE_LOGPROBS, DTYPE_TOKEN_IDS, EncodedArray};
use super::types::GenerateLogprob;
use crate::routes::openai::utils::logprobs::clamp_logprob;

/// Everything in a non-streaming generate response except the choice's
/// output logprobs.
#[derive(Debug, Clone)]
pub(super) struct GenerateEnvelope {
    pub request_id: String,
    pub finish_reason: String,
    pub token_ids: Vec<u32>,
    pub prompt_logprobs: Option<Vec<Option<HashMap<u32, GenerateLogprob>>>>,
    pub kv_transfer_params: Option<Value>,
    pub ec_transfer_params: Option<Value>,
}

/// Output logprobs of the single choice.
pub(super) enum ChoiceLogprobs {
    /// `"logprobs": null`.
    None,
    /// OpenAI chat-style `{"content": [...]}`. Positions must be non-empty.
    OpenAi(Vec<PositionLogprobs>),
    /// `"logprobs": null` plus `"compact_logprobs"` (`None` renders `null`).
    Compact(Option<CompactLogprobs>),
}

/// Positions rendered per body chunk in the OpenAI format (~14 KB each at
/// top-128).
const OPENAI_POSITIONS_PER_CHUNK: usize = 64;

/// Builds the response body as a list of byte parts.
#[derive(Default)]
struct PartsWriter {
    parts: Vec<Bytes>,
    current: Vec<u8>,
}

impl PartsWriter {
    fn raw(&mut self, bytes: &[u8]) {
        self.current.extend_from_slice(bytes);
    }

    fn json<T: Serialize + ?Sized>(&mut self, value: &T) {
        serde_json::to_writer(&mut self.current, value).expect("generate response must serialize");
    }

    fn flush(&mut self) {
        if !self.current.is_empty() {
            self.parts.push(Bytes::from(std::mem::take(&mut self.current)));
        }
    }

    fn encoded(&mut self, array: &EncodedArray) {
        self.raw(b"\"");
        self.flush();
        self.parts.extend(array.segments.iter().filter(|s| !s.is_empty()).cloned());
        self.raw(b"\"");
    }

    fn finish(mut self) -> Vec<Bytes> {
        self.flush();
        self.parts
    }
}

fn write_head(out: &mut PartsWriter, envelope: &GenerateEnvelope) {
    out.raw(b"{\"request_id\":");
    out.json(&envelope.request_id);
    out.raw(b",\"choices\":[{\"index\":0,\"logprobs\":");
}

/// Everything after the choice's `logprobs` value up to the end of the choice
/// object (exclusive of the closing `}`).
fn write_choice_fields(out: &mut PartsWriter, envelope: &GenerateEnvelope) {
    out.raw(b",\"finish_reason\":");
    out.json(&envelope.finish_reason);
    out.raw(b",\"token_ids\":");
    out.json(&envelope.token_ids);
}

fn write_tail(out: &mut PartsWriter, envelope: &GenerateEnvelope) {
    out.raw(b"}],\"prompt_logprobs\":");
    out.json(&envelope.prompt_logprobs);
    out.raw(b",\"kv_transfer_params\":");
    out.json(&envelope.kv_transfer_params);
    out.raw(b",\"ec_transfer_params\":");
    out.json(&envelope.ec_transfer_params);
    out.raw(b"}");
}

fn write_compact(out: &mut PartsWriter, block: &CompactLogprobs) {
    out.raw(b"{\"num_positions\":");
    out.json(&block.num_positions);
    out.raw(b",\"num_slots\":");
    out.json(&block.num_slots);
    out.raw(b",\"dtype_token_ids\":");
    out.json(DTYPE_TOKEN_IDS);
    out.raw(b",\"dtype_logprobs\":");
    out.json(DTYPE_LOGPROBS);
    out.raw(b",\"byteorder\":");
    out.json(BYTEORDER);
    out.raw(b",\"token_ids\":");
    out.encoded(&block.token_ids);
    out.raw(b",\"logprobs\":");
    out.encoded(&block.logprobs);
    out.raw(b",\"ranks\":");
    out.encoded(&block.ranks);
    out.raw(b"}");
}

/// Build the HTTP response for one non-streaming generate request.
pub(super) fn generate_response(envelope: GenerateEnvelope, logprobs: ChoiceLogprobs) -> Response {
    let body = match logprobs {
        ChoiceLogprobs::None => {
            let mut out = PartsWriter::default();
            write_head(&mut out, &envelope);
            out.raw(b"null");
            write_choice_fields(&mut out, &envelope);
            write_tail(&mut out, &envelope);
            Body::new(PartsBody::new(out.finish()))
        }
        ChoiceLogprobs::Compact(block) => {
            let mut out = PartsWriter::default();
            write_head(&mut out, &envelope);
            out.raw(b"null");
            write_choice_fields(&mut out, &envelope);
            out.raw(b",\"compact_logprobs\":");
            match block.as_ref() {
                Some(block) => write_compact(&mut out, block),
                None => out.raw(b"null"),
            }
            write_tail(&mut out, &envelope);
            Body::new(PartsBody::new(out.finish()))
        }
        ChoiceLogprobs::OpenAi(positions) => {
            let mut head = PartsWriter::default();
            write_head(&mut head, &envelope);
            head.raw(b"{\"content\":[");
            let mut tail = PartsWriter::default();
            tail.raw(b"]}");
            write_choice_fields(&mut tail, &envelope);
            write_tail(&mut tail, &envelope);
            Body::new(OpenAiLogprobsBody::new(
                head.finish(),
                positions,
                tail.finish(),
            ))
        }
    };

    let mut response = body.into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/json"),
    );
    response
}

/// Fixed list of byte parts with an exact size hint (so the response carries a
/// `Content-Length`).
pub(super) struct PartsBody {
    parts: VecDeque<Bytes>,
    remaining: u64,
}

impl PartsBody {
    fn new(parts: Vec<Bytes>) -> Self {
        let remaining = parts.iter().map(|p| p.len() as u64).sum();
        Self {
            parts: parts.into_iter().filter(|p| !p.is_empty()).collect(),
            remaining,
        }
    }
}

impl http_body::Body for PartsBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        match self.parts.pop_front() {
            Some(part) => {
                self.remaining -= part.len() as u64;
                Poll::Ready(Some(Ok(Frame::data(part))))
            }
            None => Poll::Ready(None),
        }
    }

    fn is_end_stream(&self) -> bool {
        self.parts.is_empty()
    }

    fn size_hint(&self) -> SizeHint {
        SizeHint::with_exact(self.remaining)
    }
}

/// Lazily renders OpenAI-format output logprobs in bounded chunks as the body
/// is polled, releasing each position after it is written.
pub(super) struct OpenAiLogprobsBody {
    head: VecDeque<Bytes>,
    positions: std::vec::IntoIter<PositionLogprobs>,
    first: bool,
    tail: VecDeque<Bytes>,
    capacity_hint: usize,
}

impl OpenAiLogprobsBody {
    fn new(head: Vec<Bytes>, positions: Vec<PositionLogprobs>, tail: Vec<Bytes>) -> Self {
        Self {
            head: head.into(),
            positions: positions.into_iter(),
            first: true,
            tail: tail.into(),
            capacity_hint: 0,
        }
    }

    fn render_chunk(&mut self) -> Bytes {
        let mut out = Vec::with_capacity(self.capacity_hint);
        for position in self.positions.by_ref().take(OPENAI_POSITIONS_PER_CHUNK) {
            if !std::mem::replace(&mut self.first, false) {
                out.push(b',');
            }
            write_openai_position(&mut out, &position);
        }
        self.capacity_hint = out.len() + out.len() / 8;
        Bytes::from(out)
    }
}

impl http_body::Body for OpenAiLogprobsBody {
    type Data = Bytes;
    type Error = Infallible;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, Infallible>>> {
        let this = &mut *self;
        if let Some(part) = this.head.pop_front() {
            return Poll::Ready(Some(Ok(Frame::data(part))));
        }
        if this.positions.len() > 0 {
            return Poll::Ready(Some(Ok(Frame::data(this.render_chunk()))));
        }
        Poll::Ready(this.tail.pop_front().map(|part| Ok(Frame::data(part))))
    }

    fn is_end_stream(&self) -> bool {
        self.head.is_empty() && self.positions.len() == 0 && self.tail.is_empty()
    }
}

/// Write one `ChatLogProbsContent` object for a raw generate position, byte-
/// identical to `serde_json` serialization of the value previously built by
/// `position_to_chat_logprobs_content`. `position.entries` must be non-empty.
pub(super) fn write_openai_position(out: &mut Vec<u8>, position: &PositionLogprobs) {
    let chosen = &position.entries[0];
    write_candidate_fields(out, chosen.token_id, chosen.logprob);
    out.extend_from_slice(b",\"top_logprobs\":[");
    for (index, entry) in position.entries.iter().enumerate() {
        if index > 0 {
            out.push(b',');
        }
        write_candidate_fields(out, entry.token_id, entry.logprob);
        out.push(b'}');
    }
    out.extend_from_slice(b"]}");
}

/// Write `{"token":"token_id:N","logprob":X,"bytes":[...]` (no closing brace).
fn write_candidate_fields(out: &mut Vec<u8>, token_id: u32, logprob: f32) {
    let mut digits_buf = [0_u8; 10];
    let digits = format_u32(token_id, &mut digits_buf);

    out.extend_from_slice(b"{\"token\":\"token_id:");
    out.extend_from_slice(digits);
    out.extend_from_slice(b"\",\"logprob\":");
    write_f32(out, clamp_logprob(logprob));
    // UTF-8 bytes of "token_id:" followed by the ASCII digits.
    out.extend_from_slice(b",\"bytes\":[116,111,107,101,110,95,105,100,58");
    for &digit in digits {
        out.extend_from_slice(DIGIT_BYTE_LITERALS[(digit - b'0') as usize]);
    }
    out.push(b']');
}

/// `",48"` .. `",57"`: the JSON array element for each ASCII digit byte.
const DIGIT_BYTE_LITERALS: [&[u8]; 10] = [
    b",48", b",49", b",50", b",51", b",52", b",53", b",54", b",55", b",56", b",57",
];

fn format_u32(mut value: u32, buf: &mut [u8; 10]) -> &[u8] {
    let mut start = buf.len();
    loop {
        start -= 1;
        buf[start] = b'0' + (value % 10) as u8;
        value /= 10;
        if value == 0 {
            break;
        }
    }
    &buf[start..]
}

/// `serde_json`'s `f32` encoding (shortest round-trip, non-finite as `null`).
fn write_f32(out: &mut Vec<u8>, value: f32) {
    serde_json::to_writer(&mut *out, &value).expect("f32 serializes");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_u32_matches_display() {
        let mut buf = [0_u8; 10];
        for value in [0, 7, 10, 99, 100, 12345, 151_935, u32::MAX] {
            assert_eq!(format_u32(value, &mut buf), value.to_string().as_bytes());
        }
    }

    /// Serve one response over real HTTP/1.1 (hyper via `axum::serve`) and
    /// return (raw header block, de-framed body).
    async fn fetch_over_http(make: fn() -> Response) -> (String, Vec<u8>) {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = axum::Router::new().route("/", axum::routing::get(move || async move { make() }));
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET / HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).await.unwrap();
        server.abort();

        let split = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        let head = String::from_utf8(raw[..split].to_vec()).unwrap().to_ascii_lowercase();
        let mut rest = &raw[split + 4..];
        if !head.contains("transfer-encoding: chunked") {
            return (head, rest.to_vec());
        }
        let mut body = Vec::new();
        loop {
            let line_end = rest.windows(2).position(|w| w == b"\r\n").unwrap();
            let size =
                usize::from_str_radix(std::str::from_utf8(&rest[..line_end]).unwrap(), 16).unwrap();
            rest = &rest[line_end + 2..];
            if size == 0 {
                break;
            }
            body.extend_from_slice(&rest[..size]);
            rest = &rest[size + 2..];
        }
        (head, body)
    }

    fn test_envelope() -> GenerateEnvelope {
        GenerateEnvelope {
            request_id: "http-1".to_string(),
            finish_reason: "abort".to_string(),
            token_ids: vec![1, 2],
            prompt_logprobs: None,
            kv_transfer_params: None,
            ec_transfer_params: None,
        }
    }

    fn positions() -> Vec<PositionLogprobs> {
        (0..300_u32)
            .map(|i| PositionLogprobs {
                entries: (0..3_u32)
                    .map(|j| vllm_llm::TokenLogprob {
                        token_id: i + j,
                        logprob: -(j as f32) * 0.5,
                        rank: j + 1,
                    })
                    .collect(),
            })
            .collect()
    }

    #[tokio::test]
    async fn http_framing_compact_has_length_and_openai_is_chunked() {
        use vllm_llm::{Logprobs, LogprobsAccumulator as _};

        use super::super::compact::CompactLogprobsAccumulator;

        let (head, body) = fetch_over_http(|| {
            let mut accumulator = CompactLogprobsAccumulator::new(3);
            accumulator.extend(Logprobs {
                positions: positions(),
            });
            let block = accumulator.finish().unwrap();
            generate_response(test_envelope(), ChoiceLogprobs::Compact(Some(block)))
        })
        .await;
        assert!(head.starts_with("http/1.1 200"), "{head}");
        assert!(head.contains("content-type: application/json"), "{head}");
        assert!(
            head.contains(&format!("content-length: {}", body.len())),
            "{head}"
        );
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["choices"][0]["compact_logprobs"]["num_positions"], 300);

        let (head, body) = fetch_over_http(|| {
            generate_response(test_envelope(), ChoiceLogprobs::OpenAi(positions()))
        })
        .await;
        assert!(head.starts_with("http/1.1 200"), "{head}");
        assert!(head.contains("content-type: application/json"), "{head}");
        assert!(head.contains("transfer-encoding: chunked"), "{head}");
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["choices"][0]["logprobs"]["content"].as_array().unwrap().len(),
            300
        );
    }
}
