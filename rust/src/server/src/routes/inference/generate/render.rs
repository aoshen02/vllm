// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

//! Direct JSON rendering for non-streaming raw generate responses.
//!
//! The non-streaming response can hold hundreds of thousands of positions with
//! 100+ candidates each. Instead of building a `ChatLogProbs` tree (two heap
//! strings and one byte vector per candidate) and serializing it into one
//! contiguous buffer, the default (`openai`) format is written straight from
//! the engine logprobs in bounded chunks by a task on the runtime that built
//! the response (the request runtime for the offloaded generate route) into a
//! bounded channel that the HTTP body drains. The produced bytes are identical
//! to serializing the previous `GenerateResponse` value with `serde_json`.

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
use tracing_futures::Instrument as _;
use vllm_llm::PositionLogprobs;

use super::types::GenerateLogprob;
use crate::routes::openai::utils::logprobs::clamp_logprob;

/// Everything in a non-streaming generate response except the choice's
/// output logprobs.
#[derive(Debug)]
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
}

/// Positions rendered per body chunk in the OpenAI format. At top-128 one
/// position is ~13.9 KB, so a chunk is ~0.89 MB.
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

    /// Only used for values whose serialization is total (strings, numbers,
    /// `Vec<u32>`, the prompt-logprob maps with `u32` keys, `serde_json::Value`)
    /// into a `Vec<u8>`, which cannot fail; `serde_json::to_vec` on the same
    /// response (what `axum::Json` did) has the same failure modes.
    fn json<T: Serialize + ?Sized>(&mut self, value: &T) {
        serde_json::to_writer(&mut self.current, value).expect("generate response must serialize");
    }

    fn flush(&mut self) {
        if !self.current.is_empty() {
            self.parts.push(Bytes::from(std::mem::take(&mut self.current)));
        }
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
        ChoiceLogprobs::OpenAi(positions) => {
            let mut head = PartsWriter::default();
            write_head(&mut head, &envelope);
            head.raw(b"{\"content\":[");
            let mut tail = PartsWriter::default();
            tail.raw(b"]}");
            write_choice_fields(&mut tail, &envelope);
            write_tail(&mut tail, &envelope);
            let (tx, rx) = tokio::sync::mpsc::channel(OPENAI_BODY_CHANNEL_CHUNKS);
            tokio::spawn(
                produce_openai_body(head.finish(), positions, tail.finish(), tx)
                    .instrument(tracing::Span::current()),
            );
            Body::new(ChannelBody {
                rx,
                complete: false,
            })
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

/// Render one OpenAI-format body (head, positions, tail) into `tx`, releasing
/// each position after it is written.
///
/// Runs as a task on the runtime that built the response, which for the
/// offloaded generate route is the request runtime, so the HTTP runtime only
/// moves ready chunks. `send` waits while the channel is full (backpressure
/// from the client); a dropped body closes the channel and stops rendering.
/// `None` marks the end of the body.
async fn produce_openai_body(
    head: Vec<Bytes>,
    positions: Vec<PositionLogprobs>,
    tail: Vec<Bytes>,
    tx: tokio::sync::mpsc::Sender<Option<Bytes>>,
) {
    for part in head {
        if tx.send(Some(part)).await.is_err() {
            return;
        }
    }
    let mut positions = positions.into_iter();
    let mut first = true;
    while positions.len() > 0 {
        let mut chunk = Vec::new();
        for position in positions.by_ref().take(OPENAI_POSITIONS_PER_CHUNK) {
            if !std::mem::replace(&mut first, false) {
                chunk.push(b',');
            }
            write_openai_position(&mut chunk, &position);
            #[cfg(test)]
            tests::RENDERED_ON_THREAD.with(|count| count.set(count.get() + 1));
        }
        if tx.send(Some(Bytes::from(chunk))).await.is_err() {
            return;
        }
        // Keep the request runtime fair when the client drains faster than
        // we render (sends then never wait).
        tokio::task::yield_now().await;
    }
    for part in tail {
        if tx.send(Some(part)).await.is_err() {
            return;
        }
    }
    let _ = tx.send(None).await;
}

/// Chunks rendered for an in-flight OpenAI-format body, at most this many
/// buffered ahead of the socket (~0.9 MB each at top-128).
const OPENAI_BODY_CHANNEL_CHUNKS: usize = 2;

/// HTTP body fed by [`produce_openai_body`]; size unknown (chunked).
///
/// If the producer stops without its completion marker (cancelled, or
/// panicked in an unwinding build; release builds abort on panic), the body
/// yields an error so hyper aborts the connection instead of writing the
/// final chunk terminator after truncated JSON.
struct ChannelBody {
    rx: tokio::sync::mpsc::Receiver<Option<Bytes>>,
    complete: bool,
}

impl http_body::Body for ChannelBody {
    type Data = Bytes;
    type Error = std::io::Error;

    fn poll_frame(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Bytes>, std::io::Error>>> {
        if self.complete {
            return Poll::Ready(None);
        }
        self.rx.poll_recv(cx).map(|message| match message {
            Some(Some(chunk)) => Some(Ok(Frame::data(chunk))),
            Some(None) => {
                self.complete = true;
                None
            }
            None => Some(Err(std::io::Error::other(
                "generate response body render task ended before completion",
            ))),
        })
    }

    fn is_end_stream(&self) -> bool {
        self.complete
    }
}

/// Write one `ChatLogProbsContent` object for a raw generate position, byte-
/// identical to `serde_json` serialization of the value previously built by
/// `position_to_chat_logprobs_content`. `position.entries` must be non-empty.
fn write_openai_position(out: &mut Vec<u8>, position: &PositionLogprobs) {
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
    write_bytes_suffix(out, digits);
}

/// `,"bytes":[...]`: the UTF-8 bytes of "token_id:" followed by the digits.
fn write_bytes_suffix(out: &mut Vec<u8>, digits: &[u8]) {
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

    thread_local! {
        /// OpenAI positions rendered on the current thread.
        pub(super) static RENDERED_ON_THREAD: std::cell::Cell<usize> =
            const { std::cell::Cell::new(0) };
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_render_task_failure_errors_the_body_instead_of_ending_it() {
        // An invariant violation (an empty row slipping past validation)
        // panics the render task midway (tests unwind; release builds abort,
        // where cancellation is the reachable early stop). The body must fail, so hyper aborts
        // the connection rather than writing a clean chunked terminator after
        // truncated JSON.
        let mut rows = positions();
        rows[200].entries.clear();
        let response = generate_response(test_envelope(), ChoiceLogprobs::OpenAi(rows));
        let result = axum::body::to_bytes(response.into_body(), usize::MAX).await;
        assert!(result.is_err(), "truncated body completed cleanly");
    }

    /// Deterministic xorshift for test data.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn direct_writer_matches_serde_reference() {
        let specials = [
            0.0_f32,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -9999.0,
            -9999.5,
            f32::MIN_POSITIVE,
            f32::MIN_POSITIVE / 3.0,
            f32::MAX,
            f32::MIN,
            -1e-7,
            -0.1,
        ];
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for row in 0..3000 {
            let width = 1 + (next(&mut state) % 8) as usize;
            let entries = (0..width)
                .map(|_| {
                    let token_id = match next(&mut state) % 4 {
                        0 => (next(&mut state) % 10) as u32,
                        1 => u32::MAX - (next(&mut state) % 2) as u32,
                        _ => (next(&mut state) % 200_000) as u32,
                    };
                    let logprob = if next(&mut state).is_multiple_of(4) {
                        specials[(next(&mut state) % specials.len() as u64) as usize]
                    } else {
                        f32::from_bits(next(&mut state) as u32)
                    };
                    vllm_llm::TokenLogprob {
                        token_id,
                        logprob,
                        rank: 1,
                    }
                })
                .collect();
            let position = PositionLogprobs { entries };
            let expected = serde_json::to_vec(
                &super::super::position_to_chat_logprobs_content(&position).unwrap(),
            )
            .unwrap();
            let mut direct = Vec::new();
            write_openai_position(&mut direct, &position);
            assert_eq!(
                String::from_utf8_lossy(&direct),
                String::from_utf8_lossy(&expected),
                "row {row}"
            );
        }
    }

    #[tokio::test]
    async fn openai_producer_stops_when_body_is_dropped_and_is_bounded() {
        let many: Vec<PositionLogprobs> = positions().into_iter().cycle().take(64 * 50).collect();
        let (tx, mut rx) = tokio::sync::mpsc::channel(OPENAI_BODY_CHANNEL_CHUNKS);
        let producer = tokio::spawn(produce_openai_body(
            vec![Bytes::from_static(b"[")],
            many,
            vec![Bytes::from_static(b"]")],
            tx,
        ));
        // Head + one rendered chunk, then let the producer fill the channel.
        assert_eq!(rx.recv().await, Some(Some(Bytes::from_static(b"["))));
        rx.recv().await.unwrap();
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        // Backpressure: the producer waits on the full channel.
        assert!(!producer.is_finished());
        drop(rx);
        tokio::time::timeout(std::time::Duration::from_secs(5), producer)
            .await
            .expect("producer stops after the body is dropped")
            .unwrap();
    }

    #[test]
    fn openai_render_runs_on_request_runtime_not_body_poller() {
        // `generate_response` runs inside the handler on the request runtime;
        // the HTTP runtime only polls the body. Rendering must stay on the
        // former so control routes on the HTTP runtime are not starved.
        let request_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .build()
            .unwrap();
        let response = request_runtime.block_on(async {
            generate_response(test_envelope(), ChoiceLogprobs::OpenAi(positions()))
        });
        let http_runtime =
            tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
        RENDERED_ON_THREAD.with(|count| count.set(0));
        let body = http_runtime
            .block_on(axum::body::to_bytes(response.into_body(), usize::MAX))
            .unwrap();
        assert_eq!(
            RENDERED_ON_THREAD.with(|count| count.get()),
            0,
            "positions were rendered on the body-polling thread"
        );
        let json: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json["choices"][0]["logprobs"]["content"].as_array().unwrap().len(),
            300
        );
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
}
