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
//! bounded channel that the HTTP body drains, and the compact format hands its
//! pre-encoded base64 segments to the body without copying. The produced bytes
//! are identical to serializing the previous `GenerateResponse` value with
//! `serde_json`.
//!
//! Observability: the handler returns once headers are ready, so
//! `http_request_duration_seconds` and the `TraceLayer` latency stop before
//! the OpenAI-format body is rendered (previously they included the eager
//! serialization). The render task runs in the request span and logs
//! `generate response body rendered` (`render_wall_s`, `bytes`) at debug level.

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
use tracing::{debug, trace};
use tracing_futures::Instrument as _;
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
    /// `"logprobs": null` plus `"compact_logprobs"` (`None` omits the key).
    Compact(Option<CompactLogprobs>),
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
            // Omitted (not null) when logprobs were not requested, like Python.
            if let Some(block) = block.as_ref() {
                out.raw(b",\"compact_logprobs\":");
                write_compact(&mut out, block);
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

/// Renders OpenAI-format output logprobs in bounded chunks, releasing each
/// position after it is written.
struct OpenAiRenderer {
    positions: std::vec::IntoIter<PositionLogprobs>,
    first: bool,
    capacity_hint: usize,
}

impl OpenAiRenderer {
    fn new(positions: Vec<PositionLogprobs>) -> Self {
        Self {
            positions: positions.into_iter(),
            first: true,
            capacity_hint: 0,
        }
    }

    fn next_chunk(&mut self) -> Option<Bytes> {
        (self.positions.len() > 0).then(|| self.render_chunk())
    }

    fn render_chunk(&mut self) -> Bytes {
        let mut out = Vec::with_capacity(self.capacity_hint);
        TOKEN_FRAGMENTS.with(|fragments| {
            let mut fragments = fragments.borrow_mut();
            for position in self.positions.by_ref().take(OPENAI_POSITIONS_PER_CHUNK) {
                if !std::mem::replace(&mut self.first, false) {
                    out.push(b',');
                }
                write_openai_position_cached(&mut out, &position, &mut fragments);
                #[cfg(test)]
                tests::RENDERED_ON_THREAD.with(|count| count.set(count.get() + 1));
            }
        });
        self.capacity_hint = out.len() + out.len() / 8;
        Bytes::from(out)
    }
}

/// Render one OpenAI-format body (head, positions, tail) into `tx`.
///
/// Runs as a task on the runtime that built the response, which for the
/// offloaded generate route is the request runtime, so the HTTP runtime only
/// moves ready chunks. `send` waits while the channel is full (backpressure
/// from the client); a dropped body closes the channel and stops rendering.
async fn produce_openai_body(
    head: Vec<Bytes>,
    positions: Vec<PositionLogprobs>,
    tail: Vec<Bytes>,
    tx: tokio::sync::mpsc::Sender<BodyChunk>,
) {
    let started = std::time::Instant::now();
    let mut bytes = 0_usize;
    let mut renderer = OpenAiRenderer::new(positions);
    for part in head {
        bytes += part.len();
        if tx.send(BodyChunk::Data(part)).await.is_err() {
            return;
        }
    }
    while let Some(chunk) = renderer.next_chunk() {
        bytes += chunk.len();
        if tx.send(BodyChunk::Data(chunk)).await.is_err() {
            trace!(bytes, "generate response body dropped before completion");
            return;
        }
        // Keep the request runtime fair when the client drains faster than
        // we render (sends then never wait).
        tokio::task::yield_now().await;
    }
    for part in tail {
        bytes += part.len();
        if tx.send(BodyChunk::Data(part)).await.is_err() {
            return;
        }
    }
    // Explicit completion marker: a channel that closes without it (render
    // task panicked) fails the body instead of ending it cleanly.
    let _ = tx.send(BodyChunk::End).await;
    debug!(
        render_wall_s = started.elapsed().as_secs_f64(),
        bytes, "generate response body rendered"
    );
}

/// One message from [`produce_openai_body`] to [`ChannelBody`].
enum BodyChunk {
    Data(Bytes),
    /// The whole body was produced.
    End,
}

/// Chunks rendered for an in-flight OpenAI-format body, at most this many
/// buffered ahead of the socket (~0.9 MB each at top-128).
const OPENAI_BODY_CHANNEL_CHUNKS: usize = 2;

/// HTTP body fed by [`produce_openai_body`]; size unknown (chunked).
///
/// If the producer stops without its completion marker, the body yields an
/// error so hyper aborts the connection instead of writing the final chunk
/// terminator after truncated JSON.
struct ChannelBody {
    rx: tokio::sync::mpsc::Receiver<BodyChunk>,
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
            Some(BodyChunk::Data(chunk)) => Some(Ok(Frame::data(chunk))),
            Some(BodyChunk::End) => {
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
#[cfg(test)]
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
/// Calling its formatter crate directly measured no faster, so the encoding
/// stays byte-identical by construction.
fn write_f32(out: &mut Vec<u8>, value: f32) {
    serde_json::to_writer(&mut *out, &value).expect("f32 serializes");
}

/// Token ids at or above this are rendered without the fragment cache (the
/// dense index is bounded at 8 B per id, i.e. 8 MiB per thread).
const MAX_CACHED_TOKEN_ID: u32 = 1 << 20;

thread_local! {
    /// Per-thread cache of the id-dependent JSON around each candidate's
    /// logprob. Render tasks run on the request runtime's few threads, so the
    /// cache is shared by all requests rendered there and needs no locking.
    static TOKEN_FRAGMENTS: std::cell::RefCell<TokenFragments> =
        std::cell::RefCell::new(TokenFragments::default());
}

/// Location of one token id's fragments in [`TokenFragments::arena`].
#[derive(Clone, Copy, Default)]
struct FragmentSlot {
    offset: u32,
    prefix_len: u8,
    /// `0` means not rendered yet (a real entry is never empty).
    total_len: u8,
}

/// Pre-rendered `{"token":"token_id:N","logprob":` (prefix) and
/// `,"bytes":[116,...,digits]` (suffix) per token id: these are pure
/// functions of the id, so a candidate only formats its `f32`.
#[derive(Default)]
pub(super) struct TokenFragments {
    slots: Vec<FragmentSlot>,
    arena: Vec<u8>,
}

impl TokenFragments {
    /// The (prefix, suffix) fragments for `token_id`, rendering them on first
    /// use; `None` for ids beyond the cache bound.
    fn get(&mut self, token_id: u32) -> Option<(&[u8], &[u8])> {
        if token_id >= MAX_CACHED_TOKEN_ID {
            return None;
        }
        let index = token_id as usize;
        if index >= self.slots.len() {
            self.slots.resize(index + 1, FragmentSlot::default());
        }
        if self.slots[index].total_len == 0 {
            let offset = self.arena.len();
            let mut digits_buf = [0_u8; 10];
            let digits = format_u32(token_id, &mut digits_buf);
            self.arena.extend_from_slice(b"{\"token\":\"token_id:");
            self.arena.extend_from_slice(digits);
            self.arena.extend_from_slice(b"\",\"logprob\":");
            let prefix_len = self.arena.len() - offset;
            write_bytes_suffix(&mut self.arena, digits);
            let total_len = self.arena.len() - offset;
            self.slots[index] = FragmentSlot {
                offset: u32::try_from(offset).expect("fragment arena below 4 GiB"),
                prefix_len: prefix_len as u8,
                total_len: total_len as u8,
            };
        }
        let slot = self.slots[index];
        let start = slot.offset as usize;
        let entry = &self.arena[start..start + slot.total_len as usize];
        Some(entry.split_at(slot.prefix_len as usize))
    }
}

/// [`write_openai_position`] using the per-thread fragment cache.
pub(super) fn write_openai_position_cached(
    out: &mut Vec<u8>,
    position: &PositionLogprobs,
    fragments: &mut TokenFragments,
) {
    let chosen = &position.entries[0];
    write_candidate_cached(out, fragments, chosen.token_id, chosen.logprob);
    out.extend_from_slice(b",\"top_logprobs\":[");
    for (index, entry) in position.entries.iter().enumerate() {
        if index > 0 {
            out.push(b',');
        }
        write_candidate_cached(out, fragments, entry.token_id, entry.logprob);
        out.push(b'}');
    }
    out.extend_from_slice(b"]}");
}

fn write_candidate_cached(
    out: &mut Vec<u8>,
    fragments: &mut TokenFragments,
    token_id: u32,
    logprob: f32,
) {
    match fragments.get(token_id) {
        Some((prefix, suffix)) => {
            out.extend_from_slice(prefix);
            write_f32(out, clamp_logprob(logprob));
            out.extend_from_slice(suffix);
        }
        None => write_candidate_fields(out, token_id, logprob),
    }
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
        // panics the render task midway. The body must fail, so hyper aborts
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
    fn cached_fragments_match_serde_reference() {
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
        let mut fragments = TokenFragments::default();
        // Two passes over overlapping id sets exercise cache fill and reuse.
        for pass in 0..2 {
            for row in 0..3000 {
                let width = 1 + (next(&mut state) % 8) as usize;
                let entries = (0..width)
                    .map(|_| {
                        let token_id = match next(&mut state) % 6 {
                            0 => (next(&mut state) % 10) as u32,
                            1 => MAX_CACHED_TOKEN_ID - 1 - (next(&mut state) % 3) as u32,
                            2 => MAX_CACHED_TOKEN_ID + (next(&mut state) % 1000) as u32,
                            3 => u32::MAX - (next(&mut state) % 2) as u32,
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
                let mut cached = Vec::new();
                write_openai_position_cached(&mut cached, &position, &mut fragments);
                let mut uncached = Vec::new();
                write_openai_position(&mut uncached, &position);
                assert_eq!(
                    String::from_utf8_lossy(&cached),
                    String::from_utf8_lossy(&expected),
                    "pass {pass} row {row}"
                );
                assert_eq!(uncached, expected, "pass {pass} row {row}");
            }
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
        assert!(
            matches!(rx.recv().await, Some(BodyChunk::Data(head)) if head == Bytes::from_static(b"["))
        );
        rx.recv().await.unwrap();
        for _ in 0..10 {
            tokio::task::yield_now().await;
        }
        // Backpressure: at most the channel capacity is rendered ahead.
        assert!(rx.len() <= OPENAI_BODY_CHANNEL_CHUNKS);
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
