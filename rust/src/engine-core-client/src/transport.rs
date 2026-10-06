// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::ops::Deref;
use std::time::Duration;

use bytes::Bytes;
use enum_as_inner::EnumAsInner;
use futures::StreamExt as _;
use futures::stream::FuturesOrdered;
use thiserror_ext::AsReport;
use tokio::sync::mpsc;
use tokio::time::timeout;
use tokio_util::task::{AbortOnDropHandle, TaskTracker};
use tracing::{debug, error, info, trace, warn};
use zeromq::prelude::{Socket, SocketRecv, SocketSend};
use zeromq::util::PeerIdentity;
use zeromq::{PullSocket, RouterSendHalf, RouterSocket, ZmqError, ZmqMessage};

use crate::coordinator::CoordinatorBootstrap;
use crate::error::{Error, Result, bail_unexpected_handshake_message};
use crate::protocol::handshake::{
    EngineCoreReadyResponse, HandshakeAddresses, HandshakeInitMessage, ReadyMessage,
};
use crate::protocol::output::{EngineCoreOutputs, decode_engine_core_outputs};
use crate::protocol::{decode_msgpack, encode_msgpack};

/// Dedicated single-frame sentinel emitted by Python `EngineCoreProc` when the
/// engine dies.
pub const ENGINE_CORE_DEAD_SENTINEL: &[u8] = b"ENGINE_CORE_DEAD";

/// Opaque routing identity of one engine on the frontend transport.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EngineId(Bytes);

impl Debug for EngineId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Display the engine id as a hex string for easier debugging.
        write!(f, "EngineId({})", hex::encode(&self.0))
    }
}

impl EngineId {
    /// Convert the engine id into a ZMQ frame for sending.
    pub fn to_frame(&self) -> Bytes {
        self.0.clone()
    }

    /// Convert the engine id into a ZMQ frame for sending.
    pub fn into_frame(self) -> Bytes {
        self.0
    }

    /// Parse the Python-compatible engine index encoded in the routing
    /// identity.
    ///
    /// Python `EngineCoreProc` currently uses a two-byte little-endian engine
    /// index as its ROUTER/DEALER identity. Coordinator control messages
    /// such as `START_DP_WAVE(exclude_engine_index)` need that engine-side
    /// index rather than any frontend-local ordering.
    pub fn engine_index(&self) -> Option<u32> {
        if self.len() != 2 {
            return None;
        }
        Some(u16::from_le_bytes([self[0], self[1]]) as u32)
    }

    /// Construct an engine id from the Python-compatible engine index encoding
    /// (two-byte little-endian).
    pub fn from_engine_index(value: u16) -> Self {
        Self(Bytes::copy_from_slice(&value.to_le_bytes()))
    }
}

impl Deref for EngineId {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

impl From<Vec<u8>> for EngineId {
    fn from(value: Vec<u8>) -> Self {
        Self(Bytes::from(value))
    }
}

impl<const N: usize> From<&[u8; N]> for EngineId {
    fn from(value: &[u8; N]) -> Self {
        Self(Bytes::copy_from_slice(value))
    }
}

impl TryFrom<EngineId> for PeerIdentity {
    type Error = ZmqError;

    fn try_from(value: EngineId) -> std::result::Result<Self, Self::Error> {
        PeerIdentity::try_from(value.into_frame())
    }
}

/// Per-engine handshake result collected while bootstrapping one shared
/// transport.
#[derive(Clone, Debug)]
pub struct ConnectedEngine {
    /// The identity of the connected engine.
    pub engine_id: EngineId,
    /// Post-initialization configuration received from the engine on the input
    /// socket registration message.
    pub ready_response: EngineCoreReadyResponse,
}

/// Represents the connected shared transport plus all registered engines after
/// a successful multi-engine startup handshake.
pub struct ConnectedTransport {
    /// The local address of the shared input socket that all engines connect to
    /// for receiving requests.
    pub input_address: String,
    /// The local address of the shared output socket that all engines connect
    /// to for sending responses.
    pub output_address: String,
    /// All engines connected through the startup handshake.
    pub engines: Vec<ConnectedEngine>,
    /// Optional engine-facing coordinator transport used for in-process wave
    /// coordination.
    pub coordinator: Option<CoordinatorBootstrap>,

    /// The sending half of the shared input socket.
    pub input_send: RouterSendHalf,
    /// The shared output socket for receiving responses from all engines.
    pub output_socket: PullSocket,
}

#[derive(Clone, Debug, EnumAsInner)]
enum EngineStartupState {
    HelloReceived,
    ReadyReceived,
}

/// Connect to one or more engines through the startup handshake protocol,
/// returning the shared data-plane transport plus the registered engines.
pub async fn connect_handshake(
    handshake_address: &str,
    engine_count: usize,
    local_host: &str,
    local_input_address: Option<&str>,
    local_output_address: Option<&str>,
    enable_inproc_coordinator: bool,
    ready_timeout: Duration,
) -> Result<ConnectedTransport> {
    if engine_count == 0 {
        bail_unexpected_handshake_message!("expected engine_count >= 1");
    }

    info!(
        engine_count,
        handshake_address, "waiting for engines to connect"
    );

    // 1. Bind shared local input/output sockets first so every engine receives the same data-plane
    //    addresses during handshake.
    debug!(
        local_host,
        ?ready_timeout,
        engine_count,
        "binding shared transport sockets"
    );
    let (input_address, mut input_socket, output_address, output_socket) =
        bind_local_sockets(local_host, local_input_address, local_output_address).await?;
    info!(%input_address, %output_address, "bound local transport sockets");

    let mut coordinator = if enable_inproc_coordinator {
        Some(CoordinatorBootstrap::bind(local_host).await?)
    } else {
        None
    };

    // 2. Bind the shared handshake socket once. All engines connect to this socket with their own
    //    identities, and startup order does not matter.
    let mut handshake_socket = RouterSocket::new();
    handshake_socket.bind(handshake_address).await?;

    let mut engines = BTreeMap::new();

    // 3. Receive HELLO from every engine and send a matching INIT. When coordinator mode is
    //    enabled, the engines will not emit READY until the coordinator barrier below completes.
    while engines.len() < engine_count {
        debug!(
            handshake_address,
            connected = engines.len(),
            waiting_for = engine_count,
            "waiting for engine HELLO"
        );
        let message = timeout(ready_timeout, handshake_socket.recv()).await.map_err(|_| {
            Error::HandshakeTimeout {
                stage: "HELLO",
                timeout: ready_timeout,
            }
        })??;
        let (engine_id, handshake_message) = decode_handshake_message(message, None)?;
        match handshake_message.status.as_deref() {
            Some("HELLO") => {
                if engines.contains_key(&engine_id) {
                    bail_unexpected_handshake_message!(
                        "duplicate engine id {engine_id:?} observed during startup handshake"
                    );
                }
                debug!(handshake_address, ?engine_id, "received HELLO from engine");

                send_init_message(
                    &mut handshake_socket,
                    &engine_id,
                    &input_address,
                    &output_address,
                    coordinator.as_ref(),
                )
                .await?;
                debug!(handshake_address, ?engine_id, "sent INIT to engine");

                engines.insert(engine_id.clone(), EngineStartupState::HelloReceived);
            }
            Some("READY") => {
                if coordinator.is_some() {
                    bail_unexpected_handshake_message!(
                        "received READY for engine id {engine_id:?} before coordinator startup gate completed"
                    );
                }
                let state = match engines.get_mut(&engine_id) {
                    Some(state) if !state.is_ready_received() => state,
                    _ => {
                        bail_unexpected_handshake_message!(
                            "received READY for unexpected or duplicate engine id {engine_id:?}"
                        );
                    }
                };
                debug!(
                    handshake_address,
                    ?engine_id,
                    ?handshake_message,
                    "received overlapping READY from engine during HELLO phase"
                );
                *state = EngineStartupState::ReadyReceived;
            }
            other => {
                bail_unexpected_handshake_message!("unexpected handshake status {other:?}");
            }
        }
    }

    // 4. Optional coordinator startup gate. Without coordinator there is nothing to do.
    if let Some(coordinator) = coordinator.as_mut() {
        coordinator.wait_for_startup_gate(engine_count, ready_timeout).await?;
    }

    // 5. After the optional gate has opened, every engine may now send READY.
    while engines.values().any(|state| !state.is_ready_received()) {
        debug!(
            handshake_address,
            connected = engines.len(),
            ready = engines.values().filter(|state| state.is_ready_received()).count(),
            waiting_for = engine_count,
            "waiting for engine READY"
        );
        let message = timeout(ready_timeout, handshake_socket.recv()).await.map_err(|_| {
            Error::HandshakeTimeout {
                stage: "READY",
                timeout: ready_timeout,
            }
        })??;
        let (engine_id, handshake_message) = decode_handshake_message(message, None)?;
        match handshake_message.status.as_deref() {
            Some("READY") => {
                let state = match engines.get_mut(&engine_id) {
                    Some(state) if !state.is_ready_received() => state,
                    _ => {
                        bail_unexpected_handshake_message!(
                            "received READY for unexpected or duplicate engine id {engine_id:?}"
                        );
                    }
                };
                debug!(
                    handshake_address,
                    ?engine_id,
                    ?handshake_message,
                    "received READY from engine"
                );
                *state = EngineStartupState::ReadyReceived;
            }
            Some("HELLO") => {
                bail_unexpected_handshake_message!(
                    "received duplicate HELLO for engine id {engine_id:?} after INIT phase completed"
                );
            }
            other => {
                bail_unexpected_handshake_message!("unexpected handshake status {other:?}");
            }
        }
    }

    // 6. Wait for every engine to connect to the shared input socket and register itself.
    let engines =
        wait_for_input_registrations(&mut input_socket, engines.into_keys(), ready_timeout).await?;
    debug!(
        engine_count = engines.len(),
        "all engines registered on shared input socket"
    );

    info!(engine_count = engines.len(), "engines connected");

    let (input_send, _) = input_socket.split();

    Ok(ConnectedTransport {
        input_address,
        output_address,
        input_send,
        output_socket,
        engines,
        coordinator,
    })
}

/// Bind to Python-supplied frontend transport addresses and wait for
/// already-initialized engines to register themselves on the input socket.
///
/// This path mirrors Python's externally managed `AsyncMPClient` bootstrap
/// model: the addresses are already fixed by the supervisor, and engine
/// identities are synthesized from contiguous rank order instead of being
/// discovered through a Rust-owned handshake.
pub async fn connect_bootstrapped(
    input_address: &str,
    output_address: &str,
    engine_start_index: u32,
    engine_count: usize,
    ready_timeout: Duration,
) -> Result<ConnectedTransport> {
    if engine_count == 0 {
        bail_unexpected_handshake_message!("expected engine_count >= 1");
    }
    let engine_start_index =
        u16::try_from(engine_start_index).map_err(|_| Error::UnexpectedHandshakeMessage {
            message: "engine_start_index exceeds the two-byte engine identity limit".to_string(),
        })?;
    let engine_end_index =
        usize::from(engine_start_index).checked_add(engine_count).ok_or_else(|| {
            Error::UnexpectedHandshakeMessage {
                message: "engine_start_index + engine_count overflows".to_string(),
            }
        })?;
    if engine_end_index > usize::from(u16::MAX) + 1 {
        return Err(Error::UnexpectedHandshakeMessage {
            message: "engine_start_index + engine_count exceeds the two-byte engine identity limit"
                .to_string(),
        });
    }

    let mut input_socket = RouterSocket::new();
    let input_address = input_socket.bind(input_address).await?.to_string();

    let mut output_socket = PullSocket::new();
    let output_address = output_socket.bind(output_address).await?.to_string();

    let engines = wait_for_input_registrations(
        &mut input_socket,
        (0..engine_count).map(|offset| {
            let offset = u16::try_from(offset).expect("validated engine offset fits u16");
            EngineId::from_engine_index(engine_start_index + offset)
        }),
        ready_timeout,
    )
    .await?;
    info!(
        engine_count = engines.len(),
        "bootstrapped engines connected"
    );

    let (input_send, _) = input_socket.split();

    Ok(ConnectedTransport {
        input_address,
        output_address,
        engines,
        coordinator: None,
        input_send,
        output_socket,
    })
}

/// Bind new input and output sockets.
async fn bind_local_sockets(
    local_host: &str,
    local_input_address: Option<&str>,
    local_output_address: Option<&str>,
) -> Result<(String, RouterSocket, String, PullSocket)> {
    let mut input_socket = RouterSocket::new();
    let input_bind_address = local_input_address
        .map(str::to_owned)
        .unwrap_or_else(|| format!("tcp://{local_host}:0"));
    let input_address = input_socket.bind(&input_bind_address).await?.to_string();

    let mut output_socket = PullSocket::new();
    let output_bind_address = local_output_address
        .map(str::to_owned)
        .unwrap_or_else(|| format!("tcp://{local_host}:0"));
    let output_address = output_socket.bind(&output_bind_address).await?.to_string();

    Ok((input_address, input_socket, output_address, output_socket))
}

/// Decode a handshake message and validate its structure and identity.
fn decode_handshake_message(
    message: ZmqMessage,
    expected_id: Option<&EngineId>,
) -> Result<(EngineId, ReadyMessage)> {
    if message.len() != 2 {
        bail_unexpected_handshake_message!("expected 2 frames, got {}", message.len());
    }

    let frames = message.into_vec();
    let actual_id = EngineId(frames[0].clone());
    if let Some(expected_id) = expected_id
        && actual_id != *expected_id
    {
        return Err(Error::UnexpectedHandshakeIdentity {
            expected: expected_id.to_vec(),
            actual: actual_id.to_vec(),
        });
    }

    let handshake_message: ReadyMessage = decode_msgpack(&frames[1])?;
    Ok((actual_id, handshake_message))
}

/// Send an INIT message to the engine with the local socket addresses for the
/// engine to connect to, using the handshake socket.
async fn send_init_message(
    handshake_socket: &mut RouterSocket,
    engine_id: &EngineId,
    input_address: &str,
    output_address: &str,
    coordinator: Option<&CoordinatorBootstrap>,
) -> Result<()> {
    let init_message = HandshakeInitMessage {
        addresses: HandshakeAddresses {
            inputs: vec![input_address.to_string()],
            outputs: vec![output_address.to_string()],
            coordinator_input: coordinator.map(|c| c.input_address.clone()),
            coordinator_output: coordinator.map(|c| c.output_address.clone()),
            frontend_stats_publish_address: None,
        },
        parallel_config: Default::default(),
    };
    let payload = encode_msgpack(&init_message)?;
    let message = ZmqMessage::try_from(vec![engine_id.to_frame(), Bytes::from(payload)])
        .expect("handshake router messages must contain identity and payload");
    handshake_socket.send(message).await?;
    Ok(())
}

/// Receive the input registration message from each engine and validate its
/// identity.
///
/// Each registration contains 2 frames: `[identity, ready-payload]`.
///
/// Since vLLM commit `c8d98f81f676552c263f35bbde55e6edbe81b4e8` ("[Core]
/// Simplify API server handshake"), the payload is a msgpack-encoded
/// [`EngineCoreReadyResponse`] carrying post-initialization values such as
/// `max_model_len`.
async fn wait_for_input_registrations(
    input_socket: &mut RouterSocket,
    expected_engines: impl IntoIterator<Item = EngineId>,
    ready_timeout: Duration,
) -> Result<Vec<ConnectedEngine>> {
    let expected_engines = expected_engines.into_iter().collect::<Vec<_>>();
    let mut pending = expected_engines.iter().cloned().collect::<BTreeSet<_>>();
    let mut ready_responses = BTreeMap::new();

    while !pending.is_empty() {
        let registration = timeout(ready_timeout, input_socket.recv()).await.map_err(|_| {
            Error::InputRegistrationTimeout {
                timeout: ready_timeout,
            }
        })??;

        if registration.len() != 2 {
            bail_unexpected_handshake_message!(
                "expected 2 frames for engine input registration, got {}",
                registration.len()
            );
        }

        let frames = registration.into_vec();
        let actual_id = EngineId(frames[0].clone());
        if !pending.remove(&actual_id) {
            bail_unexpected_handshake_message!(
                "received input registration for unexpected engine id {actual_id:?}"
            );
        }

        if frames[1].is_empty() {
            bail_unexpected_handshake_message!(
                "expected msgpack EngineCoreReadyResponse for engine input registration, got empty payload from engine id {actual_id:?}"
            );
        }

        let ready_response: EngineCoreReadyResponse = decode_msgpack(&frames[1])?;
        debug!(
            ?actual_id,
            ?ready_response,
            "received input registration from engine"
        );
        ready_responses.insert(actual_id, ready_response);
    }

    Ok(expected_engines
        .into_iter()
        .map(|engine_id| {
            let ready_response = ready_responses
                .remove(&engine_id)
                .expect("every expected engine id has a decoded ready response");
            ConnectedEngine {
                engine_id,
                ready_response,
            }
        })
        .collect())
}

/// Send an encoded message to the engine through the input socket.
pub async fn send_message(
    input_send: &mut RouterSendHalf,
    engine_id: &EngineId,
    request_type: Bytes,
    payload: Bytes,
    aux_frames: Vec<Bytes>,
) -> Result<()> {
    let mut frames = Vec::with_capacity(3 + aux_frames.len());
    frames.extend([engine_id.to_frame(), request_type, payload]);
    frames.extend(aux_frames);
    let message =
        ZmqMessage::try_from(frames).expect("router messages must contain identity and payload");

    trace!(
        ?engine_id,
        frame_count = message.len(),
        "sending ZMQ message"
    );
    input_send.send(message).await?;
    Ok(())
}

/// Environment variable bounding how many engine output messages are decoded
/// concurrently by [`run_output_loop`].
const OUTPUT_DECODE_PARALLELISM_ENV: &str = "VLLM_RS_OUTPUT_DECODE_PARALLELISM";
/// Default cap on concurrently decoded output messages.
const DEFAULT_MAX_OUTPUT_DECODE_PARALLELISM: usize = 8;
/// Accepted range for `VLLM_RS_OUTPUT_DECODE_PARALLELISM`; values outside it
/// are clamped (with a warning).
const OUTPUT_DECODE_PARALLELISM_RANGE: std::ops::RangeInclusive<usize> = 1..=64;

/// Number of engine output messages decoded concurrently, resolved once per
/// process and logged: `VLLM_RS_OUTPUT_DECODE_PARALLELISM` clamped to
/// [`OUTPUT_DECODE_PARALLELISM_RANGE`], else the available parallelism capped
/// at [`DEFAULT_MAX_OUTPUT_DECODE_PARALLELISM`].
fn output_decode_parallelism() -> usize {
    static PARALLELISM: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *PARALLELISM.get_or_init(|| {
        let default = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(DEFAULT_MAX_OUTPUT_DECODE_PARALLELISM);
        let chosen = match std::env::var(OUTPUT_DECODE_PARALLELISM_ENV) {
            Err(_) => default,
            Ok(raw) => match raw.trim().parse::<u128>() {
                Ok(value) => {
                    let (min, max) = (
                        *OUTPUT_DECODE_PARALLELISM_RANGE.start(),
                        *OUTPUT_DECODE_PARALLELISM_RANGE.end(),
                    );
                    let clamped = value.clamp(min as u128, max as u128) as usize;
                    if clamped as u128 != value {
                        warn!(
                            value = %raw,
                            clamped,
                            "{OUTPUT_DECODE_PARALLELISM_ENV} out of range {min}..={max}, clamping"
                        );
                    }
                    clamped
                }
                // An all-digit value too large even for u128 is just "huge".
                Err(_)
                    if !raw.trim().is_empty() && raw.trim().bytes().all(|b| b.is_ascii_digit()) =>
                {
                    let clamped = *OUTPUT_DECODE_PARALLELISM_RANGE.end();
                    warn!(
                        value = %raw,
                        clamped,
                        "{OUTPUT_DECODE_PARALLELISM_ENV} out of range, clamping"
                    );
                    clamped
                }
                Err(_) => {
                    warn!(
                        value = %raw,
                        default,
                        "ignoring unparsable {OUTPUT_DECODE_PARALLELISM_ENV}"
                    );
                    default
                }
            },
        };
        info!(parallelism = chosen, "engine output decode parallelism");
        chosen
    })
}

/// One message (or terminal condition) taken off the output socket.
enum ReceivedOutput {
    Frames(Vec<Bytes>),
    EngineDead,
    Failed(ZmqError),
}

/// Source of raw engine output messages: the output PULL socket in
/// production, a scripted source in tests.
pub(crate) trait OutputSource: Send + 'static {
    fn recv_message(&mut self) -> impl Future<Output = zeromq::ZmqResult<ZmqMessage>> + Send;
}

impl OutputSource for PullSocket {
    fn recv_message(&mut self) -> impl Future<Output = zeromq::ZmqResult<ZmqMessage>> + Send {
        self.recv()
    }
}

/// Decode one engine output message (the former inline body of the loop).
fn decode_output_message(frames: Vec<Bytes>) -> Result<EngineCoreOutputs> {
    let frame_len = frames[0].len();
    match decode_engine_core_outputs(&frames) {
        Ok(decoded) => {
            trace!(frame_len, outputs = ?decoded, "decoded output message");
            Ok(decoded)
        }
        Err(error) => {
            // If we fail to decode the message from the engine, notify the client but keep
            // the output loop running to continue processing future
            // messages from the engine.
            warn!(frame_len, error = %error.as_report(), "failed to decode output message");
            Err(error)
        }
    }
}

/// Receive raw output messages. Runs as its own task so that the decode
/// pipeline only ever waits on a cancel-safe channel, never on the socket.
/// It stops right after a terminal condition, so like the sequential loop it
/// never takes a message off the socket after `ENGINE_CORE_DEAD` or an error.
async fn receive_output_messages<S: OutputSource>(
    mut source: S,
    raw_tx: mpsc::Sender<ReceivedOutput>,
) {
    loop {
        let received = match source.recv_message().await {
            Ok(message) => {
                trace!(frame_count = message.len(), "received output message");
                let frames = message.into_vec();
                let frame = frames.first().expect("output message must have at least one frame");
                if frame.as_ref() == ENGINE_CORE_DEAD_SENTINEL {
                    ReceivedOutput::EngineDead
                } else {
                    ReceivedOutput::Frames(frames)
                }
            }
            Err(error) => ReceivedOutput::Failed(error),
        };
        let terminal = !matches!(received, ReceivedOutput::Frames(_));
        if raw_tx.send(received).await.is_err() || terminal {
            return;
        }
    }
}

/// Run the output loop to receive messages from the engine and send them to the
/// provided channel.
///
/// Decoding (msgpack plus logprobs resolution) is CPU-bound and used to run
/// inline on this single task, which capped a frontend at one core of decode
/// for all engines and requests. Messages are now decoded on the blocking
/// pool, up to [`output_decode_parallelism`] at a time, and forwarded strictly
/// in the order they were received, so the dispatcher sees exactly the same
/// sequence as before:
/// - `ENGINE_CORE_DEAD` / a socket error is delivered after every message
///   received before it;
/// - a panic while receiving or decoding message `k` panics this task after
///   messages `< k` were delivered (later messages are dropped, as they would
///   not have been received by the sequential loop);
/// - decodes run under `decode_tasks`: dropping (aborting) this task aborts
///   decodes that have not started, and [`EngineCoreClient::shutdown`]
///   (`crate::client`) closes and awaits the tracker, so shutdown returns only
///   once no decode is running, as when decode ran inline. This bounds
///   leftover decode work only when shutdown is awaited; a client that is
///   merely dropped leaves at most its running decodes to finish on their own.
///
/// Concurrency: results are forwarded in receive order, so a slow message at
/// the head holds its slot and the completed messages behind it; sustained
/// concurrency is limited by that head-of-line wait (an inherent cost of an
/// order-preserving bounded window; the sequential loop had a window of 1).
///
/// Memory: besides the sequential loop's one message being decoded and the
/// downstream channel, at most `parallelism` messages are in flight (raw or
/// decoded behind a slower head), one sits in the raw channel and one is held
/// by the receive task while it waits to hand it over: `parallelism + 1`
/// messages more than before, per loop. A client with the in-process
/// coordinator runs a second loop on the coordinator socket sharing the
/// decode tracker, so its budget is `2 * parallelism` concurrent decodes and
/// `2 * (parallelism + 1)` extra messages (coordinator messages are small). Nothing is read ahead of that: the zeromq PULL
/// socket only reads a peer's stream when polled, so unread data stays in the
/// kernel socket buffer and the engine's send queue exactly as before.
pub async fn run_output_loop(
    output_socket: PullSocket,
    tx: mpsc::Sender<Result<EngineCoreOutputs>>,
    decode_tasks: TaskTracker,
) {
    run_output_loop_with(
        output_socket,
        tx,
        output_decode_parallelism(),
        decode_tasks,
        decode_output_message,
    )
    .await;
}

async fn run_output_loop_with<S: OutputSource>(
    source: S,
    tx: mpsc::Sender<Result<EngineCoreOutputs>>,
    parallelism: usize,
    decode_tasks: TaskTracker,
    decode: fn(Vec<Bytes>) -> Result<EngineCoreOutputs>,
) {
    let parallelism = parallelism.max(1);
    // Capacity 1: the pipeline takes a message only when a decode slot is
    // free, so a deeper channel would only prefetch more raw messages.
    let (raw_tx, mut raw_rx) = mpsc::channel(1);
    let receiver = AbortOnDropHandle::new(tokio::spawn(receive_output_messages(source, raw_tx)));
    let mut decoding = FuturesOrdered::new();
    // Every decode that was started before a terminal condition is forwarded
    // first, so that error ordering matches the sequential loop.
    let terminal = loop {
        tokio::select! {
            biased;
            Some(decoded) = decoding.next(), if !decoding.is_empty() => {
                if !forward_decoded(&tx, decoded).await {
                    return;
                }
            }
            received = raw_rx.recv(), if decoding.len() < parallelism => {
                match received {
                    Some(ReceivedOutput::Frames(frames)) => decoding.push_back(
                        AbortOnDropHandle::new(decode_tasks.spawn_blocking(move || decode(frames))),
                    ),
                    Some(ReceivedOutput::EngineDead) => {
                        // The engine has died; notify the client and shut down the
                        // output loop.
                        warn!("received ENGINE_CORE_DEAD sentinel from engine");
                        break Some(Error::EngineCoreDead);
                    }
                    Some(ReceivedOutput::Failed(error)) => {
                        // If we fail to receive a message from the engine, it's likely that
                        // the engine has crashed or become unreachable, so we should notify
                        // the client and shut down the output loop.
                        error!(error = %error.as_report(), "failed to receive output message");
                        break Some(Error::Transport(error));
                    }
                    // The receive task ended without a terminal message: it
                    // panicked (or was cancelled).
                    None => break None,
                }
            }
        }
    };
    while let Some(decoded) = decoding.next().await {
        if !forward_decoded(&tx, decoded).await {
            return;
        }
    }
    match terminal {
        Some(error) => {
            let _ = tx.send(Err(error)).await;
        }
        None => {
            // A receive-side panic propagates through this task, as it did when
            // the socket was read inline.
            if let Err(join_error) = receiver.await
                && join_error.is_panic()
            {
                std::panic::resume_unwind(join_error.into_panic());
            }
        }
    }
}

/// Forward one decoded message; `false` when the client side has shut down.
async fn forward_decoded(
    tx: &mpsc::Sender<Result<EngineCoreOutputs>>,
    decoded: std::result::Result<Result<EngineCoreOutputs>, tokio::task::JoinError>,
) -> bool {
    let decoded = match decoded {
        Ok(decoded) => decoded,
        // A panicking decode behaves as it did inline: it panics this task.
        Err(join_error) if join_error.is_panic() => {
            std::panic::resume_unwind(join_error.into_panic())
        }
        // Cancelled: the runtime is shutting down.
        Err(_) => return false,
    };
    if tx.send(decoded).await.is_err() {
        // If we fail to send the decoded message to the client, it's likely that the
        // client has shut down, so we should shut down the output loop as
        // well.
        warn!("output loop rx dropped, shutting down output loop");
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use bytes::Bytes;
    use tokio::sync::Notify;
    use tokio::time::timeout;
    use tokio_util::task::TaskTracker;
    use zeromq::{PullSocket, PushSocket, Socket, SocketSend as _, ZmqError, ZmqMessage};

    use super::{
        ENGINE_CORE_DEAD_SENTINEL, OutputSource, decode_output_message, run_output_loop_with,
    };
    use crate::error::{Error, Result};
    use crate::protocol::encode_msgpack;
    use crate::protocol::output::{
        EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs, UtilityCallOutput,
    };
    use crate::protocol::utility::UtilityOutput;

    /// Upper bound for any single wait in these tests.
    const WAIT: Duration = Duration::from_secs(20);

    fn batch(index: u64, requests: usize, tokens: usize) -> ZmqMessage {
        let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: (0..requests)
                .map(|request| EngineCoreOutput {
                    request_id: format!("req-{index}-{request}"),
                    new_token_ids: vec![7; tokens],
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        });
        ZmqMessage::from(encode_msgpack(&outputs).unwrap())
    }

    fn utility(index: u64) -> ZmqMessage {
        let outputs = EngineCoreOutputs::Utility(UtilityCallOutput {
            engine_index: 0,
            timestamp: 0.0,
            output: UtilityOutput {
                call_id: index.into(),
                failure_message: None,
                result: None,
            },
        });
        ZmqMessage::from(encode_msgpack(&outputs).unwrap())
    }

    /// Message `index` of a sequence: a request batch for even indices, a
    /// utility reply for odd ones.
    fn assert_message(index: u64, output: Result<EngineCoreOutputs>) {
        match output.unwrap() {
            EngineCoreOutputs::RequestBatch(batch) => {
                assert_eq!(index % 2, 0, "message {index}");
                assert_eq!(batch.outputs[0].request_id, format!("req-{index}-0"));
            }
            EngineCoreOutputs::Utility(utility) => {
                assert_eq!(utility.output.call_id.as_u64(), Some(index));
            }
            other => panic!("unexpected output {other:?}"),
        }
    }

    async fn next<T>(rx: &mut tokio::sync::mpsc::Receiver<T>) -> Option<T> {
        timeout(WAIT, rx.recv()).await.expect("timed out waiting for the output loop")
    }

    enum Scripted {
        Message(ZmqMessage),
        Error(ZmqError),
        Panic,
    }

    /// Replays a script, then blocks forever (like an idle socket). Counts
    /// how many items were taken off it.
    struct ScriptedSource {
        items: VecDeque<Scripted>,
        taken: Arc<AtomicUsize>,
    }

    impl ScriptedSource {
        fn new(items: impl IntoIterator<Item = Scripted>) -> (Self, Arc<AtomicUsize>) {
            let taken = Arc::new(AtomicUsize::new(0));
            let source = Self {
                items: items.into_iter().collect(),
                taken: taken.clone(),
            };
            (source, taken)
        }
    }

    impl OutputSource for ScriptedSource {
        fn recv_message(&mut self) -> impl Future<Output = zeromq::ZmqResult<ZmqMessage>> + Send {
            let next = self.items.pop_front();
            if next.is_some() {
                self.taken.fetch_add(1, Ordering::SeqCst);
            }
            async move {
                match next {
                    Some(Scripted::Message(message)) => Ok(message),
                    Some(Scripted::Error(error)) => Err(error),
                    Some(Scripted::Panic) => panic!("scripted receive panic"),
                    None => futures::future::pending().await,
                }
            }
        }
    }

    /// Slow decoder so that later (fast) messages complete first.
    fn slow_first_decode(frames: Vec<Bytes>) -> Result<EngineCoreOutputs> {
        let decoded = decode_output_message(frames)?;
        if matches!(decoded, EngineCoreOutputs::RequestBatch(_)) {
            std::thread::sleep(Duration::from_millis(300));
        }
        Ok(decoded)
    }

    /// Messages decode concurrently but reach the dispatcher in receive order,
    /// and a terminal condition is delivered after everything received before it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn output_loop_preserves_order_with_parallel_decode() {
        let mut pull = PullSocket::new();
        let endpoint = pull.bind("tcp://127.0.0.1:0").await.unwrap();
        let mut push = PushSocket::new();
        push.connect(&endpoint.to_string()).await.unwrap();
        let (tx, mut rx) = tokio::sync::mpsc::channel(4);
        let loop_task = tokio::spawn(run_output_loop_with(
            pull,
            tx,
            4,
            TaskTracker::new(),
            decode_output_message,
        ));

        const MESSAGES: u64 = 64;
        // Send from a separate task: the loop applies backpressure to the
        // socket while the test is not yet reading the channel.
        let sender = tokio::spawn(async move {
            for index in 0..MESSAGES {
                // Alternate heavy request batches (slow to decode) with tiny
                // utility replies (fast), so out-of-order completion is likely.
                let message = if index % 2 == 0 {
                    batch(index, 64, 4096)
                } else {
                    utility(index)
                };
                push.send(message).await.unwrap();
            }
            push.send(ZmqMessage::from(ENGINE_CORE_DEAD_SENTINEL.to_vec())).await.unwrap();
            push
        });

        for index in 0..MESSAGES {
            assert_message(index, next(&mut rx).await.unwrap());
        }
        assert!(matches!(
            next(&mut rx).await,
            Some(Err(Error::EngineCoreDead))
        ));
        timeout(WAIT, loop_task).await.unwrap().unwrap();
        drop(timeout(WAIT, sender).await.unwrap().unwrap());
    }

    /// A socket error arriving while a slow head decode is still running is
    /// delivered after it and after every message received before the error;
    /// then the channel closes and the message scripted after the error is
    /// never taken off the source.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transport_error_is_delivered_after_inflight_decodes() {
        let (source, taken) = ScriptedSource::new([
            Scripted::Message(batch(0, 1, 16)),
            Scripted::Message(utility(1)),
            Scripted::Message(batch(2, 1, 16)),
            Scripted::Error(ZmqError::Socket("scripted failure")),
            Scripted::Message(utility(99)),
        ]);
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let loop_task = tokio::spawn(run_output_loop_with(
            source,
            tx,
            4,
            TaskTracker::new(),
            slow_first_decode,
        ));
        for index in 0..3 {
            assert_message(index, next(&mut rx).await.unwrap());
        }
        assert!(matches!(
            next(&mut rx).await,
            Some(Err(Error::Transport(_)))
        ));
        assert!(next(&mut rx).await.is_none());
        timeout(WAIT, loop_task).await.unwrap().unwrap();
        assert_eq!(
            taken.load(Ordering::SeqCst),
            4,
            "consumed past the transport error"
        );
    }

    /// `ENGINE_CORE_DEAD` right behind a slow first decode: batch, then the
    /// terminal error, then nothing; the message after the sentinel is never
    /// taken off the source.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn engine_dead_waits_for_slow_first_decode() {
        let (source, taken) = ScriptedSource::new([
            Scripted::Message(batch(0, 1, 16)),
            Scripted::Message(ZmqMessage::from(ENGINE_CORE_DEAD_SENTINEL.to_vec())),
            Scripted::Message(utility(99)),
        ]);
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let loop_task = tokio::spawn(run_output_loop_with(
            source,
            tx,
            4,
            TaskTracker::new(),
            slow_first_decode,
        ));
        assert_message(0, next(&mut rx).await.unwrap());
        assert!(matches!(
            next(&mut rx).await,
            Some(Err(Error::EngineCoreDead))
        ));
        assert!(next(&mut rx).await.is_none());
        timeout(WAIT, loop_task).await.unwrap().unwrap();
        assert_eq!(
            taken.load(Ordering::SeqCst),
            2,
            "consumed past ENGINE_CORE_DEAD"
        );
    }

    /// A panic in the receive path is propagated by the output-loop task after
    /// everything received before it was delivered in order.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn receive_panic_drains_then_propagates() {
        let (source, _) = ScriptedSource::new([
            Scripted::Message(batch(0, 1, 16)),
            Scripted::Message(utility(1)),
            Scripted::Panic,
        ]);
        let (tx, mut rx) = tokio::sync::mpsc::channel(8);
        let loop_task = tokio::spawn(run_output_loop_with(
            source,
            tx,
            4,
            TaskTracker::new(),
            slow_first_decode,
        ));
        assert_message(0, next(&mut rx).await.unwrap());
        assert_message(1, next(&mut rx).await.unwrap());
        assert!(next(&mut rx).await.is_none());
        let error = timeout(WAIT, loop_task)
            .await
            .unwrap()
            .expect_err("loop must propagate the panic");
        assert!(error.is_panic());
    }

    static SHUTDOWN_ACTIVE: AtomicUsize = AtomicUsize::new(0);
    static SHUTDOWN_STARTED: AtomicUsize = AtomicUsize::new(0);
    static SHUTDOWN_DECODE_STARTED: Notify = Notify::const_new();

    fn tracked_slow_decode(frames: Vec<Bytes>) -> Result<EngineCoreOutputs> {
        SHUTDOWN_STARTED.fetch_add(1, Ordering::SeqCst);
        SHUTDOWN_ACTIVE.fetch_add(1, Ordering::SeqCst);
        SHUTDOWN_DECODE_STARTED.notify_one();
        std::thread::sleep(Duration::from_millis(400));
        let decoded = decode_output_message(frames);
        SHUTDOWN_ACTIVE.fetch_sub(1, Ordering::SeqCst);
        decoded
    }

    /// Shutdown semantics: after the loop task is aborted, awaiting the closed
    /// decode tracker returns only once no decode is running (as when decode
    /// ran inline and the aborted task finished its current decode first), and
    /// no decode starts afterwards.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_waits_for_started_decodes() {
        let (source, _) =
            ScriptedSource::new((0..16).map(|index| Scripted::Message(utility(index))));
        // The downstream channel is never read: decodes pile up behind it.
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let tracker = TaskTracker::new();
        let loop_task = tokio::spawn(run_output_loop_with(
            source,
            tx,
            4,
            tracker.clone(),
            tracked_slow_decode,
        ));
        timeout(WAIT, SHUTDOWN_DECODE_STARTED.notified())
            .await
            .expect("no decode started");
        loop_task.abort();
        assert!(timeout(WAIT, loop_task).await.unwrap().unwrap_err().is_cancelled());
        tracker.close();
        timeout(WAIT, tracker.wait()).await.expect("decodes did not finish");
        assert_eq!(
            SHUTDOWN_ACTIVE.load(Ordering::SeqCst),
            0,
            "a decode outlived shutdown"
        );
        let started = SHUTDOWN_STARTED.load(Ordering::SeqCst);
        assert!(
            started <= 4,
            "at most `parallelism` decodes ran, got {started}"
        );
        // Nothing can start later either: the tracker is empty and closed.
        assert!(tracker.is_empty());
    }

    /// Whether the gate is open, with a condvar to wake the gated decode.
    static GATE: (std::sync::Mutex<bool>, std::sync::Condvar) =
        (std::sync::Mutex::new(false), std::sync::Condvar::new());
    static GATE_FIRST_RUNNING: Notify = Notify::const_new();
    static GATED_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn open_gate() {
        *GATE.0.lock().unwrap_or_else(|e| e.into_inner()) = true;
        GATE.1.notify_all();
    }

    /// Opens the gate when dropped, on every exit including unwinding, so a
    /// failing assertion cannot leave the blocking thread gated while the
    /// runtime waits for it.
    struct OpenGateOnDrop;

    impl Drop for OpenGateOnDrop {
        fn drop(&mut self) {
            open_gate();
        }
    }

    /// The first call blocks the only blocking-pool thread until the gate
    /// opens; every call is counted.
    fn gated_decode(frames: Vec<Bytes>) -> Result<EngineCoreOutputs> {
        if GATED_CALLS.fetch_add(1, Ordering::SeqCst) == 0 {
            GATE_FIRST_RUNNING.notify_one();
            let open = GATE.0.lock().unwrap_or_else(|e| e.into_inner());
            drop(GATE.1.wait_while(open, |open| !*open).unwrap_or_else(|e| e.into_inner()));
        }
        decode_output_message(frames)
    }

    /// Queued decodes are cancelled when the loop is aborted: with a single
    /// blocking-pool thread held by a gated first decode, the queued decodes
    /// never run, even after the gate opens.
    #[test]
    fn aborting_the_loop_cancels_queued_decodes() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .unwrap();
        // Declared after the runtime, so it drops (opening the gate) first.
        let _open_gate = OpenGateOnDrop;
        runtime.block_on(async {
            let (source, _) =
                ScriptedSource::new((0..8).map(|index| Scripted::Message(utility(index))));
            let (tx, _rx) = tokio::sync::mpsc::channel(1);
            let tracker = TaskTracker::new();
            let loop_task = tokio::spawn(run_output_loop_with(
                source,
                tx,
                4,
                tracker.clone(),
                gated_decode,
            ));
            timeout(WAIT, GATE_FIRST_RUNNING.notified())
                .await
                .expect("first decode never ran");
            // Wait until the loop has queued the other decodes behind it.
            timeout(WAIT, async {
                while tracker.len() < 4 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("decodes were not queued");
            loop_task.abort();
            assert!(timeout(WAIT, loop_task).await.unwrap().unwrap_err().is_cancelled());
            open_gate();
            tracker.close();
            timeout(WAIT, tracker.wait()).await.expect("decodes did not finish");
            assert_eq!(
                GATED_CALLS.load(Ordering::SeqCst),
                1,
                "a queued decode ran after abort"
            );
        });
    }

    use super::bind_local_sockets;

    #[tokio::test]
    async fn bind_local_sockets_resolves_zero_port_bindings() {
        let (input_address, _input_socket, output_address, _output_socket) =
            bind_local_sockets("127.0.0.1", None, None).await.expect("bind local sockets");

        assert!(input_address.starts_with("tcp://127.0.0.1:"));
        assert!(output_address.starts_with("tcp://127.0.0.1:"));
        assert_ne!(input_address, output_address);
    }
}
