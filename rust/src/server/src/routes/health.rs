// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use futures::StreamExt;
use serde::Deserialize;
use tokio::time::{Duration, timeout};
use uuid::Uuid;
use vllm_engine_core_client::protocol::request::EngineCoreRequest;
use vllm_engine_core_client::protocol::sampling::EngineCoreSamplingParams;

use crate::state::AppState;

#[derive(Deserialize)]
struct QueueLoad {
    num_requests: usize,
}

#[derive(Deserialize)]
struct EngineLoad {
    queues: Vec<QueueLoad>,
    #[serde(default)]
    has_pending_work: bool,
}

pub async fn health(State(state): State<Arc<AppState>>) -> StatusCode {
    if state.chat.engine_core_client().is_healthy() {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

pub async fn health_generate(State(state): State<Arc<AppState>>) -> StatusCode {
    let client = state.chat.engine_core_client();
    if !client.is_healthy() {
        return StatusCode::SERVICE_UNAVAILABLE;
    }
    let probe = async {
        let loads = client
            .call_utility::<EngineLoad, _>("get_inflight_queue_diagnostics", (0,))
            .await
            .map_err(|_| ())?;
        if loads.iter().any(|load| {
            load.has_pending_work || load.queues.iter().any(|queue| queue.num_requests > 0)
        }) {
            return Ok(());
        }
        let mut stream = client
            .call(EngineCoreRequest {
                request_id: format!("healthcheck-{}", Uuid::new_v4()),
                prompt_token_ids: Some(vec![0]),
                sampling_params: Some(EngineCoreSamplingParams {
                    max_tokens: 1,
                    temperature: 1.0,
                    top_k: 1,
                    ..Default::default()
                }),
                arrival_time: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .expect("system clock before Unix epoch")
                    .as_secs_f64(),
                ..Default::default()
            })
            .await
            .map_err(|_| ())?;
        match stream.next().await {
            Some(Ok(_)) => Ok(()),
            _ => Err(()),
        }
    };
    match timeout(Duration::from_secs(20), probe).await {
        Ok(Ok(())) if client.is_healthy() => StatusCode::OK,
        _ => StatusCode::SERVICE_UNAVAILABLE,
    }
}
