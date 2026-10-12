// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

use std::sync::Arc;

use axum::Json;
use axum::extract::{Query, State};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Deserialize)]
pub(crate) struct ServerLoadQuery {
    #[serde(default)]
    include_inflight: bool,
    #[serde(default = "default_inflight_limit")]
    inflight_limit: i32,
}

fn default_inflight_limit() -> i32 {
    100
}

#[derive(Serialize)]
pub(crate) struct ServerLoadResponse {
    server_load: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    inflight: Option<Vec<Value>>,
}

pub async fn load(
    State(state): State<Arc<AppState>>,
    Query(query): Query<ServerLoadQuery>,
) -> Result<Json<ServerLoadResponse>, ApiError> {
    let inflight = if query.include_inflight {
        Some(
            state
                .chat
                .engine_core_client()
                .call_utility(
                    "get_inflight_queue_diagnostics",
                    (query.inflight_limit.clamp(0, 1000),),
                )
                .await
                .map_err(|error| utility_call_error("get_inflight_queue_diagnostics", error))?,
        )
    } else {
        None
    };
    Ok(Json(ServerLoadResponse {
        server_load: state.server_load(),
        inflight,
    }))
}
