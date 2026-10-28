use std::collections::BTreeMap;
use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use rmpv::Value as MsgpackValue;
use serde::Deserialize;
use serde_json::{Map, Value};

use crate::error::ApiError;
use crate::state::AppState;
use crate::utils::utility_call_error;

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Action {
    Configure,
    Reset,
    GetState,
}

pub async fn control(
    Path(action): Path<Action>,
    State(state): State<Arc<AppState>>,
    body: Option<Json<Option<Map<String, Value>>>>,
) -> Result<Json<Vec<MsgpackValue>>, ApiError> {
    let action = match action {
        Action::Configure => "configure",
        Action::Reset => "reset",
        Action::GetState => "get_state",
    };
    let body = Value::Object(body.and_then(|Json(body)| body).unwrap_or_default()).to_string();
    let results = state
        .engine_core_client()
        .collective_rpc(
            "dumper_control",
            None,
            (action, body),
            BTreeMap::<String, Value>::new(),
        )
        .await
        .map_err(|error| utility_call_error("dumper_control", error))?;
    Ok(Json(results))
}
