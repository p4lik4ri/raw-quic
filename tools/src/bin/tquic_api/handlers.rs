//! Axum route handlers for every API endpoint.
//!
//! Each handler locks the relevant `ProcessState` (server or client), delegates
//! to `spawn::spawn_and_capture` to start the child process, and returns a JSON
//! response.  Stop handlers call `ProcessState::stop()`; status handlers return
//! a `ProcessStatus` snapshot including the last 1 000 lines of combined output.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;
use tokio::process::Command;

use crate::models::{ClientStartRequest, OverallStatus, ServerStartRequest};
use crate::spawn::{fmt_float, spawn_and_capture};
use crate::state::{AppState, ProcessStatus};

// ─────────────────────────────────── server ───────────────────────────────────

pub async fn server_start(
    State(state): State<Arc<AppState>>,
    Json(req):    Json<ServerStartRequest>,
) -> Json<serde_json::Value> {
    let mut proc = state.server.lock().await;
    if proc.is_running() {
        return Json(serde_json::json!({
            "ok": false, "error": "server already running", "pid": proc.pid
        }));
    }

    let bin = state.bin_dir.join("tquic_server");
    let mut cmd = Command::new(&bin);
    if let Some(dir) = &req.work_dir { cmd.current_dir(dir); }
    cmd.args(["-c", &req.cert, "-k", &req.key]);
    cmd.args(["--listen", &req.listen]);
    cmd.args(["--congestion-control-algor", &req.congestion_control]);
    cmd.args(["--log-level", &req.log_level]);
    if req.enable_multipath {
        cmd.arg("--enable-multipath");
        if let Some(algor) = &req.multipath_algor {
            cmd.args(["--multipath-algor", algor]);
        }
    }
    if let Some(keylog) = &req.keylog_file {
        cmd.args(["--keylog-file", keylog]);
    }
    for a in &req.extra_args { cmd.arg(a); }

    proc.output.lock().await.clear();
    // Parse server stdout into last_server: the server is the receiver in
    // uplink mode, so its interval rows contain real jitter values.
    match spawn_and_capture(cmd, Arc::clone(&proc.output), Some(Arc::clone(&state.last_server)), "server").await {
        Ok(child) => {
            let pid = child.id();
            proc.child = Some(child);
            proc.pid   = pid;
            Json(serde_json::json!({ "ok": "running", "pid": pid }))
        }
        Err(e) => Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
    }
}

pub async fn server_stop(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let mut proc = state.server.lock().await;
    match proc.stop().await {
        Ok(_)  => Json(serde_json::json!({ "ok": true })),
        Err(e) => Json(serde_json::json!({ "ok": false, "error": e })),
    }
}

pub async fn server_status(
    State(state): State<Arc<AppState>>,
) -> Json<ProcessStatus> {
    let mut proc = state.server.lock().await;
    Json(proc.status_snapshot().await)
}

// ─────────────────────────────────── client ───────────────────────────────────

pub async fn client_start(
    State(state): State<Arc<AppState>>,
    Json(req):    Json<ClientStartRequest>,
) -> Json<serde_json::Value> {
    let mut proc = state.client.lock().await;
    if proc.is_running() {
        return Json(serde_json::json!({
            "ok": false, "error": "client already running", "pid": proc.pid
        }));
    }

    let bin = state.bin_dir.join("tquic_client");
    let mut cmd = Command::new(&bin);
    if let Some(dir) = &req.work_dir { cmd.current_dir(dir); }
    cmd.args(["--connect-to", &req.connect_to]);
    cmd.args(["--congestion-control-algor", &req.congestion_control]);
    cmd.args(["--log-level", &req.log_level]);
    cmd.args(["--mode", &req.mode]);
    for addr in &req.local_addresses {
        cmd.args(["--local-addresses", addr]);
    }
    if let Some(d)  = req.duration          { cmd.args(["--duration",         &d.to_string()]); }
    if let Some(s)  = req.streams_per_conn  { cmd.args(["--streams-per-conn", &s.to_string()]); }
    if let Some(bw) = &req.bandwidth        { cmd.args(["--bandwidth",         bw]); }
    if req.enable_multipath {
        cmd.arg("--enable-multipath");
        if let Some(algor) = &req.multipath_algor {
            cmd.args(["--multipath-algor", algor]);
        }
    }
    for a in &req.extra_args { cmd.arg(a); }

    proc.output.lock().await.clear();
    // Clear server samples too — the server process stays running across tests,
    // so its background parse task keeps appending. Reset here so last_server
    // only holds the current test's intervals (for jitter overlay in uplink mode).
    state.last_server.lock().await.clear();
    // Remember the mode so /LastJsonResult can pick the right sample store.
    state.last_mode_uplink.store(
        req.mode == "uplink",
        std::sync::atomic::Ordering::Relaxed,
    );
    // Derive the remote server API URL from the connect_to host.
    // connect_to is "host:port" — we reuse the host with the default API port 8000.
    let derived_server_api_url = req.connect_to
        .rsplit_once(':')
        .map(|(host, _)| format!("http://{}:8000", host));
    *state.server_api_url.lock().await = derived_server_api_url;
    // Parse interval lines from client stdout into last_client (last session only).
    match spawn_and_capture(cmd, Arc::clone(&proc.output), Some(Arc::clone(&state.last_client)), "client").await {
        Ok(child) => {
            let pid = child.id();
            proc.child = Some(child);
            proc.pid   = pid;
            Json(serde_json::json!({ "ok": true, "pid": pid }))
        }
        Err(e) => Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
    }
}

pub async fn client_stop(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let mut proc = state.client.lock().await;
    match proc.stop().await {
        Ok(_)  => Json(serde_json::json!({ "ok": true })),
        Err(e) => Json(serde_json::json!({ "ok": false, "error": e })),
    }
}

pub async fn client_status(
    State(state): State<Arc<AppState>>,
) -> Json<ProcessStatus> {
    let mut proc = state.client.lock().await;
    Json(proc.status_snapshot().await)
}

// ─────────────────────────────────── combined ─────────────────────────────────

pub async fn overall_status(
    State(state): State<Arc<AppState>>,
) -> Json<OverallStatus> {
    let server = state.server.lock().await.status_snapshot().await;
    let client = state.client.lock().await.status_snapshot().await;
    Json(OverallStatus { server, client })
}

/// `GET /client/intervals` — raw per-second samples parsed from the last client session.
pub async fn client_intervals(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let samples = state.last_client.lock().await.clone();
    Json(serde_json::json!({ "client": samples }))
}

/// `GET /server/intervals` — raw per-second samples parsed from the last server session.
/// In uplink mode the server is the receiver so these rows carry real jitter values.
/// Cleared at the start of each new `/client/start` call.
pub async fn server_intervals(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let samples = state.last_server.lock().await.clone();
    Json(serde_json::json!({ "server": samples }))
}

/// `GET /LastJsonResult` — per-second samples for the last session.
///
/// Client samples always form the base (throughput, packetLoss, timestamps).
/// Server samples carry jitter (server is receiver in uplink mode).  The
/// overlay source is chosen in this priority order:
///   1. Local `last_server` store (same-host setup)
///   2. Remote `{server_api_url}/server/intervals` (different-host setup)
///
/// Matching is done by `interval_end` (relative seconds within the test,
/// e.g. 1.0, 2.0 …) which is clock-agnostic and works across machines.
///
/// Shape: `{ "client": [ { "timestamp", "throughput", "jitter", "packetLoss" }, … ] }`
pub async fn last_json_result(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let mut samples = state.last_client.lock().await.clone();

    if !samples.is_empty() {
        // 1. Try local store.
        let local_srv = state.last_server.lock().await.clone();

        // 2. If local store is empty, try fetching from remote server API.
        let srv: Vec<serde_json::Value> = if !local_srv.is_empty() {
            local_srv
        } else {
            let url_opt = state.server_api_url.lock().await.clone();
            if let Some(base_url) = url_opt {
                let fetch_url = format!("{base_url}/server/intervals");
                match reqwest::get(&fetch_url).await {
                    Ok(resp) => {
                        match resp.json::<serde_json::Value>().await {
                            Ok(v) => v["server"].as_array()
                                .cloned()
                                .unwrap_or_default(),
                            Err(e) => {
                                log::warn!("[last_json_result] parse remote intervals: {e}");
                                vec![]
                            }
                        }
                    }
                    Err(e) => {
                        log::warn!("[last_json_result] fetch {fetch_url}: {e}");
                        vec![]
                    }
                }
            } else {
                vec![]
            }
        };

        // Overlay server jitter onto client samples, matching by interval_end
        // (relative seconds from test start).  This is immune to clock skew
        // between machines because both sides count from their own t=0.
        if !srv.is_empty() {
            let srv_idx: Vec<(f64, f64)> = srv.iter().filter_map(|s| {
                let ie     = s["interval_end"].as_f64()?;
                let jitter = s["jitter"].as_f64().unwrap_or(0.0);
                Some((ie, jitter))
            }).collect();

            for sample in samples.iter_mut() {
                let client_ie = match sample["interval_end"].as_f64() {
                    Some(t) => t,
                    None    => continue,
                };
                if let Some((_, jitter)) = srv_idx.iter()
                    .min_by(|(a, _), (b, _)| {
                        (a - client_ie).abs()
                            .partial_cmp(&(b - client_ie).abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                {
                    sample["jitter"] = serde_json::json!(jitter);
                }
            }
        }
    }

    // Remove the internal interval_end key before returning.
    for sample in samples.iter_mut() {
        if let Some(obj) = sample.as_object_mut() {
            obj.remove("interval_end");
        }
    }

    Json(serde_json::json!({ "client": samples }))
}

/// Returns the client-side per-second samples from the last session as
/// Python-style plain text: `Last Json Result: [{'timestamp': …}, …]`
pub async fn server_last_json_result(
    State(state): State<Arc<AppState>>,
) -> String {
    let samples = state.last_client.lock().await.clone();
    let items: Vec<String> = samples.iter().map(|s| {
        let ts         = s["timestamp"].as_f64().unwrap_or(0.0);
        let throughput = s["throughput"].as_f64().unwrap_or(0.0);
        let jitter     = s["jitter"].as_f64().unwrap_or(0.0);
        let loss       = s["packetLoss"].as_f64().unwrap_or(0.0);
        format!(
            "{{'timestamp': {ts:.6}, 'throughput': {}, 'jitter': {}, 'packetLoss': {}}}",
            fmt_float(throughput),
            fmt_float(jitter),
            fmt_float(loss),
        )
    }).collect();
    format!("Last Json Result: [{}]", items.join(", "))
}
