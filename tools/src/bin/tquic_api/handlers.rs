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

/// Returns the per-second samples from the last session as JSON.
/// Always uses client samples as the base (non-empty for both modes).
/// For uplink, overlays server-measured jitter by interval index when available
/// (requires the server to have been (re)started via /server/start this session).
/// Shape: `{ "client": [ { "timestamp", "throughput", "jitter", "packetLoss" }, … ] }`
pub async fn last_json_result(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let mut samples = state.last_client.lock().await.clone();

    // Uplink: client has no jitter (it's the sender). Overlay server-side jitter
    // values when they exist — fall back to 0.0 gracefully if the server store
    // is empty (e.g. the server was already running before /server/start was called).
    let uplink = state.last_mode_uplink.load(std::sync::atomic::Ordering::Relaxed);
    if uplink && !samples.is_empty() {
        let srv = state.last_server.lock().await;
        if !srv.is_empty() {
            for (i, sample) in samples.iter_mut().enumerate() {
                let srv_jitter = srv.get(i)
                    .and_then(|s| s["jitter"].as_f64())
                    .unwrap_or(0.0);
                sample["jitter"] = serde_json::json!(srv_jitter);
            }
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
