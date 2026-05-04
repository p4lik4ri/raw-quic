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

use tquic_tools::wandb_logger::WandbLogger;

use crate::models::{ClientStartRequest, OverallStatus, ServerStartRequest, WandbUploadRequest};
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
    if let Some(v) = req.initial_congestion_window { cmd.args(["--initial-congestion-window", &v.to_string()]); }
    if let Some(v) = req.min_congestion_window     { cmd.args(["--min-congestion-window",     &v.to_string()]); }
    if let Some(v) = req.send_udp_payload_size     { cmd.args(["--send-udp-payload-size",     &v.to_string()]); }
    if let Some(v) = req.recv_udp_payload_size     { cmd.args(["--recv-udp-payload-size",     &v.to_string()]); }
    if let Some(v) = req.handshake_timeout         { cmd.args(["--handshake-timeout",         &v.to_string()]); }
    if let Some(v) = req.idle_timeout              { cmd.args(["--idle-timeout",              &v.to_string()]); }
    if let Some(v) = req.initial_rtt               { cmd.args(["--initial-rtt",               &v.to_string()]); }
    if let Some(v) = req.pto_linear_factor         { cmd.args(["--pto-linear-factor",         &v.to_string()]); }
    if let Some(v) = req.max_pto                   { cmd.args(["--max-pto",                   &v.to_string()]); }
    if let Some(v) = req.anti_amplification_factor { cmd.args(["--anti-amplification-factor", &v.to_string()]); }
    if let Some(v) = req.send_batch_size           { cmd.args(["--send-batch-size",           &v.to_string()]); }
    if let Some(v) = req.zerortt_buffer_size       { cmd.args(["--zerortt-buffer-size",       &v.to_string()]); }
    if req.enable_multipath {
        cmd.arg("--enable-multipath");
        if let Some(algor) = &req.multipath_algor {
            cmd.args(["--multipath-algor", algor]);
        }
    }
    if req.enable_retry       { cmd.arg("--enable-retry"); }
    if req.disable_encryption { cmd.arg("--disable-encryption"); }
    if let Some(keylog) = &req.keylog_file { cmd.args(["--keylog-file", keylog]); }
    if let Some(lf)    = &req.log_file    { cmd.args(["--log-file",    lf]); }
    if let Some(qd)    = &req.qlog_dir    { cmd.args(["--qlog-dir",    qd]); }
    for a in &req.extra_args { cmd.arg(a); }

    proc.output.lock().await.clear();
    // Parse server stdout into last_server: the server is the receiver in
    // uplink mode, so its interval rows contain real jitter values.
    match spawn_and_capture(cmd, Arc::clone(&proc.output), Some(Arc::clone(&state.last_server)), "server", None).await {
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
    if let Some(v) = req.threads                  { cmd.args(["--threads",                  &v.to_string()]); }
    if let Some(v) = req.max_concurrent_conns     { cmd.args(["--max-concurrent-conns",     &v.to_string()]); }
    if let Some(v) = req.max_concurrent_requests  { cmd.args(["--max-concurrent-requests",  &v.to_string()]); }
    if let Some(v) = req.initial_congestion_window { cmd.args(["--initial-congestion-window", &v.to_string()]); }
    if let Some(v) = req.min_congestion_window     { cmd.args(["--min-congestion-window",     &v.to_string()]); }
    if let Some(v) = req.send_udp_payload_size     { cmd.args(["--send-udp-payload-size",     &v.to_string()]); }
    if let Some(v) = req.recv_udp_payload_size     { cmd.args(["--recv-udp-payload-size",     &v.to_string()]); }
    if let Some(v) = req.handshake_timeout         { cmd.args(["--handshake-timeout",         &v.to_string()]); }
    if let Some(v) = req.idle_timeout              { cmd.args(["--idle-timeout",              &v.to_string()]); }
    if let Some(v) = req.initial_rtt               { cmd.args(["--initial-rtt",               &v.to_string()]); }
    if let Some(v) = req.pto_linear_factor         { cmd.args(["--pto-linear-factor",         &v.to_string()]); }
    if let Some(v) = req.max_pto                   { cmd.args(["--max-pto",                   &v.to_string()]); }
    if let Some(v) = req.send_batch_size           { cmd.args(["--send-batch-size",           &v.to_string()]); }
    if req.enable_multipath {
        cmd.arg("--enable-multipath");
        if let Some(algor) = &req.multipath_algor {
            cmd.args(["--multipath-algor", algor]);
        }
    }
    if req.disable_encryption { cmd.arg("--disable-encryption"); }
    if let Some(keylog) = &req.keylog_file { cmd.args(["--keylog-file", keylog]); }
    if let Some(lf)    = &req.log_file    { cmd.args(["--log-file",    lf]); }
    if let Some(qd)    = &req.qlog_dir    { cmd.args(["--qlog-dir",    qd]); }
    for a in &req.extra_args { cmd.arg(a); }

    proc.output.lock().await.clear();
    // Clear local server samples for same-host setup.
    state.last_server.lock().await.clear();
    // Derive the remote server API URL from the connect_to host.
    // connect_to is "host:port" — we reuse the host with the default API port 8000.
    let derived_server_api_url = req.connect_to
        .rsplit_once(':')
        .map(|(host, _)| format!("http://{}:8000", host));
    *state.server_api_url.lock().await = derived_server_api_url.clone();
    // Also clear the remote server's accumulated interval samples so that
    // /server/intervals only holds this test's data (not prior runs).
    if let Some(ref base_url) = derived_server_api_url {
        let clear_url = format!("{base_url}/server/clear");
        // Fire-and-forget — don't block client startup on the remote call.
        tokio::spawn(async move {
            if let Err(e) = reqwest::Client::new().post(&clear_url).send().await {
                log::warn!("[client_start] could not clear remote server intervals: {e}");
            }
        });
    }
    // Remember the mode so /LastJsonResult can pick the right sample store.
    state.last_mode_uplink.store(
        req.mode == "uplink",
        std::sync::atomic::Ordering::Relaxed,
    );
    // Parse interval lines from client stdout into last_client (last session only).
    const WANDB_KEY: &str =
        "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";
    let wandb_config = Some((WANDB_KEY.to_string(), "quic".to_string()));
    match spawn_and_capture(cmd, Arc::clone(&proc.output), Some(Arc::clone(&state.last_client)), "client", wandb_config).await {
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

/// `POST /server/clear` — discard accumulated server interval samples.
/// Called automatically by the remote client API at the start of each test
/// so that /server/intervals only holds the current session's data.
pub async fn server_clear(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    state.last_server.lock().await.clear();
    Json(serde_json::json!({ "ok": true }))
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
/// The authoritative "receiver" side is chosen by mode:
///   - **Downlink** (server→client): client is receiver → use `last_client`
///   - **Uplink**   (client→server): server is receiver → use `last_server`
///                                   (fetched locally or from remote API)
///
/// Matching is done by `interval_end` (relative seconds within the test,
/// e.g. 1.0, 2.0 …) which is clock-agnostic and works across machines.
///
/// Shape: `{ "client": [ { "timestamp", "throughput", "jitter", "packetLoss" }, … ] }`
pub async fn last_json_result(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let uplink = state.last_mode_uplink.load(std::sync::atomic::Ordering::Relaxed);

    // Helper: fetch server intervals from local store or remote API.
    async fn get_server_samples(state: &Arc<AppState>) -> Vec<serde_json::Value> {
        let local = state.last_server.lock().await.clone();
        if !local.is_empty() {
            return local;
        }
        let url_opt = state.server_api_url.lock().await.clone();
        if let Some(base_url) = url_opt {
            let fetch_url = format!("{base_url}/server/intervals");
            match reqwest::get(&fetch_url).await {
                Ok(resp) => match resp.json::<serde_json::Value>().await {
                    Ok(v) => return v["server"].as_array().cloned().unwrap_or_default(),
                    Err(e) => log::warn!("[last_json_result] parse remote intervals: {e}"),
                },
                Err(e) => log::warn!("[last_json_result] fetch {fetch_url}: {e}"),
            }
        }
        vec![]
    }

    let mut samples: Vec<serde_json::Value> = if uplink {
        // In uplink the server is the receiver: it has the real throughput,
        // jitter and packetLoss.  Use server intervals directly.
        get_server_samples(&state).await
    } else {
        // In downlink the client is the receiver.
        state.last_client.lock().await.clone()
    };

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

/// `POST /client/wandb_upload` — upload LinUCB metrics from the last client
/// run to Weights & Biases.
///
/// Because `tquic_client` may run on a machine without general internet access
/// (e.g. a Raspberry Pi on a closed lab LAN), the metrics JSONL file is written
/// locally by the client and the upload is performed here, from the API server,
/// which is reachable from the internet via Tailscale.
///
/// The metrics file path is detected automatically from the `[wandb] METRICS_FILE=`
/// line that `tquic_client` prints to stderr (captured in the output ring buffer).
/// You may also supply it explicitly in the request body.
pub async fn client_wandb_upload(
    State(state): State<Arc<AppState>>,
    body: Option<Json<WandbUploadRequest>>,
) -> Json<serde_json::Value> {
    let req = body.map(|b| b.0).unwrap_or_default();

    const DEFAULT_API_KEY: &str =
        "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";
    let api_key = req.api_key
        .as_deref()
        .unwrap_or(DEFAULT_API_KEY)
        .to_string();
    let project = req.project
        .as_deref()
        .unwrap_or("quic")
        .to_string();

    // ── Locate the metrics file ───────────────────────────────────────────────
    let metrics_path: String = if let Some(p) = req.metrics_file {
        p
    } else {
        // Scan the client output ring buffer for the most recent METRICS_FILE= line.
        let output = state.client.lock().await.output.lock().await.clone();
        let found = output.iter().rev()
            .find_map(|line| {
                let tag = "[wandb] METRICS_FILE=";
                line.find(tag).map(|pos| line[pos + tag.len()..].trim().to_string())
            });
        match found {
            Some(p) => p,
            None => return Json(serde_json::json!({
                "ok": false,
                "error": "No metrics file found in client output. Run a multipath client test first, or supply metrics_file in the request body."
            })),
        }
    };

    // ── Read the JSONL file ───────────────────────────────────────────────────
    let jsonl_content = match std::fs::read_to_string(&metrics_path) {
        Ok(s) => s,
        Err(e) => return Json(serde_json::json!({
            "ok": false,
            "error": format!("Could not read metrics file {metrics_path}: {e}")
        })),
    };
    let lines: Vec<String> = jsonl_content
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.to_string())
        .collect();
    if lines.is_empty() {
        return Json(serde_json::json!({
            "ok": false,
            "error": format!("Metrics file {metrics_path} is empty")
        }));
    }

    // ── Upload via WandbLogger (blocking) ────────────────────────────────────
    let result = tokio::task::spawn_blocking(move || {
        match WandbLogger::new(&api_key, &project) {
            None => Err("WandbLogger::new failed — check stderr for details".to_string()),
            Some(wb) => {
                let ok = wb.upload_history(&lines);
                if ok {
                    Ok(wb.run_url.clone())
                } else {
                    Err("upload_history failed — check stderr for details".to_string())
                }
            }
        }
    }).await;

    match result {
        Ok(Ok(run_url)) => Json(serde_json::json!({
            "ok": true,
            "run_url": run_url,
            "metrics_file": metrics_path,
        })),
        Ok(Err(msg)) => Json(serde_json::json!({ "ok": false, "error": msg })),
        Err(e) => Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
    }
}
