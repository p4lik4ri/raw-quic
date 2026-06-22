//! Axum route handlers for every API endpoint.
//!
//! Each handler locks the relevant `ProcessState` (server or client), delegates
//! to `spawn::spawn_and_capture` to start the child process, and returns a JSON
//! response.  Stop handlers call `ProcessState::stop()`; status handlers return
//! a `ProcessStatus` snapshot including the last 1 000 lines of combined output.

use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use tokio::process::Command;
use tokio::time::{sleep, Duration};

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
        // In looped experiments, callers may start the next run a bit early.
        // Wait a short grace period for the previous run to finish so loop N+1
        // can start without requiring explicit polling in the caller.
        let wait_start = Instant::now();
        let wait_deadline = wait_start + Duration::from_secs(20);
        while proc.is_running() && Instant::now() < wait_deadline {
            sleep(Duration::from_millis(200)).await;
        }

        if proc.is_running() {
            return Json(serde_json::json!({
                "ok": false,
                "error": "client still running; wait for /client/status running=false before starting next loop",
                "pid": proc.pid,
            }));
        }
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
    // API loop runs may restart immediately; use a safer handshake timeout unless provided.
    let effective_handshake_timeout = req.handshake_timeout.unwrap_or(30000);
    cmd.args(["--handshake-timeout", &effective_handshake_timeout.to_string()]);
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
        if let Some(v) = req.linucb_alpha { cmd.args(["--linucb-alpha", &v.to_string()]); }
    }
    if req.disable_encryption { cmd.arg("--disable-encryption"); }
    if let Some(keylog) = &req.keylog_file { cmd.args(["--keylog-file", keylog]); }
    if let Some(lf)    = &req.log_file    { cmd.args(["--log-file",    lf]); }
    if let Some(qd)    = &req.qlog_dir    { cmd.args(["--qlog-dir",    qd]); }
    for a in &req.extra_args { cmd.arg(a); }

    // In downlink the server is the sender, so server-side scheduler controls
    // path selection. Setting RoundRobin only on the client won't affect data
    // scheduling; expose this explicitly in the start response.
    let rr_downlink_warning = if req.enable_multipath
        && req.mode.eq_ignore_ascii_case("downlink")
        && req.multipath_algor.as_deref().map(|a| a.eq_ignore_ascii_case("roundrobin")).unwrap_or(false)
    {
        Some("downlink uses server-side scheduler; configure /server/start with enable_multipath=true and multipath_algor=RoundRobin")
    } else {
        None
    };

    // Guard against back-to-back loop races: give the previous run a short
    // cool-down window so remote/server state can settle before reconnecting.
    sleep(Duration::from_millis(750)).await;

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
    match spawn_and_capture(cmd, Arc::clone(&proc.output), Some(Arc::clone(&state.last_client)), "client").await {
        Ok(child) => {
            let pid = child.id();
            proc.child = Some(child);
            proc.pid   = pid;
            let mut resp = serde_json::json!({
                "ok": true,
                "pid": pid,
                "effective_handshake_timeout": effective_handshake_timeout,
                "startup_cooldown_ms": 750,
            });
            if let Some(w) = rr_downlink_warning {
                if let Some(obj) = resp.as_object_mut() {
                    obj.insert("warning".to_string(), serde_json::json!(w));
                }
            }
            Json(resp)
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
        let mut srv = get_server_samples(&state).await;

        // Merge per-path (5G/satellite) throughput from client intervals.
        // The client tracks which bytes went on each path (via PathStats sent_bytes),
        // stores them as 5G_throughput (path0) / sat_throughput (path1) in last_client,
        // and we match by interval_end (relative seconds) which is clock-agnostic.
        let client_samples = state.last_client.lock().await.clone();
        if !client_samples.is_empty() {
            // Build a lookup: interval_end → (5g_throughput, satellite_throughput, packetLoss)
            let mut path_map: std::collections::HashMap<u64, (Option<f64>, Option<f64>, Option<f64>)> =
                std::collections::HashMap::new();
            for cs in &client_samples {
                // interval_end is stored as f64; use bits as hash key for exact match.
                let ie_bits = cs["interval_end"].as_f64().map(|v| v.to_bits());
                let p0 = cs.get("5G_throughput").and_then(|v| v.as_f64());
                let p1 = cs.get("sat_throughput").and_then(|v| v.as_f64());
                let pl = cs.get("packetLoss").and_then(|v| v.as_f64());
                if let Some(bits) = ie_bits {
                    path_map.insert(bits, (p0, p1, pl));
                }
            }
            for sample in srv.iter_mut() {
                let ie_bits = sample["interval_end"].as_f64().map(|v| v.to_bits());
                if let Some(bits) = ie_bits {
                    if let Some((p0, p1, pl)) = path_map.get(&bits) {
                        if let Some(obj) = sample.as_object_mut() {
                            if let Some(v) = p0 { obj.insert("5g_throughput".to_string(), serde_json::json!(v)); }
                            if let Some(v) = p1 { obj.insert("satellite_throughput".to_string(), serde_json::json!(v)); }
                            if let Some(v) = pl { obj.insert("packetLoss".to_string(), serde_json::json!(v)); }
                        }
                    }
                }
            }
        }
        srv
    } else {
        // In downlink the client is the receiver (authoritative throughput/jitter/loss).
        // Also merge sender-side per-interval loss from server intervals.
        let mut cli = state.last_client.lock().await.clone();
        let srv = get_server_samples(&state).await;
        if !cli.is_empty() && !srv.is_empty() {
            let mut sender_loss_map: std::collections::HashMap<u64, f64> =
                std::collections::HashMap::new();
            for ss in &srv {
                if let (Some(ie), Some(pl)) = (
                    ss["interval_end"].as_f64().map(|v| v.to_bits()),
                    ss.get("packetLoss").and_then(|v| v.as_f64()),
                ) {
                    sender_loss_map.insert(ie, pl);
                }
            }
            for sample in cli.iter_mut() {
                if let Some(bits) = sample["interval_end"].as_f64().map(|v| v.to_bits()) {
                    if let Some(pl) = sender_loss_map.get(&bits) {
                        if let Some(obj) = sample.as_object_mut() {
                            obj.insert("sender_packetLoss".to_string(), serde_json::json!(pl));
                        }
                    }
                }
            }
        }
        cli
    };

    // Remove the internal interval_end key before returning,
    // and rename throughput → total_throughput.
    for sample in samples.iter_mut() {
        if let Some(obj) = sample.as_object_mut() {
            obj.remove("interval_end");
            if let Some(t) = obj.remove("throughput") {
                obj.insert("total_throughput".to_string(), t);
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
