//! tquic_api — HTTP control plane for tquic_server and tquic_client.
//!
//! Listens on 0.0.0.0:8000 by default (override with TQUIC_API_PORT).
//! Looks for tquic_server / tquic_client in the same directory as this binary
//! (override with TQUIC_BIN_DIR).
//!
//! Endpoints
//! ─────────────────────────────────────────────────────────────────────────────
//!  POST /server/start   — start tquic_server
//!  POST /server/stop    — kill running tquic_server
//!  GET  /server/status  — pid, running flag, last 1000 log lines
//!
//!  POST /client/start   — start tquic_client
//!  POST /client/stop    — kill running tquic_client
//!  GET  /client/status  — pid, running flag, last 1000 log lines
//!
//!  GET  /status         — both server + client status
//!
//! Example payloads
//! ─────────────────────────────────────────────────────────────────────────────
//!  POST /server/start
//!  {
//!    "cert": "cert.crt",
//!    "key":  "cert.key",
//!    "listen": "0.0.0.0:4433",
//!    "congestion_control": "BBR",
//!    "enable_multipath": true,
//!    "multipath_algor": "RoundRobin",
//!    "log_level": "info"
//!  }
//!
//!  POST /client/start
//!  {
//!    "connect_to": "10.1.100.221:4433",
//!    "local_addresses": ["10.1.100.142", "10.1.100.138"],
//!    "duration": 20,
//!    "bandwidth": "500M",
//!    "mode": "downlink",
//!    "congestion_control": "BBR",
//!    "enable_multipath": true,
//!    "multipath_algor": "RoundRobin"
//!  }

use std::collections::VecDeque;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

// ─────────────────────────────────── constants ────────────────────────────────

/// Maximum number of output lines kept in memory per process.
const OUTPUT_CAP: usize = 1000;

// ─────────────────────────────────── state ────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ProcessStatus {
    pub running: bool,
    pub pid: Option<u32>,
    /// Last up-to OUTPUT_CAP lines of stdout+stderr combined.
    pub output: Vec<String>,
}

struct ProcessState {
    child:  Option<Child>,
    pid:    Option<u32>,
    output: Arc<Mutex<VecDeque<String>>>,
}

impl ProcessState {
    fn new() -> Self {
        Self {
            child:  None,
            pid:    None,
            output: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Returns true if the child is still alive (polls without blocking).
    fn is_running(&mut self) -> bool {
        match &mut self.child {
            None => false,
            Some(child) => match child.try_wait() {
                Ok(Some(_)) => {
                    // Process exited — clean up handles.
                    self.child = None;
                    self.pid   = None;
                    false
                }
                Ok(None) => true, // still running
                Err(_)   => false,
            },
        }
    }

    async fn status_snapshot(&mut self) -> ProcessStatus {
        let running = self.is_running();
        let output  = self.output.lock().await;
        ProcessStatus {
            running,
            pid:    self.pid,
            output: output.iter().cloned().collect(),
        }
    }

    async fn stop(&mut self) -> Result<(), String> {
        match &mut self.child {
            None        => Err("not running".into()),
            Some(child) => {
                child.kill().await.map_err(|e| e.to_string())?;
                self.child = None;
                self.pid   = None;
                Ok(())
            }
        }
    }
}

struct AppState {
    server:  Mutex<ProcessState>,
    client:  Mutex<ProcessState>,
    bin_dir: PathBuf,
}

// ─────────────────────────────── request payloads ─────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ServerStartRequest {
    pub cert: String,
    pub key:  String,

    #[serde(default = "default_listen")]
    pub listen: String,

    /// BBR / BBR3 / Cubic  (case-insensitive, passed straight to --congestion-control-algor)
    #[serde(default = "default_cc")]
    pub congestion_control: String,

    #[serde(default = "default_log")]
    pub log_level: String,

    #[serde(default)]
    pub enable_multipath: bool,

    /// RoundRobin / MinRTT / Redundant
    pub multipath_algor: Option<String>,

    /// Any additional raw CLI flags passed verbatim, e.g. ["--send-batch-size","16"]
    #[serde(default)]
    pub extra_args: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct ClientStartRequest {
    pub connect_to: String,

    #[serde(default)]
    pub local_addresses: Vec<String>,

    pub duration:        Option<u64>,
    pub streams_per_conn: Option<u64>,

    /// Human-readable bandwidth, e.g. "500M", "1G". 0 / omit = unlimited.
    pub bandwidth: Option<String>,

    /// "downlink" (default) or "uplink"
    #[serde(default = "default_mode")]
    pub mode: String,

    #[serde(default = "default_cc")]
    pub congestion_control: String,

    #[serde(default = "default_log")]
    pub log_level: String,

    #[serde(default)]
    pub enable_multipath: bool,

    pub multipath_algor: Option<String>,

    #[serde(default)]
    pub extra_args: Vec<String>,
}

fn default_listen() -> String { "0.0.0.0:4433".into() }
fn default_cc()     -> String { "cubic".into() }
fn default_log()    -> String { "info".into() }
fn default_mode()   -> String { "downlink".into() }

// ──────────────────────────────────── helpers ─────────────────────────────────

/// Spawn `cmd`, pipe both stdout and stderr into `output_buf` (background tasks).
async fn spawn_and_capture(
    mut cmd:    Command,
    output_buf: Arc<Mutex<VecDeque<String>>>,
) -> std::io::Result<Child> {
    use std::process::Stdio;
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn()?;

    let drain = |stream: Option<tokio::process::ChildStdout>, buf: Arc<Mutex<VecDeque<String>>>| {
        if let Some(s) = stream {
            tokio::spawn(async move {
                let mut lines = BufReader::new(s).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    let mut lock = buf.lock().await;
                    if lock.len() >= OUTPUT_CAP { lock.pop_front(); }
                    lock.push_back(line);
                }
            });
        }
    };

    // stdout
    drain(child.stdout.take(), Arc::clone(&output_buf));

    // stderr — tokio uses a different type for ChildStderr; handle separately
    if let Some(stderr) = child.stderr.take() {
        let buf = Arc::clone(&output_buf);
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let mut lock = buf.lock().await;
                if lock.len() >= OUTPUT_CAP { lock.pop_front(); }
                lock.push_back(line);
            }
        });
    }

    Ok(child)
}

// ──────────────────────────────────── handlers ────────────────────────────────

async fn server_start(
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
    for a in &req.extra_args { cmd.arg(a); }

    proc.output.lock().await.clear();
    match spawn_and_capture(cmd, Arc::clone(&proc.output)).await {
        Ok(child) => {
            let pid = child.id();
            proc.child = Some(child);
            proc.pid   = pid;
            Json(serde_json::json!({ "ok": true, "pid": pid }))
        }
        Err(e) => Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
    }
}

async fn server_stop(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let mut proc = state.server.lock().await;
    match proc.stop().await {
        Ok(_)  => Json(serde_json::json!({ "ok": true })),
        Err(e) => Json(serde_json::json!({ "ok": false, "error": e })),
    }
}

async fn server_status(
    State(state): State<Arc<AppState>>,
) -> Json<ProcessStatus> {
    let mut proc = state.server.lock().await;
    Json(proc.status_snapshot().await)
}

// ── client ────────────────────────────────────────────────────────────────────

async fn client_start(
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
    cmd.args(["--connect-to", &req.connect_to]);
    cmd.args(["--congestion-control-algor", &req.congestion_control]);
    cmd.args(["--log-level", &req.log_level]);
    cmd.args(["--mode", &req.mode]);
    for addr in &req.local_addresses {
        cmd.args(["--local-addresses", addr]);
    }
    if let Some(d)  = req.duration         { cmd.args(["--duration",          &d.to_string()]); }
    if let Some(s)  = req.streams_per_conn  { cmd.args(["--streams-per-conn",  &s.to_string()]); }
    if let Some(bw) = &req.bandwidth        { cmd.args(["--bandwidth",          bw]); }
    if req.enable_multipath {
        cmd.arg("--enable-multipath");
        if let Some(algor) = &req.multipath_algor {
            cmd.args(["--multipath-algor", algor]);
        }
    }
    for a in &req.extra_args { cmd.arg(a); }

    proc.output.lock().await.clear();
    match spawn_and_capture(cmd, Arc::clone(&proc.output)).await {
        Ok(child) => {
            let pid = child.id();
            proc.child = Some(child);
            proc.pid   = pid;
            Json(serde_json::json!({ "ok": true, "pid": pid }))
        }
        Err(e) => Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
    }
}

async fn client_stop(
    State(state): State<Arc<AppState>>,
) -> Json<serde_json::Value> {
    let mut proc = state.client.lock().await;
    match proc.stop().await {
        Ok(_)  => Json(serde_json::json!({ "ok": true })),
        Err(e) => Json(serde_json::json!({ "ok": false, "error": e })),
    }
}

async fn client_status(
    State(state): State<Arc<AppState>>,
) -> Json<ProcessStatus> {
    let mut proc = state.client.lock().await;
    Json(proc.status_snapshot().await)
}

// ── combined ──────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct OverallStatus {
    server: ProcessStatus,
    client: ProcessStatus,
}

async fn overall_status(
    State(state): State<Arc<AppState>>,
) -> Json<OverallStatus> {
    let server = state.server.lock().await.status_snapshot().await;
    let client = state.client.lock().await.status_snapshot().await;
    Json(OverallStatus { server, client })
}

// ──────────────────────────────────── main ────────────────────────────────────

#[tokio::main]
async fn main() {
    env_logger::init();

    // Resolve binary directory: same dir as this exe, or TQUIC_BIN_DIR override.
    let bin_dir: PathBuf = env::var("TQUIC_BIN_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            env::current_exe()
                .ok()
                .and_then(|p| p.parent().map(|p| p.to_path_buf()))
                .unwrap_or_else(|| PathBuf::from("."))
        });

    let port: u16 = env::var("TQUIC_API_PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8000);

    let state = Arc::new(AppState {
        server:  Mutex::new(ProcessState::new()),
        client:  Mutex::new(ProcessState::new()),
        bin_dir: bin_dir.clone(),
    });

    let app = Router::new()
        .route("/server/start",  post(server_start))
        .route("/server/stop",   post(server_stop))
        .route("/server/status", get(server_status))
        .route("/client/start",  post(client_start))
        .route("/client/stop",   post(client_stop))
        .route("/client/status", get(client_status))
        .route("/status",        get(overall_status))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    eprintln!("tquic_api  listening on  http://{addr}");
    eprintln!("           binaries from {}", bin_dir.display());

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
