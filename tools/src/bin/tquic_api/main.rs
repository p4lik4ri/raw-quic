//! tquic_api — HTTP control plane for tquic_server and tquic_client.
//!
//! Listens on 0.0.0.0:8000 by default (override with TQUIC_API_PORT).
//! Looks for tquic_server / tquic_client in the same directory as this binary
//! (override with TQUIC_BIN_DIR).
//!
//! Endpoints
//! ─────────────────────────────────────────────────────────────────────────────
//!  POST /server/start            start tquic_server
//!  POST /server/stop             kill running tquic_server
//!  GET  /server/status           pid, running flag, last 1000 log lines
//!  GET  /server/LastJsonResult   last session samples, Python-style plain text
//!
//!  POST /client/start            start tquic_client
//!  POST /client/stop             kill running tquic_client
//!  GET  /client/status           pid, running flag, last 1000 log lines
//!
//!  GET  /status                  both server + client status
//!  GET  /LastJsonResult          last session client samples as JSON

mod handlers;
mod middleware;
mod models;
mod spawn;
mod state;

use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use tokio::sync::Mutex;

use handlers::{
    client_start, client_status, client_stop,
    last_json_result, overall_status,
    server_last_json_result, server_start, server_status, server_stop,
};
use state::{AppState, ProcessState};

#[tokio::main]
async fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

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
        server:      Mutex::new(ProcessState::new()),
        client:      Mutex::new(ProcessState::new()),
        bin_dir:     bin_dir.clone(),
        last_client: Arc::new(Mutex::new(Vec::new())),
    });

    let app = Router::new()
        .route("/server/start",           post(server_start))
        .route("/server/stop",            post(server_stop))
        .route("/server/status",          get(server_status))
        .route("/server/LastJsonResult",  get(server_last_json_result)) //Not need maybe remove it
        .route("/client/start",           post(client_start))
        .route("/client/stop",            post(client_stop))
        .route("/client/status",          get(client_status))
        .route("/status",                 get(overall_status))
        .route("/LastJsonResult",         get(last_json_result))
        .layer(axum::middleware::from_fn(middleware::log_request))
        .with_state(state);

    let addr = format!("0.0.0.0:{port}");
    log::info!("tquic_api  listening on  http://{addr}");
    log::info!("           binaries from {}", bin_dir.display());

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
