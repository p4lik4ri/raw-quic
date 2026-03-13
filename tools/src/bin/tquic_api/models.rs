//! Request and response data models for the tquic_api HTTP control plane.
//!
//! Defines the JSON-deserialisable payloads accepted by the `/server/start` and
//! `/client/start` endpoints (`ServerStartRequest`, `ClientStartRequest`), as
//! well as the `OverallStatus` response that aggregates server + client state.

use serde::{Deserialize, Serialize};

use crate::state::ProcessStatus;

// ─────────────────────────────────── server ───────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ServerStartRequest {
    pub cert: String,
    pub key:  String,

    #[serde(default = "default_listen")]
    pub listen: String,

    /// BBR / BBR3 / Cubic  (passed straight to --congestion-control-algor)
    #[serde(default = "default_cc")]
    pub congestion_control: String,

    #[serde(default = "default_log")]
    pub log_level: String,

    #[serde(default)]
    pub enable_multipath: bool,

    /// RoundRobin / MinRTT / Redundant
    pub multipath_algor: Option<String>,

    /// Working directory for the child process (cert/key paths resolved from here).
    pub work_dir: Option<String>,

    /// Path to write TLS key material for decryption (passed to --keylog-file).
    pub keylog_file: Option<String>,

    /// Extra raw CLI flags forwarded verbatim, e.g. ["--send-batch-size","16"].
    #[serde(default)]
    pub extra_args: Vec<String>,
}

// ─────────────────────────────────── client ───────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ClientStartRequest {
    pub connect_to: String,

    #[serde(default)]
    pub local_addresses: Vec<String>,

    pub duration:         Option<u64>,
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

    /// Working directory for the child process.
    pub work_dir: Option<String>,

    /// URL of the remote tquic_api instance running on the server machine,
    /// e.g. "http://192.168.1.5:8000".  When set, /LastJsonResult will fetch
    /// /server/intervals from that URL to overlay real jitter values onto the
    /// client samples (necessary in uplink mode when the two APIs run on
    /// different hosts and cannot share in-process state).
    pub server_api_url: Option<String>,

    #[serde(default)]
    pub extra_args: Vec<String>,
}

// ─────────────────────────────────── response ─────────────────────────────────

#[derive(Serialize)]
pub struct OverallStatus {
    pub server: ProcessStatus,
    pub client: ProcessStatus,
}

// ─────────────────────────────────── defaults ─────────────────────────────────

pub fn default_listen() -> String { "0.0.0.0:4433".into() }
pub fn default_cc()     -> String { "cubic".into() }
pub fn default_log()    -> String { "info".into() }
pub fn default_mode()   -> String { "downlink".into() }
