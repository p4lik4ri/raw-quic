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

    /// Initial congestion window in packets [default: 32]
    pub initial_congestion_window: Option<u64>,

    /// Minimum congestion window in packets [default: 4]
    pub min_congestion_window: Option<u64>,

    #[serde(default = "default_log")]
    pub log_level: String,

    /// Log file path. If omitted, logs go to stderr.
    pub log_file: Option<String>,

    /// Save qlog file into the given directory.
    pub qlog_dir: Option<String>,

    #[serde(default)]
    pub enable_multipath: bool,

    /// RoundRobin / MinRTT / Redundant
    pub multipath_algor: Option<String>,

    /// Maximum outgoing UDP payload size in bytes [default: 1200].
    /// Lower this (e.g. 1100) when packets are dropped through intermediate VMs/NATs.
    pub send_udp_payload_size: Option<u64>,

    /// max_udp_payload_size transport parameter [default: 65527]
    pub recv_udp_payload_size: Option<u64>,

    /// Handshake timeout in microseconds [default: 10000]
    pub handshake_timeout: Option<u64>,

    /// Connection idle timeout in microseconds [default: 30000]
    pub idle_timeout: Option<u64>,

    /// Initial RTT estimate in milliseconds [default: 333].
    /// Increase for high-latency multi-hop paths.
    pub initial_rtt: Option<u64>,

    /// Linear factor for probe timeout calculation [default: 10]
    pub pto_linear_factor: Option<u64>,

    /// Upper limit of probe timeout in microseconds [default: 10000]
    pub max_pto: Option<u64>,

    /// Anti-amplification factor [default: 3]
    pub anti_amplification_factor: Option<u64>,

    /// Batch size for sending packets [default: 16]
    pub send_batch_size: Option<u64>,

    /// Enable stateless retry
    #[serde(default)]
    pub enable_retry: bool,

    /// Buffer size for disordered 0-RTT packets on the server [default: 1000]
    pub zerortt_buffer_size: Option<u64>,

    /// Disable encryption on 1-RTT packets (testing only)
    #[serde(default)]
    pub disable_encryption: bool,

    /// Working directory for the child process (cert/key paths resolved from here).
    pub work_dir: Option<String>,

    /// Path to write TLS key material for decryption (passed to --keylog-file).
    pub keylog_file: Option<String>,

    /// Extra raw CLI flags forwarded verbatim, e.g. ["--active-cid-limit","4"].
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

    /// Initial congestion window in packets [default: 32]
    pub initial_congestion_window: Option<u64>,

    /// Minimum congestion window in packets [default: 4]
    pub min_congestion_window: Option<u64>,

    #[serde(default = "default_log")]
    pub log_level: String,

    /// Log file path. If omitted, logs go to stderr.
    pub log_file: Option<String>,

    /// Save qlog file into the given directory.
    pub qlog_dir: Option<String>,

    #[serde(default)]
    pub enable_multipath: bool,

    pub multipath_algor: Option<String>,

    /// Maximum outgoing UDP payload size in bytes [default: 1200].
    /// Lower this (e.g. 1100) when packets are dropped through intermediate VMs/NATs.
    pub send_udp_payload_size: Option<u64>,

    /// max_udp_payload_size transport parameter [default: 65527]
    pub recv_udp_payload_size: Option<u64>,

    /// Handshake timeout in microseconds [default: 10000]
    pub handshake_timeout: Option<u64>,

    /// Connection idle timeout in microseconds [default: 30000]
    pub idle_timeout: Option<u64>,

    /// Initial RTT estimate in milliseconds [default: 333].
    /// Increase for high-latency multi-hop paths.
    pub initial_rtt: Option<u64>,

    /// Linear factor for probe timeout calculation [default: 10]
    pub pto_linear_factor: Option<u64>,

    /// Upper limit of probe timeout in microseconds [default: 10000]
    pub max_pto: Option<u64>,

    /// Batch size for sending packets [default: 1]
    pub send_batch_size: Option<u64>,

    /// Number of threads [default: 1]
    pub threads: Option<u64>,

    /// Number of concurrent connections per thread [default: 1]
    pub max_concurrent_conns: Option<u64>,

    /// Number of concurrent requests per connection [default: 1]
    pub max_concurrent_requests: Option<u64>,

    /// Path to write TLS key material for decryption.
    pub keylog_file: Option<String>,

    /// Disable encryption on 1-RTT packets (testing only)
    #[serde(default)]
    pub disable_encryption: bool,

    /// Working directory for the child process.
    pub work_dir: Option<String>,

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
