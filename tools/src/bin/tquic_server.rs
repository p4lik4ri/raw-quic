// Copyright (c) 2023 The TQUIC Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! A raw-QUIC throughput server (reverse mode, like `iperf3 -R`).
//! HTTP/3 and HTTP/0.9 have been removed.  All QUIC transport features
//! (congestion control, multipath, qlog …) are preserved.
//!
//! The server waits for the client to open a bidirectional stream, then pumps
//! bulk data back on that same stream.

use std::collections::HashMap;
use std::fs::create_dir_all;
use std::fs::File;
use std::net::SocketAddr;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;
use std::time::Instant;

use bytes::Bytes;
use clap::Parser;
use log::*;
use mio::event::Event;
use rustc_hash::FxHashMap;

use tquic::CertCompressionAlgorithm;
use tquic::Config;
use tquic::CongestionControlAlgorithm;
use tquic::Connection;
use tquic::Endpoint;
use tquic::Error;
use tquic::MultipathAlgorithm;
use tquic::PacketInfo;
use tquic::TlsConfig;
use tquic::TransportHandler;
use tquic_tools::CertCompressionAlgorithmArg;
use tquic_tools::QuicSocket;
use tquic_tools::Result;

#[cfg(unix)]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

// ─────────────────────────────── CLI options ─────────────────────────────────

#[derive(Parser, Debug)]
#[clap(name = "server", version = env!("CARGO_PKG_VERSION"))]
pub struct ServerOpt {
    /// Address to listen on.
    #[clap(short, long, default_value = "0.0.0.0:4433", value_name = "ADDR")]
    pub listen: SocketAddr,

    /// TLS certificate in PEM format.
    #[clap(short, long = "cert", default_value = "./cert.crt", value_name = "FILE")]
    pub cert_file: String,

    /// TLS private key in PEM format.
    #[clap(short, long = "key", default_value = "./cert.key", value_name = "FILE")]
    pub key_file: String,

    /// Total bytes to send per stream (0 = unlimited until client closes connection).
    #[clap(long, default_value = "0", value_name = "BYTES", help_heading = "Transfer")]
    pub send_size: usize,

    /// Chunk size for each stream_write call.
    #[clap(long, default_value = "65536", value_name = "BYTES", help_heading = "Transfer")]
    pub chunk_size: usize,

    // ── Protocol ──────────────────────────────────────────────────────────────
    /// Session ticket key.
    #[clap(short, long, default_value = "tquic key", value_name = "STR", help_heading = "Protocol")]
    pub ticket_key: String,

    /// Key for generating address tokens.
    #[clap(long, value_name = "STR", help_heading = "Protocol")]
    pub address_token_key: Option<String>,

    /// Enable certificate compression.
    #[clap(long, value_name = "STR", help_heading = "Protocol")]
    pub certificate_compression: Vec<CertCompressionAlgorithmArg>,

    /// Enable stateless retry.
    #[clap(long, help_heading = "Protocol")]
    pub enable_retry: bool,

    /// Disable stateless reset.
    #[clap(long, help_heading = "Protocol")]
    pub disable_stateless_reset: bool,

    /// Congestion control algorithm.
    #[clap(long, default_value = "BBR", help_heading = "Protocol")]
    pub congestion_control_algor: CongestionControlAlgorithm,

    /// Initial congestion window in packets.
    #[clap(long, default_value = "32", value_name = "NUM", help_heading = "Protocol")]
    pub initial_congestion_window: u64,

    /// Minimum congestion window in packets.
    #[clap(long, default_value = "4", value_name = "NUM", help_heading = "Protocol")]
    pub min_congestion_window: u64,

    /// Enable multipath transport.
    #[clap(short, long, help_heading = "Protocol")]
    pub enable_multipath: bool,

    /// Multipath scheduling algorithm.
    #[clap(short, long, default_value = "MINRTT", help_heading = "Protocol")]
    pub multipath_algor: MultipathAlgorithm,

    /// Set active_connection_id_limit transport parameter.
    #[clap(long, default_value = "2", value_name = "NUM", help_heading = "Protocol")]
    pub active_cid_limit: u64,

    /// Set max_udp_payload_size transport parameter.
    #[clap(long, default_value = "65527", value_name = "NUM", help_heading = "Protocol")]
    pub recv_udp_payload_size: u16,

    /// Set the maximum outgoing UDP payload size.
    #[clap(long, default_value = "1200", value_name = "NUM", help_heading = "Protocol")]
    pub send_udp_payload_size: usize,

    /// Handshake timeout in microseconds.
    #[clap(long, default_value = "10000", value_name = "TIME", help_heading = "Protocol")]
    pub handshake_timeout: u64,

    /// Connection idle timeout in microseconds.
    #[clap(long, default_value = "30000", value_name = "TIME", help_heading = "Protocol")]
    pub idle_timeout: u64,

    /// Initial RTT in milliseconds.
    #[clap(long, default_value = "333", value_name = "TIME", help_heading = "Protocol")]
    pub initial_rtt: u64,

    /// Linear factor for calculating the probe timeout.
    #[clap(long, default_value = "10", value_name = "NUM", help_heading = "Protocol")]
    pub pto_linear_factor: u64,

    /// Upper limit of probe timeout in microseconds.
    #[clap(long, default_value = "10000", value_name = "TIME", help_heading = "Protocol")]
    pub max_pto: u64,

    /// Anti amplification factor.
    #[clap(long, default_value = "3", value_name = "NUM", help_heading = "Protocol")]
    pub anti_amplification_factor: usize,

    /// Length of connection id in bytes.
    #[clap(long, default_value = "8", value_name = "NUM", help_heading = "Protocol")]
    pub cid_len: usize,

    // ── Output ────────────────────────────────────────────────────────────────
    /// Log level (OFF/ERROR/WARN/INFO/DEBUG/TRACE).
    #[clap(long, default_value = "INFO", help_heading = "Output")]
    pub log_level: log::LevelFilter,

    /// Log file path (defaults to stderr).
    #[clap(long, value_name = "FILE", help_heading = "Output")]
    pub log_file: Option<String>,

    /// Save TLS key log into the given file.
    #[clap(long, value_name = "FILE", help_heading = "Output")]
    pub keylog_file: Option<String>,

    /// Save qlog file (<trace_id>.qlog) into the given directory.
    #[clap(long, value_name = "DIR", help_heading = "Output")]
    pub qlog_dir: Option<String>,

    // ── Misc ──────────────────────────────────────────────────────────────────
    /// Batch size for sending packets.
    #[clap(long, default_value = "16", value_name = "NUM", help_heading = "Misc")]
    pub send_batch_size: usize,

    /// Buffer size for disordered 0-RTT packets.
    #[clap(long, default_value = "1000", value_name = "NUM", help_heading = "Misc")]
    pub zerortt_buffer_size: usize,

    /// Disable encryption on 1-RTT packets.
    #[clap(long, help_heading = "Misc")]
    pub disable_encryption: bool,
}

// ─────────────────────────── Transfer direction ─────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum TransferMode {
    #[default]
    Downlink, // Server → Client
    Uplink,   // Client → Server
}

// ─────────────────────────── Per-stream send state ───────────────────────────

struct StreamSendState {
    mode: TransferMode,
    bytes_sent: usize,
    finished: bool,
    /// True once the client trigger (bandwidth header) has been received.
    ready: bool,
    /// Accumulates the 8 trigger bytes sent by the client.
    trigger_buf: Vec<u8>,
    /// Rate limit in bytes/sec (0 = unlimited).
    bandwidth_limit: u64,
    /// Token-bucket: available bytes we may send right now.
    tokens: f64,
    /// When the token bucket was last refilled.
    last_refill: Instant,
}

// ─────────────────────────── Per-connection handler ──────────────────────────

#[derive(Default)]
struct ConnectionHandler {
    /// Total bytes to send per stream (0 = unlimited).
    send_size: usize,
    streams: HashMap<u64, StreamSendState>,
    /// Snapshot of conn.stats().sent_bytes from the previous reporter update.
    prev_bytes_sent: u64,
    /// Snapshot of conn.stats().recv_bytes for uplink tracking.
    prev_bytes_recv: u64,
    /// Last time on_stream_readable fired (for server-side jitter in uplink).
    last_recv_time: Option<Instant>,
    /// Previous inter-arrival interval in ms (for deviation calculation).
    prev_interval_ms: f64,
    /// RFC 3550-style running jitter (mean deviation of inter-arrival intervals).
    jitter_ms: f64,
}

impl ConnectionHandler {
    /// Register a new stream; pumping is deferred until the client trigger arrives.
    fn register_stream(&mut self, stream_id: u64) {
        self.streams.insert(
            stream_id,
            StreamSendState {
                mode: TransferMode::Downlink,
                bytes_sent: 0,
                finished: false,
                ready: false,
                trigger_buf: Vec::new(),
                bandwidth_limit: 0,
                tokens: 0.0,
                last_refill: Instant::now(),
            },
        );
    }

    /// Read the 8-byte bandwidth trigger from the client.
    /// Returns true the first time the trigger is fully received so the caller
    /// can kick off pumping.
    fn parse_trigger(&mut self, conn: &mut Connection, stream_id: u64) -> bool {
        let state = match self.streams.get_mut(&stream_id) {
            Some(s) => s,
            None => return false,
        };
        if state.ready {
            return false;
        }
        let mut tmp = [0u8; 64];
        loop {
            match conn.stream_read(stream_id, &mut tmp) {
                Ok((n, fin)) => {
                    if n > 0 {
                        state.trigger_buf.extend_from_slice(&tmp[..n]);
                    }
                    let have_header = state.trigger_buf.len() >= 9;
                    if have_header || fin {
                        let (mode, bw) = if have_header {
                            let m = if state.trigger_buf[0] == 1 {
                                TransferMode::Uplink
                            } else {
                                TransferMode::Downlink
                            };
                            let arr: [u8; 8] = state.trigger_buf[1..9].try_into().unwrap();
                            (m, u64::from_le_bytes(arr))
                        } else {
                            (TransferMode::Downlink, 0) // legacy / FIN with no data
                        };
                        state.mode = mode;
                        state.bandwidth_limit = bw;
                        state.tokens = 0.0; // start empty; first refill happens in pump()
                        state.last_refill = Instant::now();
                        state.ready = true;
                        return true;
                    }
                    if n == 0 { break; }
                }
                Err(Error::Done) => break,
                Err(e) => {
                    error!("{} trigger read: {:?}", conn.trace_id(), e);
                    break;
                }
            }
        }
        false
    }

    /// Push as many bytes as possible; applies token-bucket rate limiting when
    /// `bandwidth_limit > 0`.  Registers `stream_want_write` on backpressure.
    fn pump(&mut self, conn: &mut Connection, stream_id: u64, buf: &[u8]) {
        let state = match self.streams.get_mut(&stream_id) {
            Some(s) => s,
            None => return,
        };

        if state.finished || !state.ready {
            return;
        }

        loop {
            // ── Token-bucket rate limiting ────────────────────────────────────
            if state.bandwidth_limit > 0 {
                let now = Instant::now();
                let elapsed = now.duration_since(state.last_refill).as_secs_f64();
                // Refill; cap at one second's worth of tokens (burst limit).
                state.tokens = (state.tokens + elapsed * state.bandwidth_limit as f64)
                    .min(state.bandwidth_limit as f64);
                state.last_refill = now;

                if state.tokens < 1.0 {
                    // No tokens yet – re-register and wait for the next callback.
                    _ = conn.stream_want_write(stream_id, true);
                    return;
                }
            }

            // ── Determine chunk to write ──────────────────────────────────────
            let to_send = if self.send_size > 0 {
                let remaining = self.send_size.saturating_sub(state.bytes_sent);
                if remaining == 0 {
                    match conn.stream_write(stream_id, Bytes::new(), true) {
                        Ok(_) | Err(Error::Done) => {}
                        Err(e) => error!("{} stream FIN: {:?}", conn.trace_id(), e),
                    }
                    state.finished = true;
                    return;
                }
                remaining.min(buf.len())
            } else {
                buf.len()
            };

            // Honour token budget.
            let to_send = if state.bandwidth_limit > 0 {
                to_send.min(state.tokens as usize).max(1)
            } else {
                to_send
            };

            let fin = self.send_size > 0 && (state.bytes_sent + to_send >= self.send_size);

            match conn.stream_write(
                stream_id,
                Bytes::copy_from_slice(&buf[..to_send]),
                fin,
            ) {
                Ok(written) => {
                    state.bytes_sent += written;
                    if state.bandwidth_limit > 0 {
                        state.tokens -= written as f64;
                    }
                    if fin && written == to_send {
                        state.finished = true;
                        return;
                    }
                    if written < to_send {
                        _ = conn.stream_want_write(stream_id, true);
                        return;
                    }
                }
                Err(Error::Done) => {
                    _ = conn.stream_want_write(stream_id, true);
                    return;
                }
                Err(e) => {
                    error!("{} stream {} write: {:?}", conn.trace_id(), stream_id, e);
                    return;
                }
            }
        }
    }

    /// Drain incoming data on an uplink stream (client → server). Data is discarded.
    fn drain_uplink(&mut self, conn: &mut Connection, stream_id: u64) {
        let should_drain = self.streams.get(&stream_id)
            .map(|s| s.ready && s.mode == TransferMode::Uplink)
            .unwrap_or(false);
        if !should_drain {
            return;
        }
        let mut tmp = [0u8; 65536];
        loop {
            match conn.stream_read(stream_id, &mut tmp) {
                Ok((0, _)) | Err(Error::Done) => break,
                Ok(_) => {}
                Err(e) => {
                    error!("{} uplink drain {}: {:?}", conn.trace_id(), stream_id, e);
                    break;
                }
            }
        }
    }
}

// ─────────────────────────────── ServerHandler ───────────────────────────────

struct ServerHandler {
    conns: FxHashMap<u64, ConnectionHandler>,
    /// Zero-filled send buffer.
    send_buf: Vec<u8>,
    send_size: usize,
    keylog: Option<File>,
    qlog_dir: Option<String>,
    /// Shared live counters for the interval reporter thread.
    live_bytes:  Arc<AtomicU64>,
    live_lost:   Arc<AtomicU64>,
    live_sent:   Arc<AtomicU64>,
    live_jitter: Arc<AtomicU64>,
    /// True when the active transfer is uplink (client→server).
    is_uplink: Arc<AtomicBool>,
    /// Signals the reporter thread to stop (set on connection close).
    rep_done: Arc<AtomicBool>,
}

impl ServerHandler {
    fn new(
        option: &ServerOpt,
        live_bytes:  Arc<AtomicU64>,
        live_lost:   Arc<AtomicU64>,
        live_sent:   Arc<AtomicU64>,
        live_jitter: Arc<AtomicU64>,
        is_uplink:   Arc<AtomicBool>,
        rep_done:    Arc<AtomicBool>,
    ) -> Result<Self> {
        let keylog = match &option.keylog_file {
            Some(f) => Some(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(f)?,
            ),
            None => None,
        };

        Ok(Self {
            conns: FxHashMap::default(),
            send_buf: vec![0u8; option.chunk_size],
            send_size: option.send_size,
            keylog,
            qlog_dir: option.qlog_dir.clone(),
            live_bytes,
            live_lost,
            live_sent,
            live_jitter,
            is_uplink,
            rep_done,
        })
    }

    fn ensure_conn_handler(&mut self, conn: &mut Connection) {
        let idx = conn.index().unwrap();
        if self.conns.contains_key(&idx) {
            return;
        }
        self.conns.insert(
            idx,
            ConnectionHandler { send_size: self.send_size, ..Default::default() },
        );
    }
}

impl TransportHandler for ServerHandler {
    fn on_conn_created(&mut self, conn: &mut Connection) {
        debug!("{} connection created", conn.trace_id());

        if let Some(keylog) = &mut self.keylog {
            if let Ok(kl) = keylog.try_clone() {
                conn.set_keylog(Box::new(kl));
            }
        }

        if let Some(qlog_dir) = &self.qlog_dir {
            let path = Path::new(qlog_dir).join(format!("{}.qlog", conn.trace_id()));
            if let Ok(f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                conn.set_qlog(
                    Box::new(f),
                    "server qlog".into(),
                    format!("id={}", conn.trace_id()),
                );
            } else {
                error!("{} set qlog {:?} failed", conn.trace_id(), path);
            }
        }
    }

    fn on_conn_established(&mut self, conn: &mut Connection) {
        debug!("{} connection established", conn.trace_id());
        self.ensure_conn_handler(conn);
    }

    fn on_conn_closed(&mut self, conn: &mut Connection) {
        let s = conn.stats();
        info!(
            "{} connection closed — recv pkts/bytes: {}/{}, sent pkts/bytes: {}/{}, lost pkts/bytes: {}/{}",
            conn.trace_id(),
            s.recv_count, s.recv_bytes,
            s.sent_count, s.sent_bytes,
            s.lost_count, s.lost_bytes,
        );
        self.conns.remove(&conn.index().unwrap());
        // Stop the interval reporter as soon as the transfer finishes.
        self.rep_done.store(true, Ordering::Relaxed);
    }

    fn on_stream_created(&mut self, conn: &mut Connection, stream_id: u64) {
        debug!("{} stream {} created", conn.trace_id(), stream_id);

        // 0-RTT: stream may arrive before on_conn_established.
        self.ensure_conn_handler(conn);

        // Only handle client-initiated streams (even IDs: bidi 0,4,8,…; uni 2,6,10,…).
        if stream_id % 2 == 1 {
            return;
        }

        let idx = conn.index().unwrap();
        if let Some(handler) = self.conns.get_mut(&idx) {
            handler.register_stream(stream_id);
        }
    }

    fn on_stream_readable(&mut self, conn: &mut Connection, stream_id: u64) {
        // Only even-ID streams are client-initiated.
        if stream_id % 2 == 1 {
            return;
        }
        let idx = conn.index().unwrap();
        // Phase 1: parse the 9-byte trigger ([mode:u8][bandwidth:u64 LE]).
        let just_ready = {
            if let Some(handler) = self.conns.get_mut(&idx) {
                handler.parse_trigger(conn, stream_id)
            } else {
                false
            }
        };
        if just_ready {
            let mode = self.conns.get(&idx)
                .and_then(|h| h.streams.get(&stream_id))
                .map(|s| s.mode)
                .unwrap_or(TransferMode::Downlink);
            match mode {
                TransferMode::Downlink => {
                    let send_buf = self.send_buf.clone();
                    if let Some(handler) = self.conns.get_mut(&idx) {
                        handler.pump(conn, stream_id, &send_buf);
                    }
                }
                TransferMode::Uplink => {
                    self.is_uplink.store(true, Ordering::Relaxed);
                    if let Some(handler) = self.conns.get_mut(&idx) {
                        handler.drain_uplink(conn, stream_id);
                    }
                }
            }
        } else {
            if let Some(handler) = self.conns.get_mut(&idx) {
                handler.drain_uplink(conn, stream_id);
            }
        }
        // Update live uplink stats for the interval reporter.
        if self.is_uplink.load(Ordering::Relaxed) {
            if let Some(handler) = self.conns.get_mut(&idx) {
                let stats = conn.stats();
                let now   = Instant::now();
                let delta = stats.recv_bytes.saturating_sub(handler.prev_bytes_recv);
                if delta > 0 {
                    handler.prev_bytes_recv = stats.recv_bytes;
                    self.live_bytes.fetch_add(delta, Ordering::Relaxed);
                    if let Some(last) = handler.last_recv_time {
                        let actual_ms = now.duration_since(last).as_secs_f64() * 1000.0;
                        // Jitter = mean deviation of inter-arrival intervals (RFC 3550 §A.8 style)
                        if handler.prev_interval_ms > 0.0 {
                            let deviation = (actual_ms - handler.prev_interval_ms).abs();
                            handler.jitter_ms += (deviation - handler.jitter_ms) / 16.0;
                            self.live_jitter.store(handler.jitter_ms.to_bits(), Ordering::Relaxed);
                        }
                        handler.prev_interval_ms = actual_ms;
                    }
                    handler.last_recv_time = Some(now);
                }
                // For uplink server: "sent" = total datagrams received by server.
                self.live_sent.store(stats.recv_count, Ordering::Relaxed);
                self.live_lost.store(stats.lost_count, Ordering::Relaxed);
            }
        }
    }

    fn on_stream_writable(&mut self, conn: &mut Connection, stream_id: u64) {
        _ = conn.stream_want_write(stream_id, false);
        let idx = conn.index().unwrap();
        let is_uplink_stream = self.conns.get(&idx)
            .and_then(|h| h.streams.get(&stream_id))
            .map(|s| s.mode == TransferMode::Uplink)
            .unwrap_or(false);
        if is_uplink_stream {
            return;
        }
        let send_buf = self.send_buf.clone();
        if let Some(handler) = self.conns.get_mut(&idx) {
            handler.pump(conn, stream_id, &send_buf);
            // Update live downlink stats for the interval reporter.
            let stats = conn.stats();
            let delta = stats.sent_bytes.saturating_sub(handler.prev_bytes_sent);
            if delta > 0 {
                handler.prev_bytes_sent = stats.sent_bytes;
                self.live_bytes.fetch_add(delta, Ordering::Relaxed);
            }
            self.live_sent.store(stats.sent_count, Ordering::Relaxed);
            self.live_lost.store(stats.lost_count, Ordering::Relaxed);
        }
    }

    fn on_stream_closed(&mut self, conn: &mut Connection, stream_id: u64) {
        debug!("{} stream {} closed", conn.trace_id(), stream_id);
        if let Some(handler) = self.conns.get_mut(&conn.index().unwrap()) {
            handler.streams.remove(&stream_id);
        }
    }

    fn on_new_token(&mut self, _conn: &mut Connection, _token: Vec<u8>) {}
}

// ─────────────────────────────── Server event loop ───────────────────────────

const MAX_BUF_SIZE: usize = 65536;

struct Server {
    endpoint: Endpoint,
    poll: mio::Poll,
    sock: Rc<QuicSocket>,
    recv_buf: Vec<u8>,
}

impl Server {
    fn new(
        option: &ServerOpt,
        live_bytes:  Arc<AtomicU64>,
        live_lost:   Arc<AtomicU64>,
        live_sent:   Arc<AtomicU64>,
        live_jitter: Arc<AtomicU64>,
        is_uplink:   Arc<AtomicBool>,
        rep_done:    Arc<AtomicBool>,
    ) -> Result<Self> {
        let mut config = Config::new()?;
        config.set_recv_udp_payload_size(option.recv_udp_payload_size);
        config.set_send_udp_payload_size(option.send_udp_payload_size);
        config.set_max_handshake_timeout(option.handshake_timeout);
        config.enable_retry(option.enable_retry);
        config.enable_stateless_reset(!option.disable_stateless_reset);
        config.set_max_idle_timeout(option.idle_timeout);
        config.set_initial_rtt(option.initial_rtt);
        config.set_pto_linear_factor(option.pto_linear_factor);
        config.set_max_pto(option.max_pto);
        config.set_cid_len(option.cid_len);
        config.set_anti_amplification_factor(option.anti_amplification_factor);
        config.set_send_batch_size(option.send_batch_size);
        config.set_zerortt_buffer_size(option.zerortt_buffer_size);
        config.set_congestion_control_algorithm(option.congestion_control_algor);
        config.set_initial_congestion_window(option.initial_congestion_window);
        config.set_min_congestion_window(option.min_congestion_window);
        config.enable_multipath(option.enable_multipath);
        config.set_multipath_algorithm(option.multipath_algor);
        config.set_active_connection_id_limit(option.active_cid_limit);
        config.enable_encryption(!option.disable_encryption);

        if let Some(ak) = &option.address_token_key {
            config.set_address_token_key(vec![convert_address_token_key(ak)])?;
        }

        let mut tls_config = TlsConfig::new_server_config(
            &option.cert_file,
            &option.key_file,
            vec![b"rawquic".to_vec()],
            true,
        )?;
        let mut ticket_key = option.ticket_key.clone().into_bytes();
        ticket_key.resize(48, 0);
        tls_config.set_ticket_key(&ticket_key)?;

        if !option.certificate_compression.is_empty() {
            let algs: Vec<CertCompressionAlgorithm> =
                option.certificate_compression.iter().map(|&a| a.into()).collect();
            tls_config.enable_certificate_compression(algs)?;
        }

        config.set_tls_config(tls_config);

        let poll = mio::Poll::new()?;
        let handler = ServerHandler::new(option, live_bytes, live_lost, live_sent, live_jitter, is_uplink, rep_done)?;
        let sock = Rc::new(QuicSocket::new(&option.listen, poll.registry())?);

        Ok(Server {
            endpoint: Endpoint::new(Box::new(config), true, Box::new(handler), sock.clone()),
            poll,
            sock,
            recv_buf: vec![0u8; MAX_BUF_SIZE],
        })
    }

    fn process_read_event(&mut self, event: &Event) -> Result<()> {
        loop {
            let (len, local, remote) =
                match self.sock.recv_from(&mut self.recv_buf, event.token()) {
                    Ok(v) => v,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        debug!("socket recv would block");
                        break;
                    }
                    Err(e) => return Err(format!("socket recv error: {:?}", e).into()),
                };
            debug!("socket recv {} bytes from {:?}", len, remote);

            let pkt_info = PacketInfo {
                src: remote,
                dst: local,
                time: Instant::now(),
            };
            if let Err(e) = self.endpoint.recv(&mut self.recv_buf[..len], &pkt_info) {
                error!("endpoint recv error: {:?}", e);
            }
        }
        Ok(())
    }
}

fn convert_address_token_key(key: &str) -> [u8; 16] {
    let mut kd = key.to_owned().into_bytes();
    kd.resize(16, 0);
    let mut tk = [0u8; 16];
    tk.copy_from_slice(&kd[..]);
    tk
}

fn process_option(option: &mut ServerOpt) -> Result<()> {
    env_logger::builder()
        .target(tquic_tools::log_target(&option.log_file)?)
        .filter_level(option.log_level)
        .format_timestamp_millis()
        .init();

    if let Some(qlog_dir) = &option.qlog_dir {
        create_dir_all(qlog_dir)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut option = ServerOpt::parse();
    process_option(&mut option)?;

    // Shared live counters for the interval reporter.
    let live_bytes   = Arc::new(AtomicU64::new(0));
    let live_lost    = Arc::new(AtomicU64::new(0));
    let live_sent    = Arc::new(AtomicU64::new(0));
    let live_jitter  = Arc::new(AtomicU64::new(0));
    let is_uplink    = Arc::new(AtomicBool::new(false));
    let rep_done     = Arc::new(AtomicBool::new(false));
    let terminated   = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&terminated))?;

    let mut server = Server::new(
        &option,
        Arc::clone(&live_bytes), Arc::clone(&live_lost),
        Arc::clone(&live_sent),  Arc::clone(&live_jitter),
        Arc::clone(&is_uplink),  Arc::clone(&rep_done),
    )?;

    info!(
        "{} listening on {:?}  send_size={} chunk={} CC={:?} multipath={}",
        server.endpoint.trace_id(),
        option.listen,
        option.send_size,
        option.chunk_size,
        option.congestion_control_algor,
        option.enable_multipath,
    );

    // ── Interval reporter thread ───────────────────────────────────────────────
    let rb = Arc::clone(&live_bytes);
    let rl = Arc::clone(&live_lost);
    let rs = Arc::clone(&live_sent);
    let rj = Arc::clone(&live_jitter);
    let ru = Arc::clone(&is_uplink);
    let rd = Arc::clone(&rep_done);
    let rt = Arc::clone(&terminated);
    let reporter = thread::spawn(move || {
        'session: loop {
            // ── Wait for a new session (data to start flowing) ────────────────
            loop {
                if rt.load(Ordering::Relaxed) { return; }
                thread::sleep(Duration::from_millis(200));
                if rb.load(Ordering::Relaxed) > 0 { break; }
            }

            // ── Per-session header ────────────────────────────────────────────
            println!();
            println!("[ rawquic ] Server interval report");
            println!(
                "  {:<12}  {:>10}  {:>16}  {:>10}  {}",
                "Interval", "Transfer", "Bitrate", "Jitter", "Lost/Total Datagrams"
            );
            let mut last_bytes: u64 = 0;
            let mut last_lost:  u64 = 0;
            let mut last_sent:  u64 = 0;
            let mut interval:   u64 = 0;

            // ── Interval loop ─────────────────────────────────────────────────
            loop {
                thread::sleep(Duration::from_secs(1));
                let done      = rd.load(Ordering::Relaxed);
                let current   = rb.load(Ordering::Relaxed);
                let cur_lost  = rl.load(Ordering::Relaxed);
                let cur_sent  = rs.load(Ordering::Relaxed);
                let jitter_ms = f64::from_bits(rj.load(Ordering::Relaxed));
                let delta  = current.saturating_sub(last_bytes);
                let d_lost = cur_lost.saturating_sub(last_lost);
                let d_sent = cur_sent.saturating_sub(last_sent);
                last_bytes = current; last_lost = cur_lost; last_sent = cur_sent;
                let t_start  = interval as f64;
                let t_end    = interval as f64 + 1.0;
                interval    += 1;
                let mb       = delta as f64 / 1e6;
                let mbps     = (delta as f64 * 8.0) / 1e6;
                let loss_pct = if d_sent > 0 { d_lost as f64 / d_sent as f64 * 100.0 } else { 0.0 };
                println!(
                    "  {:<12}  {:>10}  {:>16}  {:>13}  {}/{} ({:.0}%)",
                    format!("{:.2}-{:.2} s", t_start, t_end),
                    format!("{:.2} MB", mb),
                    format!("{:.2} Mbits/sec", mbps),
                    format!("{:.3} ms", jitter_ms),
                    d_lost, d_sent, loss_pct,
                );
                if rt.load(Ordering::Relaxed) { break 'session; }
                if done { break; }
            }

            // ── Summary row ───────────────────────────────────────────────────
            let current   = rb.load(Ordering::Relaxed);
            let cur_lost  = rl.load(Ordering::Relaxed);
            let cur_sent  = rs.load(Ordering::Relaxed);
            let jitter_ms = f64::from_bits(rj.load(Ordering::Relaxed));
            if current > 0 {
                let uplink     = ru.load(Ordering::Relaxed);
                let total_secs = interval.max(1) as f64;
                let total_mb   = current as f64 / 1e6;
                let total_mbps = (current as f64 * 8.0) / 1e6 / total_secs;
                let loss_pct   = if cur_sent > 0 { cur_lost as f64 / cur_sent as f64 * 100.0 } else { 0.0 };
                let role = if uplink { "receiver" } else { "sender" };
                println!("- - - - - - - - - - - - - - - - - - - - - - - - -");
                println!(
                    "  {:<12}  {:>10}  {:>16}  {:>10}  {}",
                    "Interval", "Transfer", "Bitrate", "Jitter", "Lost/Total Datagrams"
                );
                println!(
                    "  {:<12}  {:>10}  {:>16}  {:>13}  {}/{} ({:.0}%)  {}",
                    format!("0.00-{:.2} s", total_secs),
                    format!("{:.2} MB", total_mb),
                    format!("{:.2} Mbits/sec", total_mbps),
                    format!("{:.3} ms", jitter_ms),
                    cur_lost, cur_sent, loss_pct, role,
                );
            }

            // ── Reset all counters for the next session ───────────────────────
            rb.store(0, Ordering::Relaxed);
            rl.store(0, Ordering::Relaxed);
            rs.store(0, Ordering::Relaxed);
            rj.store(0u64, Ordering::Relaxed);
            ru.store(false, Ordering::Relaxed);
            rd.store(false, Ordering::Relaxed);
        }
    });

    // ── Main event loop ─────────────────────────────────────────────────────────
    let mut events = mio::Events::with_capacity(1024);
    loop {
        if terminated.load(Ordering::Relaxed) {
            rep_done.store(true, Ordering::Relaxed);
            reporter.join().unwrap();
            break;
        }
        if let Err(e) = server.endpoint.process_connections() {
            error!("process_connections: {:?}", e);
        }
        server.poll.poll(&mut events, server.endpoint.timeout())?;
        for event in events.iter() {
            if event.is_readable() {
                server.process_read_event(event)?;
            }
        }
        server.endpoint.on_timeout(Instant::now());
    }
    Ok(())
}
