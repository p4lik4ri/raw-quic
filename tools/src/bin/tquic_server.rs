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
use std::io::Write;
use std::rc::Rc;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

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
use tquic_tools::wandb_logger::WandbLogger;

#[cfg(unix)]
#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

/// Size of one jitter-measurement datagram in the stream:
/// 8-byte send-timestamp header followed by (DATAGRAM_STRIDE − 8) bytes of zeros.
/// Must match the constant in tquic_client.rs.
const DATAGRAM_STRIDE: usize = 1400;

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

// ─────────────────────── Trigger / stats-reply codec (pure) ──────────────────

/// Result of decoding a complete client trigger message.
#[derive(Debug, PartialEq)]
enum TriggerDecision {
    /// Client sent 0xFF + FIN: requesting server-side stats after a downlink run.
    StatsRequest,
    /// Normal transfer trigger: carries the desired direction and bandwidth cap.
    Transfer { mode: TransferMode, bandwidth: u64 },
}

/// Decode the accumulated trigger bytes `buf` given whether the stream FIN
/// has been seen.  Returns `None` when more bytes are still needed.
fn decode_trigger_buf(buf: &[u8], fin: bool) -> Option<TriggerDecision> {
    // Stats request: single byte 0xFF + FIN
    if buf.first() == Some(&0xFF) && fin {
        return Some(TriggerDecision::StatsRequest);
    }
    let have_header = buf.len() >= 9;
    if have_header || fin {
        if have_header {
            let mode = if buf[0] == 1 { TransferMode::Uplink } else { TransferMode::Downlink };
            let arr: [u8; 8] = buf[1..9].try_into().unwrap();
            Some(TriggerDecision::Transfer { mode, bandwidth: u64::from_le_bytes(arr) })
        } else {
            // FIN arrived with fewer than 9 bytes: treat as downlink, unlimited.
            Some(TriggerDecision::Transfer { mode: TransferMode::Downlink, bandwidth: 0 })
        }
    } else {
        None
    }
}

/// Build the 16-byte reply sent back on an uplink stream after FIN:
/// `[jitter_ms as f64 bits: u64 LE][server_recv_count: u64 LE]`.
/// The second field is the server's received packet count so the client can
/// compute receiver loss as (client_sent_count − server_recv_count).
fn encode_uplink_stats_reply(jitter_bits: u64, recv_count: u64) -> [u8; 16] {
    let mut reply = [0u8; 16];
    reply[0..8].copy_from_slice(&jitter_bits.to_le_bytes());
    reply[8..16].copy_from_slice(&recv_count.to_le_bytes());
    reply
}

/// Build the 16-byte reply sent back on a downlink stats-request stream:
/// `[lost_count: u64 LE][sent_count: u64 LE]`.
fn encode_downlink_stats_reply(lost_count: u64, sent_count: u64) -> [u8; 16] {
    let mut reply = [0u8; 16];
    reply[0..8].copy_from_slice(&lost_count.to_le_bytes());
    reply[8..16].copy_from_slice(&sent_count.to_le_bytes());
    reply
}

// ─────────────────────────── Per-stream send state ───────────────────────────

struct StreamSendState {
    mode: TransferMode,
    bytes_sent: usize,
    finished: bool,
    /// True once the client trigger (bandwidth header) has been received.
    ready: bool,
    /// Accumulates the trigger bytes sent by the client.
    trigger_buf: Vec<u8>,
    /// Rate limit in bytes/sec (0 = unlimited).
    bandwidth_limit: u64,
    /// Token-bucket: available bytes we may send right now.
    tokens: f64,
    /// When the token bucket was last refilled.
    last_refill: Instant,
    /// True once we have written the 16-byte stats reply back to the client.
    reply_sent: bool,
    /// True if this stream is a stats-request stream (client sent 0xFF).
    is_stats_request: bool,
    /// Staging buffer for the current datagram (DATAGRAM_STRIDE bytes).
    /// We always build a full datagram before the first stream_write so that
    /// QUIC flow-control backpressure never produces a partial write at a
    /// datagram boundary.
    dg_buf: Vec<u8>,
    /// How many bytes of dg_buf have already been written to the stream.
    dg_buf_pos: usize,
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
    // ─ Uplink jitter — per-datagram timestamp parser (mirrors client downlink) ─
    /// Position within the current DATAGRAM_STRIDE block on the receive side.
    dg_pos: usize,
    /// Accumulates the 8-byte send-timestamp header of the datagram being parsed.
    ts_partial_buf: [u8; 8],
    /// Send timestamp (µs) of the most recently completed datagram header.
    last_send_us: Option<u64>,
    /// Wall-clock receive time (µs since UNIX epoch) of that datagram.
    last_recv_us: Option<u64>,
    /// RFC 3550 running jitter from per-datagram one-way delay variation.
    jitter_ms: f64,
    /// Application-level datagrams received (completed DATAGRAM_STRIDE blocks).
    app_datagrams_received: u64,
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
                reply_sent: false,
                is_stats_request: false,
                dg_buf: vec![0u8; DATAGRAM_STRIDE],
                dg_buf_pos: DATAGRAM_STRIDE, // sentinel: no partial datagram pending
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
                    if let Some(decision) = decode_trigger_buf(&state.trigger_buf, fin) {
                        match decision {
                            TriggerDecision::StatsRequest => {
                                state.is_stats_request = true;
                            }
                            TriggerDecision::Transfer { mode, bandwidth } => {
                                state.mode = mode;
                                state.bandwidth_limit = bandwidth;
                                state.tokens = 0.0;
                                state.last_refill = Instant::now();
                            }
                        }
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
    ///
    /// Each write is exactly DATAGRAM_STRIDE bytes (or fewer for the tail when
    /// `send_size` is finite).  The first 8 bytes carry a send timestamp
    /// (µs since `start_time`) so the client can compute RFC 3550 jitter from
    /// one-way delay variation — the same method iperf3 UDP uses.
    fn pump(&mut self, conn: &mut Connection, stream_id: u64, start_time: Instant) {
        let state = match self.streams.get_mut(&stream_id) {
            Some(s) => s,
            None => return,
        };

        if state.finished || !state.ready {
            return;
        }

        // Stack-allocated datagram buffer: 8-byte timestamp + zeros.
        let mut datagram = [0u8; DATAGRAM_STRIDE];

        loop {
            // ── Flush any partial datagram left from a previous backpressure stall ─
            // If we returned early last time because QUIC couldn't accept the full
            // datagram, finish flushing it before producing a new one.
            if state.dg_buf_pos < DATAGRAM_STRIDE && state.bytes_sent > 0 {
                let pending_size = state.dg_buf.len().min(DATAGRAM_STRIDE);
                while state.dg_buf_pos < pending_size {
                    let slice = &state.dg_buf[state.dg_buf_pos..pending_size];
                    match conn.stream_write(stream_id, Bytes::copy_from_slice(slice), false) {
                        Ok(written) => {
                            state.bytes_sent  += written;
                            state.dg_buf_pos  += written;
                            if state.bandwidth_limit > 0 { state.tokens -= written as f64; }
                            if state.dg_buf_pos < pending_size {
                                _ = conn.stream_want_write(stream_id, true);
                                return;
                            }
                        }
                        Err(Error::Done) => { _ = conn.stream_want_write(stream_id, true); return; }
                        Err(e) => { error!("{} stream {} write: {:?}", conn.trace_id(), stream_id, e); return; }
                    }
                }
            }

            // ── Determine chunk to write ──────────────────────────────────────
            // Always send whole DATAGRAM_STRIDE datagrams so the client's
            // fixed-stride timestamp parser stays in sync. A partial write
            // would cause payload bytes to be misread as send-timestamps.
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
                remaining.min(DATAGRAM_STRIDE)
            } else {
                DATAGRAM_STRIDE
            };

            // ── Token-bucket rate limiting ────────────────────────────────────
            // Refill first, then require a full datagram's worth of tokens.
            // Never write a partial datagram — wait until we can send `to_send`
            // bytes in one shot so the timestamp framing is never broken.
            if state.bandwidth_limit > 0 {
                let now = Instant::now();
                let elapsed = now.duration_since(state.last_refill).as_secs_f64();
                state.tokens = (state.tokens + elapsed * state.bandwidth_limit as f64)
                    .min(state.bandwidth_limit as f64);
                state.last_refill = now;

                if state.tokens < to_send as f64 {
                    // Not enough tokens for a full datagram – wait.
                    _ = conn.stream_want_write(stream_id, true);
                    return;
                }
            }

            // Stamp the send timestamp into bytes 0-7 of this datagram.
            let ts_us = start_time.elapsed().as_micros() as u64;
            datagram[0..8].copy_from_slice(&ts_us.to_le_bytes());

            // Copy into the state's staging buffer and reset position.
            // We always fill the staging buffer completely before writing,
            // so QUIC backpressure can resume mid-datagram without framing loss.
            state.dg_buf[..to_send].copy_from_slice(&datagram[..to_send]);
            state.dg_buf_pos = 0;
            let datagram_size = to_send;

            // Flush the staging buffer (may take multiple stream_write calls
            // if QUIC flow control only accepts part of it).
            while state.dg_buf_pos < datagram_size {
                let slice = &state.dg_buf[state.dg_buf_pos..datagram_size];
                match conn.stream_write(stream_id, Bytes::copy_from_slice(slice), false) {
                    Ok(written) => {
                        state.bytes_sent  += written;
                        state.dg_buf_pos  += written;
                        if state.bandwidth_limit > 0 {
                            state.tokens -= written as f64;
                        }
                        if state.dg_buf_pos < datagram_size {
                            // QUIC can't accept more right now; retry when writable.
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
    }

    /// Drain incoming data on an uplink stream (client → server).
    /// Parses the DATAGRAM_STRIDE-aligned send timestamps embedded by the client
    /// and computes RFC 3550 jitter from one-way delay variation (same method as
    /// the client uses for downlink).  Publishes the result to `live_jitter`.
    /// When the client's FIN arrives, writes a 16-byte stats reply back with FIN.
    fn drain_uplink(
        &mut self,
        conn: &mut Connection,
        stream_id: u64,
        live_jitter: &Arc<AtomicU64>,
    ) {
        let state = match self.streams.get_mut(&stream_id) {
            Some(s) if s.ready && s.mode == TransferMode::Uplink => s,
            _ => return,
        };
        let mut tmp = [0u8; 65536];
        let mut got_fin = false;
        loop {
            match conn.stream_read(stream_id, &mut tmp) {
                Ok((0, true)) => { got_fin = true; break; }
                Ok((0, false)) | Err(Error::Done) => break,
                Ok((n, fin)) => {
                    if fin { got_fin = true; }
                    // Parse per-datagram timestamps and update jitter.
                    let recv_us = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_micros() as u64;
                    let mut i = 0;
                    while i < n {
                        if self.dg_pos < 8 {
                            let hdr_remain = 8 - self.dg_pos;
                            let to_copy = hdr_remain.min(n - i);
                            self.ts_partial_buf[self.dg_pos..self.dg_pos + to_copy]
                                .copy_from_slice(&tmp[i..i + to_copy]);
                            self.dg_pos += to_copy;
                            i          += to_copy;
                            if self.dg_pos == 8 {
                                let send_us = u64::from_le_bytes(self.ts_partial_buf);
                                if let (Some(ps), Some(pr)) = (self.last_send_us, self.last_recv_us) {
                                    let rd = recv_us as i64 - pr as i64;
                                    let sd = send_us as i64 - ps as i64;
                                    let diff_ms = (rd - sd).unsigned_abs() as f64 / 1000.0;
                                    self.jitter_ms += (diff_ms - self.jitter_ms) / 16.0;
                                    live_jitter.store(self.jitter_ms.to_bits(), Ordering::Relaxed);
                                }
                                self.last_send_us = Some(send_us);
                                self.last_recv_us = Some(recv_us);
                            }
                        } else {
                            let payload_remain = DATAGRAM_STRIDE - self.dg_pos;
                            let to_skip = payload_remain.min(n - i);
                            self.dg_pos += to_skip;
                            i           += to_skip;
                            if self.dg_pos >= DATAGRAM_STRIDE {
                                self.dg_pos = 0;
                                self.app_datagrams_received += 1;
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("{} uplink drain {}: {:?}", conn.trace_id(), stream_id, e);
                    break;
                }
            }
        }
        if got_fin && !state.reply_sent {
            state.reply_sent = true;
            let jitter_bits = live_jitter.load(Ordering::Relaxed);
            let reply = encode_uplink_stats_reply(jitter_bits, self.app_datagrams_received);
            match conn.stream_write(stream_id, Bytes::copy_from_slice(&reply), true) {
                Ok(_) => {}
                Err(e) => error!("{} uplink reply write {}: {:?}", conn.trace_id(), stream_id, e),
            }
        }
    }
}

// ─────────────────────────────── ServerHandler ───────────────────────────────

struct ServerHandler {
    conns: FxHashMap<u64, ConnectionHandler>,
    send_size: usize,
    /// When the server started; used to stamp per-datagram send timestamps.
    server_start: Instant,
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
    /// Actual session duration in f64 bits (computed on conn close).
    actual_duration_bits: Arc<AtomicU64>,
    /// Time of the first connection establishment in this session.
    session_start: Option<Instant>,
    /// wandb API key, consumed on the first multipath connection with LinUCB metrics.
    wandb_key: Option<String>,
    /// Scheduler name for the wandb run label (e.g. "minrtt", "linucb").
    scheduler_name: String,
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
        actual_duration_bits: Arc<AtomicU64>,
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
            send_size: option.send_size,
            server_start: Instant::now(),
            keylog,
            qlog_dir: option.qlog_dir.clone(),
            live_bytes,
            live_lost,
            live_sent,
            live_jitter,
            is_uplink,
            rep_done,
            actual_duration_bits,
            session_start: None,
            wandb_key: {
                const DEFAULT_KEY: &str = "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";
                let key = std::env::var("WANDB_API_KEY")
                    .unwrap_or_else(|_| DEFAULT_KEY.to_string());
                if !key.is_empty() { Some(key) } else { None }
            },
            scheduler_name: match option.multipath_algor {
                MultipathAlgorithm::MinRtt        => "minrtt".to_string(),
                MultipathAlgorithm::RoundRobin    => "roundrobin".to_string(),
                MultipathAlgorithm::Redundant     => "redundant".to_string(),
                MultipathAlgorithm::LinUCB        => "linucb".to_string(),
                MultipathAlgorithm::EpsilonGreedy => "egreedy".to_string(),
            },
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
        if self.session_start.is_none() {
            self.session_start = Some(Instant::now());
        }
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

        // Per-path breakdown (multipath).
        let paths: Vec<_> = conn.paths_iter().collect();
        if paths.len() > 1 {
            if let Some(summary) = conn.multipath_scheduler_summary() {
                info!("{}", summary);
            }
            // Upload metrics to wandb only in downlink mode: the server is the
            // sender, so its scheduler metrics are meaningful. In uplink the
            // client uploads instead.
            let metrics = conn.multipath_scheduler_metrics_jsonl();
            if !metrics.is_empty() && !self.is_uplink.load(std::sync::atomic::Ordering::Relaxed) {
                if let Some(key) = self.wandb_key.take() {
                    if let Some(wb) = WandbLogger::new(&key, "quic", &self.scheduler_name) {
                        wb.upload_history(&metrics);
                    }
                }
            }
            info!("{} per-path stats ({} paths):", conn.trace_id(), paths.len());
            for (i, four_tuple) in paths.iter().enumerate() {
                if let Ok(ps) = conn.get_path_stats(four_tuple.local, four_tuple.remote) {
                    info!(
                        "  path[{}] {}→{}  recv={} B ({} pkts)  sent={} B ({} pkts)  lost={} B  srtt={} µs  latest_rtt={} µs",
                        i,
                        four_tuple.local,
                        four_tuple.remote,
                        ps.recv_bytes, ps.recv_count,
                        ps.sent_bytes, ps.sent_count,
                        ps.lost_bytes,
                        ps.srtt,
                        ps.latest_rtt,
                    );
                }
            }
        }

        // Record actual session duration so the reporter shows real elapsed time.
        if let Some(start) = self.session_start.take() {
            let secs = Instant::now().duration_since(start).as_secs_f64();
            self.actual_duration_bits.store(secs.to_bits(), Ordering::Relaxed);
        }
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
            let is_stats_req = self.conns.get(&idx)
                .and_then(|h| h.streams.get(&stream_id))
                .map(|s| s.is_stats_request)
                .unwrap_or(false);
            if is_stats_req {
                // Client requested stats after downlink duration: reply with [lost:u64][sent:u64]+FIN.
                let stats = conn.stats();
                let reply = encode_downlink_stats_reply(stats.lost_count, stats.sent_count);
                let _ = conn.stream_write(stream_id, Bytes::copy_from_slice(&reply), true);
                return;
            }
            let mode = self.conns.get(&idx)
                .and_then(|h| h.streams.get(&stream_id))
                .map(|s| s.mode)
                .unwrap_or(TransferMode::Downlink);
            match mode {
                TransferMode::Downlink => {
                    if let Some(handler) = self.conns.get_mut(&idx) {
                        handler.pump(conn, stream_id, self.server_start);
                    }
                }
                TransferMode::Uplink => {
                    self.is_uplink.store(true, Ordering::Relaxed);
                    if let Some(handler) = self.conns.get_mut(&idx) {
                        handler.drain_uplink(conn, stream_id, &self.live_jitter);
                    }
                }
            }
        } else {
            if let Some(handler) = self.conns.get_mut(&idx) {
                handler.drain_uplink(conn, stream_id, &self.live_jitter);
            }
        }
        // Update live uplink stats for the interval reporter.
        if self.is_uplink.load(Ordering::Relaxed) {
            if let Some(handler) = self.conns.get_mut(&idx) {
                let stats = conn.stats();
                let delta = stats.recv_bytes.saturating_sub(handler.prev_bytes_recv);
                if delta > 0 {
                    handler.prev_bytes_recv = stats.recv_bytes;
                    self.live_bytes.fetch_add(delta, Ordering::Relaxed);
                    // Jitter is now updated inside drain_uplink via per-datagram timestamps.
                }
                // For uplink: "total datagrams" = app-level datagrams received.
                self.live_sent.store(handler.app_datagrams_received, Ordering::Relaxed);
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
        if let Some(handler) = self.conns.get_mut(&idx) {
            handler.pump(conn, stream_id, self.server_start);
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
        actual_duration_bits: Arc<AtomicU64>,
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
        let handler = ServerHandler::new(option, live_bytes, live_lost, live_sent, live_jitter, is_uplink, rep_done, actual_duration_bits)?;
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
    let actual_duration_bits = Arc::new(AtomicU64::new(0));
    let terminated   = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&terminated))?;

    let mut server = Server::new(
        &option,
        Arc::clone(&live_bytes), Arc::clone(&live_lost),
        Arc::clone(&live_sent),  Arc::clone(&live_jitter),
        Arc::clone(&is_uplink),  Arc::clone(&rep_done),
        Arc::clone(&actual_duration_bits),
    )?;

    info!(
        "{} listening on {:?}  CC={:?} multipath={}",
        server.endpoint.trace_id(),
        option.listen,
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
    let rdb = Arc::clone(&actual_duration_bits);
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
            let _ = std::io::stdout().flush();
            let session_uplink = ru.load(Ordering::Relaxed);
            if session_uplink {
                println!(
                    "  {:<12}  {:>10}  {:>16}  {:>10}  {}",
                    "Interval", "Transfer", "Bitrate", "Jitter", "Lost/Total Datagrams"
                );
            } else {
                println!(
                    "  {:<12}  {:>10}  {:>16}  {}",
                    "Interval", "Transfer", "Bitrate", "Total Datagrams"
                );
            }
            let _ = std::io::stdout().flush();
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
                if session_uplink {
                    let loss_pct = if d_sent > 0 { d_lost as f64 / d_sent as f64 * 100.0 } else { 0.0 };
                    println!(
                        "  {:<12}  {:>10}  {:>16}  {:>13}  {}/{} ({:.2}%)",
                        format!("{:.2}-{:.2} s", t_start, t_end),
                        format!("{:.2} MB", mb),
                        format!("{:.2} Mbits/sec", mbps),
                        format!("{:.3} ms", jitter_ms),
                        d_lost, d_sent, loss_pct,
                    );
                } else {
                    println!(
                        "  {:<12}  {:>10}  {:>16}  {}",
                        format!("{:.2}-{:.2} s", t_start, t_end),
                        format!("{:.2} MB", mb),
                        format!("{:.2} Mbits/sec", mbps),
                        d_sent,
                    );
                }
                let _ = std::io::stdout().flush();
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
                let total_secs = {
                    let d = f64::from_bits(rdb.load(Ordering::Relaxed));
                    if d > 0.0 { d } else { interval.max(1) as f64 }
                };
                let total_mb   = current as f64 / 1e6;
                let total_mbps = (current as f64 * 8.0) / 1e6 / total_secs;
                let loss_pct   = if cur_sent > 0 { cur_lost as f64 / cur_sent as f64 * 100.0 } else { 0.0 };
                let role = if uplink { "receiver" } else { "sender" };
                println!("- - - - - - - - - - - - - - - - - - - - - - - - -");
                println!(
                    "  {:<12}  {:>10}  {:>16}  {:>10}  {}",
                    "Interval", "Transfer", "Bitrate", "Jitter", "Lost/Total Datagrams"
                );
                // Jitter is only meaningful for the receiver; leave blank for sender (downlink).
                let jitter_col = if uplink {
                    format!("{:.3} ms", jitter_ms)
                } else {
                    String::new()
                };
                println!(
                    "  {:<12}  {:>10}  {:>16}  {:>13}  {}/{} ({:.2}%)  {}",
                    format!("0.00-{:.2} s", total_secs),
                    format!("{:.2} MB", total_mb),
                    format!("{:.2} Mbits/sec", total_mbps),
                    jitter_col,
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
            rdb.store(0u64, Ordering::Relaxed);
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

// ─────────────────────────────────── tests ───────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── decode_trigger_buf ────────────────────────────────────────────────────

    #[test]
    fn trigger_stats_request_exact() {
        // 0xFF + FIN → StatsRequest
        assert_eq!(
            decode_trigger_buf(&[0xFF], true),
            Some(TriggerDecision::StatsRequest)
        );
    }

    #[test]
    fn trigger_stats_request_needs_fin() {
        // 0xFF without FIN → not ready yet
        assert_eq!(decode_trigger_buf(&[0xFF], false), None);
    }

    #[test]
    fn trigger_downlink_unlimited() {
        // mode=0 (downlink), bandwidth=0
        let mut buf = [0u8; 9];
        buf[0] = 0;
        buf[1..9].copy_from_slice(&0u64.to_le_bytes());
        assert_eq!(
            decode_trigger_buf(&buf, false),
            Some(TriggerDecision::Transfer { mode: TransferMode::Downlink, bandwidth: 0 })
        );
    }

    #[test]
    fn trigger_downlink_with_bandwidth() {
        let bw: u64 = 62_500_000; // 500 Mbit/s in bytes/s
        let mut buf = [0u8; 9];
        buf[0] = 0;
        buf[1..9].copy_from_slice(&bw.to_le_bytes());
        assert_eq!(
            decode_trigger_buf(&buf, false),
            Some(TriggerDecision::Transfer { mode: TransferMode::Downlink, bandwidth: bw })
        );
    }

    #[test]
    fn trigger_uplink_with_bandwidth() {
        let bw: u64 = 125_000_000; // 1 Gbit/s in bytes/s
        let mut buf = [0u8; 9];
        buf[0] = 1; // Uplink
        buf[1..9].copy_from_slice(&bw.to_le_bytes());
        assert_eq!(
            decode_trigger_buf(&buf, false),
            Some(TriggerDecision::Transfer { mode: TransferMode::Uplink, bandwidth: bw })
        );
    }

    #[test]
    fn trigger_fin_no_data_defaults_to_downlink() {
        // FIN with empty buf → Downlink, 0
        assert_eq!(
            decode_trigger_buf(&[], true),
            Some(TriggerDecision::Transfer { mode: TransferMode::Downlink, bandwidth: 0 })
        );
    }

    #[test]
    fn trigger_partial_no_decision() {
        // Only 4 bytes, no FIN → still waiting
        let buf = [0x00, 0x01, 0x02, 0x03];
        assert_eq!(decode_trigger_buf(&buf, false), None);
    }

    #[test]
    fn trigger_extra_bytes_ignored() {
        // 12-byte buffer: first 9 bytes are parsed, rest ignored
        let bw: u64 = 1_000;
        let mut buf = [0xAAu8; 12];
        buf[0] = 0; // Downlink
        buf[1..9].copy_from_slice(&bw.to_le_bytes());
        assert_eq!(
            decode_trigger_buf(&buf, false),
            Some(TriggerDecision::Transfer { mode: TransferMode::Downlink, bandwidth: bw })
        );
    }

    // ── encode_uplink_stats_reply ─────────────────────────────────────────────

    #[test]
    fn uplink_reply_roundtrip() {
        let jitter_bits = (1.234_f64).to_bits();
        let lost_count: u64 = 42;
        let reply = encode_uplink_stats_reply(jitter_bits, lost_count);

        let decoded_jitter = u64::from_le_bytes(reply[0..8].try_into().unwrap());
        let decoded_lost   = u64::from_le_bytes(reply[8..16].try_into().unwrap());

        assert_eq!(decoded_jitter, jitter_bits);
        assert_eq!(decoded_lost,   lost_count);
        assert_eq!(f64::from_bits(decoded_jitter), 1.234_f64);
    }

    #[test]
    fn uplink_reply_zero_values() {
        let reply = encode_uplink_stats_reply(0, 0);
        assert_eq!(reply, [0u8; 16]);
    }

    #[test]
    fn uplink_reply_field_order() {
        // jitter is in bytes [0..8], lost_count is in bytes [8..16] — not swapped.
        let reply = encode_uplink_stats_reply(1, 2);
        assert_eq!(u64::from_le_bytes(reply[0..8].try_into().unwrap()), 1);
        assert_eq!(u64::from_le_bytes(reply[8..16].try_into().unwrap()), 2);
    }

    // ── encode_downlink_stats_reply ───────────────────────────────────────────

    #[test]
    fn downlink_reply_roundtrip() {
        let lost_count: u64 = 7;
        let sent_count: u64 = 100_000;
        let reply = encode_downlink_stats_reply(lost_count, sent_count);

        let decoded_lost = u64::from_le_bytes(reply[0..8].try_into().unwrap());
        let decoded_sent = u64::from_le_bytes(reply[8..16].try_into().unwrap());

        assert_eq!(decoded_lost, lost_count);
        assert_eq!(decoded_sent, sent_count);
    }

    #[test]
    fn downlink_reply_zero_values() {
        let reply = encode_downlink_stats_reply(0, 0);
        assert_eq!(reply, [0u8; 16]);
    }

    #[test]
    fn downlink_reply_field_order() {
        // lost is in bytes [0..8], sent is in bytes [8..16] — not swapped.
        let reply = encode_downlink_stats_reply(1, 2);
        assert_eq!(u64::from_le_bytes(reply[0..8].try_into().unwrap()), 1);
        assert_eq!(u64::from_le_bytes(reply[8..16].try_into().unwrap()), 2);
    }
}
