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

// ─────────────────────────── Per-stream send state ───────────────────────────

struct StreamSendState {
    bytes_sent: usize,
    finished: bool,
}

// ─────────────────────────── Per-connection handler ──────────────────────────

#[derive(Default)]
struct ConnectionHandler {
    /// Total bytes to send per stream (0 = unlimited).
    send_size: usize,
    streams: HashMap<u64, StreamSendState>,
}

impl ConnectionHandler {
    /// Register a new stream and start pumping data.
    fn on_new_stream(&mut self, conn: &mut Connection, stream_id: u64, buf: &[u8]) {
        self.streams.insert(
            stream_id,
            StreamSendState {
                bytes_sent: 0,
                finished: false,
            },
        );
        self.pump(conn, stream_id, buf);
    }

    /// Push as many bytes as possible; registers `stream_want_write` on backpressure.
    fn pump(&mut self, conn: &mut Connection, stream_id: u64, buf: &[u8]) {
        let state = match self.streams.get_mut(&stream_id) {
            Some(s) => s,
            None => return,
        };

        if state.finished {
            return;
        }

        loop {
            let to_send = if self.send_size > 0 {
                let remaining = self.send_size.saturating_sub(state.bytes_sent);
                if remaining == 0 {
                    // Done: send FIN.
                    match conn.stream_write(stream_id, Bytes::new(), true) {
                        Ok(_) | Err(Error::Done) => {}
                        Err(e) => error!("{} stream FIN error: {:?}", conn.trace_id(), e),
                    }
                    state.finished = true;
                    return;
                }
                remaining.min(buf.len())
            } else {
                buf.len()
            };

            let fin =
                self.send_size > 0 && (state.bytes_sent + to_send >= self.send_size);

            match conn.stream_write(
                stream_id,
                Bytes::copy_from_slice(&buf[..to_send]),
                fin,
            ) {
                Ok(written) => {
                    state.bytes_sent += written;
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
}

// ─────────────────────────────── ServerHandler ───────────────────────────────

struct ServerHandler {
    conns: FxHashMap<u64, ConnectionHandler>,
    /// Scratch buffer for discarding client data.
    recv_buf: Vec<u8>,
    /// Zero-filled send buffer.
    send_buf: Vec<u8>,
    send_size: usize,
    keylog: Option<File>,
    qlog_dir: Option<String>,
}

impl ServerHandler {
    fn new(option: &ServerOpt) -> Result<Self> {
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
            recv_buf: vec![0u8; option.chunk_size],
            send_buf: vec![0u8; option.chunk_size],
            send_size: option.send_size,
            keylog,
            qlog_dir: option.qlog_dir.clone(),
        })
    }

    fn ensure_conn_handler(&mut self, conn: &mut Connection) {
        let idx = conn.index().unwrap();
        if self.conns.contains_key(&idx) {
            return;
        }
        self.conns.insert(
            idx,
            ConnectionHandler {
                send_size: self.send_size,
                streams: HashMap::default(),
            },
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
        let send_buf = self.send_buf.clone();
        if let Some(handler) = self.conns.get_mut(&idx) {
            handler.on_new_stream(conn, stream_id, &send_buf);
        }
    }

    fn on_stream_readable(&mut self, conn: &mut Connection, stream_id: u64) {
        // Drain and discard — the trigger data sent by the client is irrelevant.
        loop {
            match conn.stream_read(stream_id, &mut self.recv_buf) {
                Ok((0, _)) | Err(Error::Done) => break,
                Ok((n, _)) => debug!(
                    "{} discarded {} client bytes on stream {}",
                    conn.trace_id(),
                    n,
                    stream_id
                ),
                Err(e) => {
                    error!("{} stream_read error: {:?}", conn.trace_id(), e);
                    break;
                }
            }
        }
    }

    fn on_stream_writable(&mut self, conn: &mut Connection, stream_id: u64) {
        _ = conn.stream_want_write(stream_id, false);
        let idx = conn.index().unwrap();
        let send_buf = self.send_buf.clone();
        if let Some(handler) = self.conns.get_mut(&idx) {
            handler.pump(conn, stream_id, &send_buf);
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
    fn new(option: &ServerOpt) -> Result<Self> {
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
        let handler = ServerHandler::new(option)?;
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

    let mut server = Server::new(&option)?;

    info!(
        "{} listening on {:?}  send_size={} chunk={} CC={:?} multipath={}",
        server.endpoint.trace_id(),
        option.listen,
        option.send_size,
        option.chunk_size,
        option.congestion_control_algor,
        option.enable_multipath,
    );

    let mut events = mio::Events::with_capacity(1024);
    loop {
        if let Err(e) = server.endpoint.process_connections() {
            error!("process_connections: {:?}", e);
        }

        server
            .poll
            .poll(&mut events, server.endpoint.timeout())?;

        for event in events.iter() {
            if event.is_readable() {
                server.process_read_event(event)?;
            }
        }

        server.endpoint.on_timeout(Instant::now());
    }
}
