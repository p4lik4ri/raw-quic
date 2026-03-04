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

//! A raw-QUIC throughput client (reverse mode, like `iperf3 -R`).
//! HTTP/3 and HTTP/0.9 have been removed.  All QUIC transport features
//! (congestion control, multipath, qlog …) are preserved.
//!
//! The client connects to the server, opens a bidirectional QUIC stream with
//! an empty FIN to signal readiness, then receives and measures all bulk data
//! that the server pumps back.

use std::cell::RefCell;
use std::fs::create_dir_all;
use std::net::IpAddr;
use std::net::Ipv4Addr;
use std::net::Ipv6Addr;
use std::net::SocketAddr;
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::sync::Mutex;
use std::thread;
use std::time::Instant;

use bytes::Bytes;
use clap::Parser;
use log::debug;
use log::error;
use log::info;
use mio::event::Event;
use rustc_hash::FxHashMap;

use tquic::connection::ConnectionStats;
use tquic::error::Error;
use tquic::CertCompressionAlgorithm;
use tquic::Config;
use tquic::CongestionControlAlgorithm;
use tquic::Connection;
use tquic::Endpoint;
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

#[derive(Parser, Debug, Clone)]
#[clap(name = "client", version = env!("CARGO_PKG_VERSION"))]
pub struct ClientOpt {
    /// Server address (host:port).
    #[clap(short, long, value_name = "ADDR")]
    pub connect_to: SocketAddr,

    /// Optional local IP addresses, comma-separated. e.g. 192.168.1.10,192.168.2.20
    #[clap(long, value_delimiter = ',', value_name = "ADDR")]
    pub local_addresses: Vec<IpAddr>,

    /// Server name for TLS SNI (defaults to server IP).
    #[clap(long, value_name = "STR")]
    pub server_name: Option<String>,

    // ── Concurrency ──────────────────────────────────────────────────────────
    /// Number of threads.
    #[clap(short, long, default_value = "1", value_name = "NUM", help_heading = "Concurrency")]
    pub threads: u32,

    /// Number of concurrent connections per thread.
    #[clap(long, default_value = "1", value_name = "NUM", help_heading = "Concurrency")]
    pub max_concurrent_conns: u32,

    /// Number of concurrent streams per connection.
    #[clap(long, default_value = "1", value_name = "NUM", help_heading = "Concurrency")]
    pub streams_per_conn: u64,

    /// Benchmarking duration in seconds (0 = run until server sends FIN).
    #[clap(short, long, default_value = "10", value_name = "SEC", help_heading = "Concurrency")]
    pub duration: u64,

    /// Target bandwidth in bits/sec. Accepts K / M / G suffix (e.g. 500M = 500 Mbit/s).
    /// 0 or omitted means unlimited. The value is negotiated with the server via the
    /// trigger stream so no server-side flag is needed.
    #[clap(long, default_value = "0", value_name = "BPS",
           value_parser = parse_bandwidth_cli,
           help_heading = "Concurrency")]
    pub bandwidth: u64,  // stored internally as bytes/sec

    // ── Protocol ──────────────────────────────────────────────────────────────
    /// File used for session resumption.
    #[clap(short, long, value_name = "FILE", help_heading = "Protocol")]
    pub session_file: Option<String>,

    /// Enable early data.
    #[clap(short, long, help_heading = "Protocol")]
    pub enable_early_data: bool,

    /// Enable certificate compression.
    #[clap(long, value_name = "STR", help_heading = "Protocol")]
    pub certificate_compression: Vec<CertCompressionAlgorithmArg>,

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
    #[clap(long, help_heading = "Protocol")]
    pub enable_multipath: bool,

    /// Multipath scheduling algorithm.
    #[clap(long, default_value = "MINRTT", help_heading = "Protocol")]
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

    /// Length of connection id in bytes.
    #[clap(long, default_value = "8", value_name = "NUM", help_heading = "Protocol")]
    pub cid_len: usize,

    // ── Output ────────────────────────────────────────────────────────────────
    /// Log level (OFF/ERROR/WARN/INFO/DEBUG/TRACE).
    #[clap(long, default_value = "INFO", value_name = "STR", help_heading = "Output")]
    pub log_level: log::LevelFilter,

    /// Log file path (defaults to stderr).
    #[clap(long, value_name = "FILE", help_heading = "Output")]
    pub log_file: Option<String>,

    /// Save TLS key log into the given file.
    #[clap(short, long, value_name = "FILE", help_heading = "Output")]
    pub keylog_file: Option<String>,

    /// Save qlog file (<trace_id>.qlog) into the given directory.
    #[clap(long, value_name = "DIR", help_heading = "Output")]
    pub qlog_dir: Option<String>,

    // ── Misc ──────────────────────────────────────────────────────────────────
    /// Exit if this many consecutive connection failures occur at start.
    #[clap(long, default_value = "10", value_name = "NUM", help_heading = "Misc")]
    pub connection_failure_threshold: u64,

    /// Batch size for sending packets.
    #[clap(long, default_value = "1", value_name = "NUM", help_heading = "Misc")]
    pub send_batch_size: usize,

    /// Disable encryption on 1-RTT packets.
    #[clap(long, help_heading = "Misc")]
    pub disable_encryption: bool,
}

const MAX_BUF_SIZE: usize = 65536;

// ─────────────────────────── Shared client context ───────────────────────────

#[derive(Default)]
struct ClientContext {
    session: Option<Vec<u8>>,
    bytes_received: u64,
    conn_total: u64,
    conn_handshake_success: u64,
    conn_finish: u64,
    conn_finish_success: u64,
    conn_finish_failed: u64,
    end_time: Option<Instant>,
    conn_stats: ConnectionStats,
}

fn accum_conn_stats(total: &mut ConnectionStats, one: &ConnectionStats) {
    total.recv_count += one.recv_count;
    total.sent_count += one.sent_count;
    total.lost_count += one.lost_count;
    total.recv_bytes += one.recv_bytes;
    total.sent_bytes += one.sent_bytes;
    total.lost_bytes += one.lost_bytes;
}

// ─────────────────────────── Multi-thread client ─────────────────────────────

struct Client {
    option: ClientOpt,
    context: Arc<Mutex<ClientContext>>,
    start_time: Instant,
    terminated: Arc<AtomicBool>,
}

impl Client {
    pub fn new(option: ClientOpt) -> Result<Self> {
        let context = Arc::new(Mutex::new(ClientContext::default()));
        let terminated = Arc::new(AtomicBool::new(false));
        signal_hook::flag::register(signal_hook::consts::SIGINT, Arc::clone(&terminated))?;
        Ok(Self { option, context, start_time: Instant::now(), terminated })
    }

    pub fn start(&mut self) {
        self.start_time = Instant::now();
        let mut handles = vec![];
        for _ in 0..self.option.threads {
            let opt = self.option.clone();
            let ctx = self.context.clone();
            let term = self.terminated.clone();
            handles.push(thread::spawn(move || {
                Worker::new(opt, ctx, term).unwrap().start().unwrap();
            }));
        }
        for h in handles { h.join().unwrap(); }
        self.print_stats();
    }

    fn print_stats(&self) {
        let ctx = self.context.lock().unwrap();
        let duration = ctx.end_time.unwrap_or_else(Instant::now) - self.start_time;
        let secs = duration.as_secs_f64().max(1e-9);
        let bytes = ctx.bytes_received;
        let gbps = (bytes as f64 * 8.0) / 1e9 / secs;
        let mbps = (bytes as f64 * 8.0) / 1e6 / secs;

        println!();
        println!("[ rawquic ] Reverse-mode throughput (server → client)");
        println!("  Duration  : {:.3} s", secs);
        println!("  Transfer  : {} bytes  ({:.3} GB)", bytes, bytes as f64 / 1e9);
        println!("  Bitrate   : {:.3} Gbits/sec  ({:.1} Mbits/sec)", gbps, mbps);
        println!(
            "  Conns     : total {}, ok {}, failed {}",
            ctx.conn_total, ctx.conn_finish_success, ctx.conn_finish_failed,
        );
        println!(
            "  Pkts  recv/sent/lost : {}/{}/{}",
            ctx.conn_stats.recv_count, ctx.conn_stats.sent_count, ctx.conn_stats.lost_count,
        );
        println!(
            "  Bytes recv/sent/lost : {}/{}/{}",
            ctx.conn_stats.recv_bytes, ctx.conn_stats.sent_bytes, ctx.conn_stats.lost_bytes,
        );
        println!();
    }
}

// ─────────────────────────── Worker (single thread) ──────────────────────────

#[derive(Default)]
struct WorkerContext {
    session: Option<Vec<u8>>,
    bytes_received: u64,
    conn_total: u64,
    conn_handshake_success: u64,
    conn_finish: u64,
    conn_finish_success: u64,
    conn_finish_failed: u64,
    concurrent_conns: u32,
    conn_stats: ConnectionStats,
    connected: bool,
}

impl WorkerContext {
    fn with_option(opt: &ClientOpt) -> Self {
        let mut ctx = WorkerContext::default();
        if let Some(sf) = &opt.session_file {
            if let Ok(data) = std::fs::read(sf) {
                ctx.session = Some(data);
            }
        }
        ctx
    }
}

/// Per-connection receive bookkeeping.
struct DataReceiver {
    bytes_received: u64,
    streams_opened: u64,
    streams_finished: u64,
}

impl DataReceiver {
    fn new() -> Self {
        Self { bytes_received: 0, streams_opened: 0, streams_finished: 0 }
    }
}

struct Worker {
    option: ClientOpt,
    endpoint: Endpoint,
    poll: mio::Poll,
    remote: SocketAddr,
    sock: Rc<QuicSocket>,
    worker_ctx: Rc<RefCell<WorkerContext>>,
    client_ctx: Arc<Mutex<ClientContext>>,
    receivers: Rc<RefCell<FxHashMap<u64, DataReceiver>>>,
    recv_buf: Vec<u8>,
    start_time: Instant,
    end_time: Option<Instant>,
    terminated: Arc<AtomicBool>,
}

impl Worker {
    pub fn new(
        option: ClientOpt,
        client_ctx: Arc<Mutex<ClientContext>>,
        terminated: Arc<AtomicBool>,
    ) -> Result<Self> {
        let mut config = Config::new()?;
        config.enable_stateless_reset(!option.disable_stateless_reset);
        config.set_max_handshake_timeout(option.handshake_timeout);
        config.set_max_idle_timeout(option.idle_timeout);
        config.set_initial_rtt(option.initial_rtt);
        config.set_pto_linear_factor(option.pto_linear_factor);
        config.set_max_pto(option.max_pto);
        config.set_max_concurrent_conns(option.max_concurrent_conns);
        config.set_initial_max_streams_bidi(option.streams_per_conn);
        config.set_cid_len(option.cid_len);
        config.set_send_batch_size(option.send_batch_size);
        config.set_recv_udp_payload_size(option.recv_udp_payload_size);
        config.set_send_udp_payload_size(option.send_udp_payload_size);
        config.set_congestion_control_algorithm(option.congestion_control_algor);
        config.set_initial_congestion_window(option.initial_congestion_window);
        config.set_min_congestion_window(option.min_congestion_window);
        config.enable_multipath(option.enable_multipath);
        config.set_multipath_algorithm(option.multipath_algor);
        config.set_active_connection_id_limit(option.active_cid_limit);
        config.enable_encryption(!option.disable_encryption);

        let mut tls = TlsConfig::new_client_config(
            vec![b"rawquic".to_vec()],
            option.enable_early_data,
        )?;
        if !option.certificate_compression.is_empty() {
            let algs: Vec<CertCompressionAlgorithm> =
                option.certificate_compression.iter().map(|&a| a.into()).collect();
            tls.enable_certificate_compression(algs)?;
        }
        config.set_tls_config(tls);

        let poll = mio::Poll::new()?;
        let worker_ctx = Rc::new(RefCell::new(WorkerContext::with_option(&option)));
        let receivers = Rc::new(RefCell::new(FxHashMap::default()));

        let remote = option.connect_to;
        let local = if !option.local_addresses.is_empty() {
            SocketAddr::new(option.local_addresses[0], 0)
        } else if remote.is_ipv4() {
            SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0)
        } else {
            SocketAddr::new(IpAddr::V6(Ipv6Addr::UNSPECIFIED), 0)
        };

        let mut sock = QuicSocket::new(&local, poll.registry())?;
        let mut assigned_addrs = vec![sock.local_addr()];
        for ip in option.local_addresses.get(1..).unwrap_or(&[]) {
            let addr = sock.add(&SocketAddr::new(*ip, 0), poll.registry())?;
            assigned_addrs.push(addr);
        }
        let sock = Rc::new(sock);

        let handler = WorkerHandler::new(
            &option,
            &assigned_addrs,
            worker_ctx.clone(),
            receivers.clone(),
        );

        Ok(Worker {
            option,
            endpoint: Endpoint::new(Box::new(config), false, Box::new(handler), sock.clone()),
            poll,
            remote,
            sock,
            worker_ctx,
            client_ctx,
            receivers,
            recv_buf: vec![0u8; MAX_BUF_SIZE],
            start_time: Instant::now(),
            end_time: None,
            terminated,
        })
    }

    pub fn start(&mut self) -> Result<()> {
        self.start_time = Instant::now();
        let mut events = mio::Events::with_capacity(1024);
        loop {
            if self.process()? { break; }
            self.poll.poll(&mut events, self.endpoint.timeout())?;
            for event in events.iter() {
                if event.is_readable() {
                    self.process_read_event(event)?;
                }
            }
            self.endpoint.on_timeout(Instant::now());
        }
        self.finish();
        Ok(())
    }

    fn should_exit(&self) -> bool {
        if self.terminated.load(Ordering::Relaxed) {
            info!("terminated by signal");
            return true;
        }
        let ctx = self.worker_ctx.borrow();
        if !ctx.connected
            && ctx.conn_finish_failed >= self.option.connection_failure_threshold
        {
            error!("connect {:?} failed repeatedly", self.option.connect_to);
            return true;
        }
        if self.option.duration > 0
            && (Instant::now() - self.start_time).as_secs() >= self.option.duration
        {
            return true;
        }
        false
    }

    fn process(&mut self) -> Result<bool> {
        self.endpoint.process_connections()?;

        if self.should_exit() {
            self.endpoint.close(false);
            let idxs: Vec<u64> = self.receivers.borrow().keys().cloned().collect();
            for idx in idxs {
                if let Some(conn) = self.endpoint.conn_get_mut(idx) {
                    _ = conn.close(true, 0x00, b"done");
                }
            }
            if self.end_time.is_none() {
                self.end_time = Some(Instant::now());
            }
            if self.receivers.borrow().is_empty() {
                return Ok(true);
            }
            return Ok(false);
        }

        // Spawn new connections up to the limit.
        let mut ctx = self.worker_ctx.borrow_mut();
        while ctx.concurrent_conns < self.option.max_concurrent_conns {
            let sni = self.option.server_name.as_deref();
            match self.endpoint.connect(
                self.sock.local_addr(),
                self.remote,
                sni,
                ctx.session.as_deref(),
                None,
                None,
            ) {
                Ok(_) => {
                    ctx.concurrent_conns += 1;
                    ctx.conn_total += 1;
                }
                Err(e) => return Err(format!("connect: {:?}", e).into()),
            }
        }
        drop(ctx);
        Ok(false)
    }

    fn process_read_event(&mut self, event: &Event) -> Result<()> {
        loop {
            let (len, local, remote) =
                match self.sock.recv_from(&mut self.recv_buf, event.token()) {
                    Ok(v) => v,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        debug!("recv would block");
                        break;
                    }
                    Err(e) => return Err(format!("socket recv: {:?}", e).into()),
                };
            let pkt_info = PacketInfo { src: remote, dst: local, time: Instant::now() };
            if let Err(e) = self.endpoint.recv(&mut self.recv_buf[..len], &pkt_info) {
                error!("endpoint recv: {:?}", e);
            }
        }
        Ok(())
    }

    fn finish(&mut self) {
        let ctx = self.worker_ctx.borrow();
        let mut client_ctx = self.client_ctx.lock().unwrap();
        client_ctx.session.clone_from(&ctx.session);
        client_ctx.bytes_received += ctx.bytes_received;
        client_ctx.conn_total += ctx.conn_total;
        client_ctx.conn_handshake_success += ctx.conn_handshake_success;
        client_ctx.conn_finish += ctx.conn_finish;
        client_ctx.conn_finish_success += ctx.conn_finish_success;
        client_ctx.conn_finish_failed += ctx.conn_finish_failed;
        if self.end_time > client_ctx.end_time {
            client_ctx.end_time = self.end_time;
        }
        accum_conn_stats(&mut client_ctx.conn_stats, &ctx.conn_stats);
    }
}

// ─────────────────────────── TransportHandler ────────────────────────────────

struct WorkerHandler {
    option: ClientOpt,
    worker_ctx: Rc<RefCell<WorkerContext>>,
    receivers: Rc<RefCell<FxHashMap<u64, DataReceiver>>>,
    remote: SocketAddr,
    local_addresses: Vec<SocketAddr>,
    recv_buf: Vec<u8>,
}

impl WorkerHandler {
    fn new(
        option: &ClientOpt,
        local_addresses: &[SocketAddr],
        worker_ctx: Rc<RefCell<WorkerContext>>,
        receivers: Rc<RefCell<FxHashMap<u64, DataReceiver>>>,
    ) -> Self {
        Self {
            option: option.clone(),
            worker_ctx,
            receivers,
            remote: option.connect_to,
            local_addresses: local_addresses.to_owned(),
            recv_buf: vec![0u8; MAX_BUF_SIZE],
        }
    }

    /// Open `streams_per_conn` trigger streams.
    /// Sends 8 bytes (bandwidth limit in bytes/sec as u64 LE) + FIN so the
    /// server knows how fast to send.  0 means unlimited.
    fn open_trigger_streams(&self, conn: &mut Connection) {
        let idx = conn.index().unwrap();
        let mut receivers = self.receivers.borrow_mut();
        let recv = match receivers.get_mut(&idx) {
            Some(r) => r,
            None => return,
        };
        // 8-byte little-endian bandwidth (bytes/sec).  Sent with FIN.
        let trigger = Bytes::copy_from_slice(&self.option.bandwidth.to_le_bytes());
        for _ in 0..self.option.streams_per_conn {
            let stream_id = recv.streams_opened * 4; // 0, 4, 8, … client-initiated bidi
            match conn.stream_write(stream_id, trigger.clone(), true) {
                Ok(_) => {
                    recv.streams_opened += 1;
                    debug!("{} opened trigger stream {}", conn.trace_id(), stream_id);
                }
                Err(Error::StreamLimitError) => {
                    debug!("{} stream limit reached", conn.trace_id());
                    break;
                }
                Err(e) => {
                    error!("{} trigger stream {}: {:?}", conn.trace_id(), stream_id, e);
                }
            }
        }
    }
}

impl TransportHandler for WorkerHandler {
    fn on_conn_created(&mut self, conn: &mut Connection) {
        debug!("{} connection created", conn.trace_id());

        if let Some(kf) = &self.option.keylog_file {
            if let Ok(f) = std::fs::OpenOptions::new().create(true).append(true).open(kf) {
                conn.set_keylog(Box::new(f));
            }
        }
        if let Some(dir) = &self.option.qlog_dir {
            let path = Path::new(dir).join(format!("{}.qlog", conn.trace_id()));
            if let Ok(f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
                conn.set_qlog(
                    Box::new(f),
                    "client qlog".into(),
                    format!("id={}", conn.trace_id()),
                );
            }
        }

        let idx = conn.index().unwrap();
        self.receivers.borrow_mut().insert(idx, DataReceiver::new());

        if conn.is_in_early_data() {
            self.open_trigger_streams(conn);
        }
    }

    fn on_conn_established(&mut self, conn: &mut Connection) {
        debug!("{} connection established (resumed={})", conn.trace_id(), conn.is_resumed());
        {
            let mut ctx = self.worker_ctx.borrow_mut();
            ctx.conn_handshake_success += 1;
            ctx.connected = true;
        }

        // Add additional multipath paths if configured.
        for local in self.local_addresses.get(1..).unwrap_or(&[]) {
            match conn.add_path(*local, self.remote) {
                Ok(_) => debug!("{} add_path {}-{}", conn.trace_id(), local, self.remote),
                Err(e) => debug!("{} add_path {}-{}: {}", conn.trace_id(), local, self.remote, e),
            }
        }

        self.open_trigger_streams(conn);
    }

    fn on_conn_closed(&mut self, conn: &mut Connection) {
        debug!("{} connection closed", conn.trace_id());
        let idx = conn.index().unwrap();

        let mut ctx = self.worker_ctx.borrow_mut();
        if let Some(recv) = self.receivers.borrow().get(&idx) {
            ctx.bytes_received += recv.bytes_received;
        }
        self.receivers.borrow_mut().remove(&idx);
        accum_conn_stats(&mut ctx.conn_stats, conn.stats());

        if self.option.session_file.is_some() {
            if let Some(session) = conn.session() {
                ctx.session = Some(session.to_vec());
            }
        }
        ctx.conn_finish += 1;

        let local_app_close = conn.local_error().map(|e| e.is_app).unwrap_or(false);
        let peer_app_close = conn.peer_error().map(|e| e.is_app).unwrap_or(false);

        if local_app_close {
            ctx.conn_finish_success += 1;
        } else if peer_app_close {
            ctx.concurrent_conns -= 1;
            ctx.conn_finish_success += 1;
        } else {
            debug!(
                "{} connection failed — local: {:?}, peer: {:?}",
                conn.trace_id(),
                conn.local_error(),
                conn.peer_error()
            );
            ctx.conn_finish_failed += 1;
            ctx.concurrent_conns -= 1;
        }
    }

    fn on_stream_created(&mut self, conn: &mut Connection, stream_id: u64) {
        debug!("{} stream {} created", conn.trace_id(), stream_id);
        _ = conn.stream_want_read(stream_id, true);
    }

    fn on_stream_readable(&mut self, conn: &mut Connection, stream_id: u64) {
        let idx = conn.index().unwrap();
        loop {
            match conn.stream_read(stream_id, &mut self.recv_buf) {
                Ok((0, _)) | Err(Error::Done) => break,
                Ok((n, fin)) => {
                    if let Some(recv) = self.receivers.borrow_mut().get_mut(&idx) {
                        recv.bytes_received += n as u64;
                    }
                    debug!("{} stream {} +{} B fin={}", conn.trace_id(), stream_id, n, fin);
                }
                Err(e) => {
                    error!("{} stream_read {}: {:?}", conn.trace_id(), stream_id, e);
                    break;
                }
            }
        }
    }

    fn on_stream_writable(&mut self, conn: &mut Connection, stream_id: u64) {
        _ = conn.stream_want_write(stream_id, false);
    }

    fn on_stream_closed(&mut self, conn: &mut Connection, stream_id: u64) {
        debug!("{} stream {} closed", conn.trace_id(), stream_id);
        let idx = conn.index().unwrap();
        let done = {
            let mut receivers = self.receivers.borrow_mut();
            if let Some(recv) = receivers.get_mut(&idx) {
                recv.streams_finished += 1;
                recv.streams_finished >= recv.streams_opened && recv.streams_opened > 0
            } else {
                false
            }
        };
        if done {
            // All streams finished: close the connection gracefully.
            self.worker_ctx.borrow_mut().concurrent_conns -= 1;
            match conn.close(true, 0x00, b"done") {
                Ok(_) | Err(Error::Done) => {}
                Err(e) => error!("{} close: {:?}", conn.trace_id(), e),
            }
        }
    }

    fn on_new_token(&mut self, _conn: &mut Connection, _token: Vec<u8>) {}
}

// ─────────────────────────────── Entry point ─────────────────────────────────

/// Parse a bandwidth string like "500M", "1G", "2.5G" into bytes/sec.
/// Input is in bits/sec with optional SI suffix (K=1 000, M=1 000 000, G=1 000 000 000).
fn parse_bandwidth_cli(s: &str) -> std::result::Result<u64, String> {
    let s = s.trim();
    if s == "0" {
        return Ok(0);
    }
    let (num, mul): (&str, u64) =
        if let Some(p) = s.strip_suffix(['G', 'g']) { (p, 1_000_000_000) }
        else if let Some(p) = s.strip_suffix(['M', 'm']) { (p, 1_000_000) }
        else if let Some(p) = s.strip_suffix(['K', 'k']) { (p, 1_000) }
        else { (s, 1) };
    let f: f64 = num.parse().map_err(|_| format!("invalid bandwidth '{}'", s))?;
    let bits_per_sec = (f * mul as f64) as u64;
    Ok(bits_per_sec / 8) // convert bits/sec → bytes/sec
}

fn process_option(option: &mut ClientOpt) -> Result<()> {
    env_logger::builder()
        .target(tquic_tools::log_target(&option.log_file)?)
        .filter_level(option.log_level)
        .format_timestamp_millis()
        .init();
    if let Some(dir) = &option.qlog_dir {
        create_dir_all(dir)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut option = ClientOpt::parse();
    process_option(&mut option)?;

    let bw_str = if option.bandwidth == 0 {
        "unlimited".to_string()
    } else {
        format!("{:.1} Mbit/s", option.bandwidth as f64 * 8.0 / 1e6)
    };
    info!(
        "Connecting to {:?}  duration={}s streams_per_conn={} bandwidth={} CC={:?} multipath={}",
        option.connect_to,
        option.duration,
        option.streams_per_conn,
        bw_str,
        option.congestion_control_algor,
        option.enable_multipath,
    );

    let mut client = Client::new(option)?;
    client.start();
    Ok(())
}
