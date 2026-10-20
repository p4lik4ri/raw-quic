// Per-second traffic metrics collector shared by all non-LinUCB schedulers.
//
// Embeds in each scheduler struct. Call `record(best_pid, paths)` once per
// `on_select`. Once per second it emits a flat JSON line into `metrics`,
// using the same wandb section/naming convention as the LinUCB scheduler:
//
//   traffic/    — % of selections routed to each path
//   latency/    — smoothed RTT (ms) per path
//   throughput/ — actual send rate (Mbps) per path
//   congestion/ — cwnd (KB), bytes-in-flight (KB), pacing (Mbps)
//   loss/       — cumulative sent / lost packet counts

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::connection::path::PathMap;

pub struct TrafficMetricsCollector {
    start_time:      Instant,
    start_unix_secs: u64,
    last_log:        Option<Instant>,
    prev_sent_bytes: Vec<u64>,
    window_counts:   Vec<u64>,
    window_total:    u64,
    pub metrics:     Vec<String>,
}

impl TrafficMetricsCollector {
    pub fn new() -> Self {
        let unix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            start_time:      Instant::now(),
            start_unix_secs: unix,
            last_log:        None,
            prev_sent_bytes: Vec::new(),
            window_counts:   Vec::new(),
            window_total:    0,
            metrics:         Vec::new(),
        }
    }

    /// Return (elapsed_seconds_since_start, unix_timestamp_seconds_now)
    /// using this collector's start clock.
    pub fn elapsed_and_timestamp(&self, now: Instant) -> (u64, u64) {
        let elapsed_s = now.duration_since(self.start_time).as_secs();
        let timestamp = self.start_unix_secs + elapsed_s;
        (elapsed_s, timestamp)
    }

    /// Record a path selection and emit a JSONL line once per second.
    pub fn record(&mut self, best_pid: usize, paths: &mut PathMap) {
        if best_pid >= self.window_counts.len() {
            self.window_counts.resize(best_pid + 1, 0);
        }
        self.window_counts[best_pid] += 1;
        self.window_total += 1;

        let now = Instant::now();
        let do_snapshot = self
            .last_log
            .map(|t| now.duration_since(t) >= Duration::from_secs(1))
            .unwrap_or(true);
        if !do_snapshot {
            return;
        }

        let elapsed_s = now.duration_since(self.start_time).as_secs();
        let timestamp = self.start_unix_secs + elapsed_s;
        let step      = self.metrics.len() as u64;
        let total     = self.window_total.max(1);

        let mut jline = format!(
            "{{\"_step\":{step},\"_timestamp\":{timestamp},\"t\":{elapsed_s}"
        );

        for (pid, path) in paths.iter_mut() {
            let rtt_us  = path.recovery.rtt.smoothed_rtt().as_micros() as u64;
            let rtt_ms  = rtt_us as f64 / 1000.0;
            let cwnd    = path.recovery.congestion.congestion_window();
            let bif     = path.recovery.bytes_in_flight;
            let pacing  = path.recovery.congestion.pacing_rate().unwrap_or(0);
            let sent_b  = path.recovery.stats.sent_bytes;
            let sent_pk = path.recovery.stats.sent_count;
            let lost_pk = path.recovery.stats.lost_count;

            let cnt = self.window_counts.get(pid).copied().unwrap_or(0);
            let pct = cnt as f64 * 100.0 / total as f64;

            if pid >= self.prev_sent_bytes.len() {
                self.prev_sent_bytes.resize(pid + 1, 0);
            }
            let delta_bytes = sent_b.saturating_sub(self.prev_sent_bytes[pid]);
            self.prev_sent_bytes[pid] = sent_b;

            let tput_mbps   = delta_bytes as f64 * 8.0 / 1_000_000.0;
            let pacing_mbps = pacing     as f64 * 8.0 / 1_000_000.0;
            let cwnd_kb     = cwnd       as f64 / 1024.0;
            let inflight_kb = bif        as f64 / 1024.0;

            jline.push_str(&format!(
                ",\"traffic/path{pid}_pct\":{pct:.2}\
                 ,\"latency/path{pid}_rtt_ms\":{rtt_ms:.3}\
                 ,\"throughput/path{pid}_mbps\":{tput_mbps:.3}\
                 ,\"congestion/path{pid}_cwnd_KB\":{cwnd_kb:.1}\
                 ,\"congestion/path{pid}_inflight_KB\":{inflight_kb:.1}\
                 ,\"congestion/path{pid}_pacing_mbps\":{pacing_mbps:.3}\
                 ,\"loss/path{pid}_sent\":{sent_pk}\
                 ,\"loss/path{pid}_lost\":{lost_pk}",
            ));
        }

        jline.push('}');
        self.metrics.push(jline);

        self.last_log = Some(now);
        for c in &mut self.window_counts {
            *c = 0;
        }
        self.window_total = 0;
    }
}
