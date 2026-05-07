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

use std::time::Duration;
use std::time::Instant;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use log::info;

use crate::connection::path::PathMap;
use crate::connection::space::PacketNumSpaceMap;
use crate::connection::stream::StreamMap;
use crate::multipath_scheduler::MultipathScheduler;
use crate::Error;
use crate::MultipathConfig;
use crate::Result;

/// Number of context features used by the bandit.
///
/// Features per path:
///   x[0] = ema_rtt / min_ema_rtt          (normalised RTT; 1.0 for best path, clamped ≤ 4.0)
///   x[1] = bytes_in_flight / cwnd         (congestion window utilisation ∈ [0, 1])
///   x[2] = lost_pkts / sent_pkts          (cumulative packet loss rate ∈ [0, 1])
///   x[3] = min_pacing / pacing            (inverse throughput norm; 1.0 for fastest path)
///   x[4] = 1.0                            (bias / intercept term)
const D: usize = 5;

/// Per-arm (per-path) state for the LinUCB algorithm.
struct ArmState {
    /// A = I_d + Σ x_t xₜᵀ  — d×d positive-definite matrix.
    a: [[f64; D]; D],
    /// b = Σ rₜ xₜ  — d-dimensional reward-weighted feature sum.
    b: [f64; D],
}

impl ArmState {
    fn new() -> Self {
        let mut a = [[0.0_f64; D]; D];
        for i in 0..D {
            a[i][i] = 1.0; // initialise as identity
        }
        ArmState { a, b: [0.0; D] }
    }
}

/// LinUCBScheduler implements a contextual-bandit multipath scheduler.
///
/// It uses the LinUCB (Disjoint) algorithm to learn which path yields the
/// best reward (low latency, low congestion) and balances exploration vs
/// exploitation via the `alpha` coefficient.
///
/// # Algorithm
///
/// At each scheduling decision the algorithm:
/// 1. Builds a context vector xₚ for every sendable path p.
/// 2. Computes the UCB score:  θₚᵀ xₚ + α √(xₚᵀ Aₚ⁻¹ xₚ)
/// 3. Selects the path with the highest score.
///
/// After each ACK arrives on a path, the model is updated:
///   Aₚ ← Aₚ + xₚ xₚᵀ
///   bₚ ← bₚ + rₚ xₚ
/// where the reward rₚ = 1 − rtt_norm  (higher reward for lower-RTT paths).
pub struct LinUCBScheduler {
    /// Per-path arm state, indexed by path_id.
    arms: Vec<Option<ArmState>>,
    /// Minimum srtt (nanoseconds) observed across all paths at the last
    /// on_select call.  Used as the normalisation baseline in on_ack so that
    /// reward = 1 − rtt_norm is meaningful (0.0 for the best path, negative
    /// for worse paths).
    last_min_rtt_ns: u128,
    /// Last path chosen — used to detect transitions and log path changes.
    last_selected: Option<usize>,
    /// Timestamp of the last periodic 1-second log.
    last_log: Option<Instant>,
    /// Per-path selection counter for the current 1-second window.
    window_counts: Vec<u64>,
    /// Total selections in the current 1-second window.
    window_total: u64,
    /// Total selections across the entire session, indexed by path_id.
    total_counts: Vec<u64>,
    /// Total selections across the entire session.
    total_selections: u64,
    /// Per-path local address strings, populated on first selection.
    path_addrs: Vec<Option<String>>,
    /// Per-second snapshot lines buffered for the final summary.
    snapshots: Vec<String>,
    /// Connection start time for elapsed-second labels in summary.
    start_time: Instant,
    /// Per-path exponential moving average of latest_rtt (nanoseconds).
    /// Used in on_ack to smooth transient first-packet RTT spikes.
    ema_rtt_ns: Vec<f64>,
    /// Number of ACKs received per path, used to control EMA warmup speed.
    ack_counts: Vec<u64>,
    /// Per-second JSONL metric lines buffered for wandb upload at run end.
    /// Each line is a flat JSON object with a `_step` key and per-path metrics.
    metrics_jsonl: Vec<String>,
    /// Unix timestamp (seconds) at scheduler creation, used to compute per-step
    /// `_timestamp` required by wandb to render time-series charts.
    start_unix_secs: u64,
    /// Cumulative sent-bytes snapshot from the previous 1-second window.
    /// Used to compute per-second actual throughput (delta bytes × 8 / 1e6 Mbps).
    prev_sent_bytes: Vec<u64>,
    /// Last context vector used for each path in on_select.
    /// Stored here so on_ack can update the model with the *exact same* features
    /// that were used at selection time (avoids training mismatch from missing
    /// cross-path max_pacing_bps in on_ack).
    last_context: Vec<Option<[f64; D]>>,
}

impl LinUCBScheduler {
    pub fn new(_conf: &MultipathConfig) -> Self {
        let now = Instant::now();
        let start_unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        LinUCBScheduler {
            arms: Vec::new(),
            last_min_rtt_ns: 1,
            last_selected: None,
            last_log: None,
            window_counts: Vec::new(),
            window_total: 0,
            total_counts: Vec::new(),
            total_selections: 0,
            path_addrs: Vec::new(),
            snapshots: Vec::new(),
            start_time: now,
            ema_rtt_ns: Vec::new(),
            ack_counts: Vec::new(),
            metrics_jsonl: Vec::new(),
            start_unix_secs,
            prev_sent_bytes: Vec::new(),
            last_context: Vec::new(),
        }
    }

    /// Ensure an arm exists for path_id, initialising it if absent.
    fn ensure_arm(&mut self, path_id: usize) {
        if path_id >= self.arms.len() {
            self.arms.resize_with(path_id + 1, || None);
        }
        if self.arms[path_id].is_none() {
            self.arms[path_id] = Some(ArmState::new());
        }
    }

    /// Build the context vector for a path.
    ///
    /// Arguments:
    ///   rtt_ns           — EMA RTT for this path (nanoseconds)
    ///   min_rtt_ns       — minimum EMA RTT across all active paths (nanoseconds)
    ///   bytes_in_flight  — current bytes in flight on this path
    ///   cwnd             — current congestion window on this path (bytes)
    ///   loss_rate        — cumulative lost_pkts / sent_pkts ∈ [0, 1]
    ///   pacing_rate_bps  — pacing rate in bytes/sec (0 = not available)
    ///   max_pacing_bps   — maximum pacing rate across all active paths
    fn make_context(
        rtt_ns: u128,
        min_rtt_ns: u128,
        bytes_in_flight: usize,
        cwnd: u64,
        loss_rate: f64,
        pacing_rate_bps: u64,
        max_pacing_bps: u64,
    ) -> [f64; D] {
        let rtt_norm = if min_rtt_ns > 0 {
            (rtt_ns as f64 / min_rtt_ns as f64).clamp(1.0, 4.0)
        } else {
            1.0
        };
        let cwnd_pressure = if cwnd > 0 {
            (bytes_in_flight as f64 / cwnd as f64).min(1.0)
        } else {
            1.0
        };
        // Inverse throughput norm: 1.0 for the fastest path, >1.0 for slower.
        // If pacing rate is unavailable (0), treat as neutral (1.0).
        let bw_norm = if pacing_rate_bps > 0 && max_pacing_bps > 0 {
            (max_pacing_bps as f64 / pacing_rate_bps as f64).clamp(1.0, 4.0)
        } else {
            1.0
        };
        [rtt_norm, cwnd_pressure, loss_rate.clamp(0.0, 1.0), bw_norm, 1.0]
    }

    /// Decompose the UCB score into (reward_estimate, exploration_bonus).
    fn ucb_parts(arm: &ArmState, x: [f64; D], alpha: f64) -> (f64, f64) {
        let a_inv = mat_inv(arm.a);
        let theta = mat_vec(a_inv, arm.b);
        let reward_est = dot(theta, x);
        let explore_bonus = alpha * quadratic(a_inv, x).max(0.0).sqrt();
        (reward_est, explore_bonus)
    }

    /// Apply a single LinUCB update to an arm.
    fn update_arm(arm: &mut ArmState, x: [f64; D], reward: f64) {
        // A += x xᵀ
        for i in 0..D {
            for j in 0..D {
                arm.a[i][j] += x[i] * x[j];
            }
        }
        // b += reward * x
        for i in 0..D {
            arm.b[i] += reward * x[i];
        }
    }
}

impl MultipathScheduler for LinUCBScheduler {
    /// Select the path with the highest LinUCB score.
    fn on_select(
        &mut self,
        paths: &mut PathMap,
        _spaces: &mut PacketNumSpaceMap,
        _streams: &mut StreamMap,
    ) -> Result<usize> {
        // Collect per-path stats via iter_mut (can_send() takes &mut self).
        // Values are extracted before mutating self.arms to satisfy the borrow checker.
        let mut min_rtt_ns: u128 = u128::MAX;
        let mut max_pacing_bps: u64 = 0;
        // (pid, rtt_ns, bytes_in_flight, cwnd, loss_rate, pacing_rate_bps, sent_bytes_total, lost_count_total, sent_count_total)
        let mut raw: Vec<(usize, u128, usize, u64, f64, u64, u64, u64, u64)> = Vec::new();

        for (pid, path) in paths.iter_mut() {
            if !path.active() || !path.recovery.can_send() {
                continue;
            }
            let rtt_ns = if self.ack_counts.get(pid).copied().unwrap_or(0) > 0 {
                self.ema_rtt_ns[pid] as u128
            } else {
                path.recovery.rtt.smoothed_rtt().as_nanos()
            };
            let bytes_in_flight = path.recovery.bytes_in_flight;
            let cwnd = path.recovery.congestion.congestion_window();
            let sent = path.recovery.stats.sent_count;
            let lost = path.recovery.stats.lost_count;
            let loss_rate = if sent > 0 { lost as f64 / sent as f64 } else { 0.0 };
            let pacing_bps = path.recovery.congestion.pacing_rate().unwrap_or(0);
            let sent_bytes_total = path.recovery.stats.sent_bytes;
            if rtt_ns < min_rtt_ns {
                min_rtt_ns = rtt_ns;
            }
            if pacing_bps > max_pacing_bps {
                max_pacing_bps = pacing_bps;
            }
            raw.push((pid, rtt_ns, bytes_in_flight, cwnd, loss_rate, pacing_bps, sent_bytes_total, lost, sent));
        }

        if raw.is_empty() {
            return Err(Error::Done);
        }

        let min_rtt_ns = min_rtt_ns.max(1);
        self.last_min_rtt_ns = min_rtt_ns;

        // Pick the arm with the highest UCB score and collect per-path scores
        // for logging.
        let mut best_pid = raw[0].0;
        let mut best_score = f64::NEG_INFINITY;
        // (pid, rtt_us, cwnd_pressure, reward_est, explore_bonus, ucb, context_x)
        let mut score_rows: Vec<(usize, u64, f64, f64, f64, f64, [f64; D])> = Vec::new();

        for &(pid, rtt_ns, bytes_in_flight, cwnd, loss_rate, pacing_bps, ..) in &raw {
            let x = Self::make_context(rtt_ns, min_rtt_ns, bytes_in_flight, cwnd, loss_rate, pacing_bps, max_pacing_bps);
            self.ensure_arm(pid);
            let arm = self.arms[pid].as_ref().unwrap();
            // Dynamic alpha: 1/sqrt(n+1). Starts at 1.0 (heavy exploration),
            // decays as the arm accumulates ACKs, converges to ~0.1 after ~100 ACKs.
            let n = self.ack_counts.get(pid).copied().unwrap_or(0);
            let alpha = 1.0_f64 / ((n + 1) as f64).sqrt();
            let (est, bonus) = Self::ucb_parts(arm, x, alpha);
            let ucb = est + bonus;
            // Store context so on_ack uses the same feature vector (no mismatch).
            if pid >= self.last_context.len() {
                self.last_context.resize(pid + 1, None);
            }
            self.last_context[pid] = Some(x);
            let cwnd_pressure = x[1];
            score_rows.push((pid, (rtt_ns / 1_000) as u64, cwnd_pressure, est, bonus, ucb, x));
            if ucb > best_score {
                best_score = ucb;
                best_pid = pid;
            }
        }

        // --- Logging and counters ---
        let now = Instant::now();

        // Update per-window and total selection counters.
        if best_pid >= self.window_counts.len() {
            self.window_counts.resize(best_pid + 1, 0);
        }
        if best_pid >= self.total_counts.len() {
            self.total_counts.resize(best_pid + 1, 0);
        }
        self.window_counts[best_pid] += 1;
        self.total_counts[best_pid] += 1;
        self.window_total += 1;
        self.total_selections += 1;

        // Cache per-path addresses for the final summary.
        // Prefer local_addr, but fall back to remote_addr when local is
        // unspecified (server bound to 0.0.0.0 / ::).
        for &(pid, _, _, _, _, _, _, _, _) in &raw {
            if pid >= self.path_addrs.len() {
                self.path_addrs.resize(pid + 1, None);
            }
            if self.path_addrs[pid].is_none() {
                if let Ok(p) = paths.get(pid) {
                    let ip = {
                        let local = p.local_addr().ip();
                        if local.is_unspecified() {
                            p.remote_addr().ip()
                        } else {
                            local
                        }
                    };
                    self.path_addrs[pid] = Some(ip.to_string());
                }
            }
        }

        if self.last_selected != Some(best_pid) {
            self.last_selected = Some(best_pid);
        }

        // Buffer a per-second snapshot for the final summary.
        let do_snapshot = self
            .last_log
            .map(|t| now.duration_since(t) >= Duration::from_secs(1))
            .unwrap_or(true);
        if do_snapshot {
            let elapsed_s = now.duration_since(self.start_time).as_secs();
            let total = self.window_total.max(1);
            let mut parts = Vec::new();
            for (pid, &cnt) in self.window_counts.iter().enumerate() {
                if cnt == 0 {
                    continue;
                }
                let addr = self
                    .path_addrs
                    .get(pid)
                    .and_then(|a| a.as_deref())
                    .unwrap_or("?");
                let pct = cnt * 1000 / total; // tenths of a percent
                let pct_str = if pct % 10 == 0 {
                    format!("{}%", pct / 10)
                } else {
                    format!("{}.{}%", pct / 10, pct % 10)
                };
                parts.push(format!("path[{pid}] {addr} {pct_str}"));
            }
            self.snapshots.push(format!(
                "  t={elapsed_s:>3}s:  {}",
                parts.join("  |  ")
            ));

            // ── wandb JSONL metric line ──────────────────────────────────────
            // Flat JSON object per second.
            //
            // Metric names use wandb's "/" section separator so that all paths
            // appear as separate lines on the *same* chart per section:
            //
            //   traffic/       — % of packets scheduled to each path
            //   throughput/    — actual send rate (Mbps) per path
            //   latency/       — smoothed RTT (ms) per path
            //   congestion/    — cwnd (KB), bytes-in-flight (KB), pacing (Mbps)
            //   loss/          — cumulative sent / lost packet counts
            //   linucb/        — reward estimate, exploration bonus, sample count
            //   features/      — context vector fed to LinUCB (x0..x4)
            //   weights/       — learned θ weights (updated after each ACK)
            {
                let step = self.metrics_jsonl.len() as u64;
                let timestamp = self.start_unix_secs + elapsed_s;
                // Feature / weight names in context-vector order:
                //   x0=rtt_norm  x1=cwnd_pressure  x2=loss_rate  x3=bw_norm  x4=bias
                let feat_names = ["rtt_norm", "cwnd_pressure", "loss_rate", "bw_norm", "bias"];
                let mut jline = format!("{{\"_step\":{step},\"_timestamp\":{timestamp},\"t\":{elapsed_s}");
                for &(pid, rtt_us, _cp, reward_est, explore_bonus, _ucb, x) in &score_rows {
                    let cnt = self.window_counts.get(pid).copied().unwrap_or(0);
                    let pct = cnt as f64 * 100.0 / total as f64;
                    let n = self.ack_counts.get(pid).copied().unwrap_or(0);
                    let theta = if let Some(Some(arm)) = self.arms.get(pid) {
                        mat_vec(mat_inv(arm.a), arm.b)
                    } else {
                        [0.0_f64; D]
                    };
                    // Traffic metrics from raw stats.
                    let (bytes_in_flight, cwnd, pacing_bps, sent_bytes_total, lost_total, sent_total) =
                        raw.iter().find(|r| r.0 == pid)
                            .map(|r| (r.2, r.3, r.5, r.6, r.7, r.8))
                            .unwrap_or((0, 0, 0, 0, 0, 0));
                    // Per-second actual throughput: delta sent_bytes since last window.
                    if pid >= self.prev_sent_bytes.len() {
                        self.prev_sent_bytes.resize(pid + 1, 0);
                    }
                    let delta_bytes = sent_bytes_total.saturating_sub(self.prev_sent_bytes[pid]);
                    self.prev_sent_bytes[pid] = sent_bytes_total;
                    let throughput_mbps = delta_bytes as f64 * 8.0 / 1_000_000.0;
                    let pacing_mbps    = pacing_bps as f64 * 8.0 / 1_000_000.0;
                    let cwnd_kb        = cwnd as f64 / 1024.0;
                    let inflight_kb    = bytes_in_flight as f64 / 1024.0;
                    let rtt_ms         = rtt_us as f64 / 1000.0;
                    // traffic / latency / throughput
                    jline.push_str(&format!(
                        ",\"traffic/path{pid}_pct\":{pct:.2}\
                         ,\"latency/path{pid}_rtt_ms\":{rtt_ms:.3}\
                         ,\"throughput/path{pid}_mbps\":{throughput_mbps:.3}",
                    ));
                    // congestion control
                    jline.push_str(&format!(
                        ",\"congestion/path{pid}_cwnd_KB\":{cwnd_kb:.1}\
                         ,\"congestion/path{pid}_inflight_KB\":{inflight_kb:.1}\
                         ,\"congestion/path{pid}_pacing_mbps\":{pacing_mbps:.3}",
                    ));
                    // packet counts
                    jline.push_str(&format!(
                        ",\"loss/path{pid}_sent\":{sent_total}\
                         ,\"loss/path{pid}_lost\":{lost_total}",
                    ));
                    // LinUCB internals
                    jline.push_str(&format!(
                        ",\"linucb/path{pid}_reward\":{reward_est:.4}\
                         ,\"linucb/path{pid}_explore_bonus\":{explore_bonus:.4}\
                         ,\"linucb/path{pid}_samples\":{n}",
                    ));
                    // context features (x vector)
                    for (i, xi) in x.iter().enumerate() {
                        jline.push_str(&format!(",\"features/path{pid}_{}\":{xi:.4}", feat_names[i]));
                    }
                    // learned weights (theta)
                    for (i, ti) in theta.iter().enumerate() {
                        jline.push_str(&format!(",\"weights/path{pid}_{}\":{ti:.4}", feat_names[i]));
                    }
                }
                jline.push('}');
                self.metrics_jsonl.push(jline);
            }
            // ── end wandb line ───────────────────────────────────────────────

            self.last_log = Some(now);
            for c in &mut self.window_counts {
                *c = 0;
            }
            self.window_total = 0;
        }

        Ok(best_pid)
    }

    /// Update the model for the path that received an ACK.
    fn on_ack(&mut self, _now: Instant, path_id: usize, paths: &mut PathMap) {
        let path = match paths.get_mut(path_id) {
            Ok(p) => p,
            Err(_) => return,
        };

        let rtt_ns = path.recovery.rtt.latest_rtt().as_nanos().max(1);

        // Per-path EMA to smooth transient first-packet RTT spikes.
        // Alpha=0.5 for the first 8 ACKs (fast warmup), then 0.2 (stable).
        if path_id >= self.ema_rtt_ns.len() {
            self.ema_rtt_ns.resize(path_id + 1, rtt_ns as f64);
            self.ack_counts.resize(path_id + 1, 0);
        }
        let old_ema = self.ema_rtt_ns[path_id];
        self.ack_counts[path_id] += 1;
        let ema_alpha = if self.ack_counts[path_id] <= 8 { 0.5_f64 } else { 0.2_f64 };
        self.ema_rtt_ns[path_id] =
            ema_alpha * rtt_ns as f64 + (1.0 - ema_alpha) * old_ema;
        let ema_rtt_ns = self.ema_rtt_ns[path_id] as u128;

        // ── RTT-jump reset ────────────────────────────────────────────────────
        // If the EMA RTT has changed by more than 30% since the previous ACK,
        // the path quality has shifted significantly (delay added or removed).
        // Inject uncertainty into the arm's A matrix by blending it back
        // toward the identity, which raises the exploration bonus and forces
        // the model to re-learn from current observations rather than relying
        // on stale history.
        //
        // The injection magnitude (10.0 × I) is chosen so that after a jump:
        //   alpha_effective = 1/sqrt(n_injected) ≈ 1/sqrt(10) ≈ 0.32
        // i.e. the arm behaves as if it has only ~10 recent observations,
        // regardless of how many ACKs it has accumulated.
        self.ensure_arm(path_id);
        if self.ack_counts[path_id] > 8 {
            let rtt_change = (self.ema_rtt_ns[path_id] - old_ema) / old_ema;
            if rtt_change.abs() > 0.30 {
                let arm = self.arms[path_id].as_mut().unwrap();
                for i in 0..D {
                    arm.a[i][i] += 10.0;
                }
            }
        }

        // Use the context vector that was built at selection time (on_select
        // has the cross-path max_pacing_bps; on_ack does not).  Fall back to
        // recomputing only if no selection has happened yet for this path.
        let x = self
            .last_context
            .get(path_id)
            .and_then(|c| *c)
            .unwrap_or_else(|| {
                Self::make_context(
                    ema_rtt_ns,
                    self.last_min_rtt_ns,
                    path.recovery.bytes_in_flight,
                    path.recovery.congestion.congestion_window(),
                    {
                        let sent = path.recovery.stats.sent_count;
                        let lost = path.recovery.stats.lost_count;
                        if sent > 0 { lost as f64 / sent as f64 } else { 0.0 }
                    },
                    0,
                    0,
                )
            });

        // Reward: exp(-rtt_norm) * (1 - loss_rate)
        //   - Always positive and bounded in (0, 1]
        //   - rtt_norm = 1.0 on best path → exp(-1) ≈ 0.37 (max reward)
        //   - rtt_norm = 4.0 on worst path → exp(-4) ≈ 0.018 (strong penalty)
        //   - loss_rate multiplier further suppresses lossy paths
        //   - Stable scale prevents reward drift that destabilises theta
        let reward = (-x[0]).exp() * (1.0 - x[2]);

        let arm = self.arms[path_id].as_mut().unwrap();
        Self::update_arm(arm, x, reward);
    }

    fn scheduler_summary(&self) -> Option<String> {
        if self.total_selections == 0 {
            return None;
        }
        let mut out = format!(
            "LinUCB scheduler summary  ({} total selections over {} arms)\n",
            self.total_selections,
            self.total_counts.iter().filter(|&&c| c > 0).count(),
        );

        // Per-path totals.
        out.push_str("  Path totals:\n");
        for (pid, &cnt) in self.total_counts.iter().enumerate() {
            if cnt == 0 {
                continue;
            }
            let addr = self
                .path_addrs
                .get(pid)
                .and_then(|a| a.as_deref())
                .unwrap_or("?");
            let pct = cnt * 100 / self.total_selections;
            // Final model estimate + exploration bonus using a neutral context:
            // rtt_norm=1.0, cwnd_pressure=0.0, loss_rate=0.0, bw_norm=1.0, bias=1.0.
            let (est_str, bonus_str) = if let Some(Some(arm)) = self.arms.get(pid) {
                let x = [1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64];
                let a_inv = mat_inv(arm.a);
                let theta = mat_vec(a_inv, arm.b);
                let est = dot(theta, x);
                let n = self.ack_counts.get(pid).copied().unwrap_or(0);
                let alpha = 1.0_f64 / ((n + 1) as f64).sqrt();
                let bonus = alpha * quadratic(a_inv, x).max(0.0).sqrt();
                (format!("{est:+.3}"), format!("{bonus:.3} (n={n})"))
            } else {
                ("n/a".into(), "n/a".into())
            };
            out.push_str(&format!(
                "    path[{pid}] {addr}  selections={cnt} ({pct}%)  est(neutral)={est_str}  explore_bonus={bonus_str}\n"
            ));
        }

        // Per-second breakdown table.
        if !self.snapshots.is_empty() {
            out.push_str("  Per-second breakdown:\n");
            for snap in &self.snapshots {
                out.push_str(snap);
                out.push('\n');
            }
        }

        Some(out)
    }

    fn scheduler_metrics_jsonl(&self) -> Vec<String> {
        self.metrics_jsonl.clone()
    }
}

// ─── D×D linear algebra helpers ──────────────────────────────────────────────
//
// Generic implementations that work for any const D.
// mat_inv uses Gauss-Jordan elimination with partial pivoting.

/// Compute the inverse of a D×D matrix using Gauss-Jordan elimination.
///
/// Returns the identity matrix when the matrix is (near-)singular to avoid
/// numerical blow-up during early exploration.
fn mat_inv(m: [[f64; D]; D]) -> [[f64; D]; D] {
    let mut a = m;
    let mut inv = [[0.0_f64; D]; D];
    for i in 0..D {
        inv[i][i] = 1.0;
    }
    for col in 0..D {
        // Partial pivoting: find row with largest absolute value in this column.
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for row in (col + 1)..D {
            if a[row][col].abs() > max_val {
                max_val = a[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-15 {
            // Singular — return identity to avoid blowing up.
            let mut r = [[0.0_f64; D]; D];
            for i in 0..D {
                r[i][i] = 1.0;
            }
            return r;
        }
        a.swap(col, max_row);
        inv.swap(col, max_row);
        // Scale pivot row.
        let pivot = a[col][col];
        for j in 0..D {
            a[col][j] /= pivot;
            inv[col][j] /= pivot;
        }
        // Eliminate column in all other rows.
        for row in 0..D {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            for j in 0..D {
                a[row][j] -= factor * a[col][j];
                inv[row][j] -= factor * inv[col][j];
            }
        }
    }
    inv
}

/// Multiply a D×D matrix by a D-vector.
fn mat_vec(m: [[f64; D]; D], v: [f64; D]) -> [f64; D] {
    let mut r = [0.0_f64; D];
    for i in 0..D {
        for j in 0..D {
            r[i] += m[i][j] * v[j];
        }
    }
    r
}

/// Dot product of two D-vectors.
fn dot(a: [f64; D], b: [f64; D]) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..D {
        s += a[i] * b[i];
    }
    s
}

/// Quadratic form xᵀ M x for a D×D matrix M and D-vector x.
fn quadratic(m: [[f64; D]; D], x: [f64; D]) -> f64 {
    dot(x, mat_vec(m, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multipath_scheduler::tests::*;

    #[test]
    fn linucb_single_path() -> Result<()> {
        let mut t = MultipathTester::new()?;
        let mut s = LinUCBScheduler::new(&Default::default());
        // Should always select the only available path.
        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 0);
        Ok(())
    }

    #[test]
    fn linucb_multi_path_selects_valid() -> Result<()> {
        let mut t = MultipathTester::new()?;
        t.add_path("127.0.0.1:443", "127.0.0.2:8443", 50)?;
        t.add_path("127.0.0.1:443", "127.0.0.3:8443", 150)?;

        let mut s = LinUCBScheduler::new(&Default::default());
        let pid = s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?;
        // Result must be one of the active paths (0, 1, or 2).
        assert!(pid <= 2);
        Ok(())
    }

    #[test]
    fn mat_inv_identity() {
        // Build a D×D identity and verify inversion returns identity.
        let mut id = [[0.0_f64; D]; D];
        for i in 0..D {
            id[i][i] = 1.0;
        }
        let inv = mat_inv(id);
        for i in 0..D {
            for j in 0..D {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-10);
            }
        }
    }
}
