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

/// Idle threshold (seconds) before a path is treated as quiescent.
/// While idle, the LinUCB covariance matrix `A` is periodically decayed
/// toward the identity, restoring uncertainty so the path is re-probed.
const IDLE_FORGET_THRESHOLD_SECS: f64 = 2.0;

/// Forgetting factor applied per IDLE_FORGET_THRESHOLD_SECS of inactivity.
/// A ratio < 1 pulls A back toward identity and shrinks b, increasing the
/// exploration bonus. Smaller ⇒ faster forgetting.
const IDLE_FORGET_LAMBDA: f64 = 0.92;

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
/// where the reward rₚ = exp(-rtt_norm) × (1 - cwnd_util).
///
/// # Exploration parameters
///
/// alpha(n) = max(alpha_floor, alpha_init / sqrt(n + 1))
///
/// - `alpha_init`  — initial exploration weight (default 1.0). Higher values
///   cause the scheduler to explore undersampled paths more aggressively at
///   the start. Setting this very large approximates Round Robin.
/// - `alpha_floor` — minimum exploration weight (default 0.15). Prevents
///   the scheduler from converging to pure exploitation even after many
///   samples, keeping a residual probe budget on each path. Setting this to
///   0.0 allows full exploitation; setting it very high approximates
///   Round Robin.
pub struct LinUCBScheduler {
    /// Per-path arm state, indexed by path_id.
    arms: Vec<Option<ArmState>>,
    /// Minimum srtt (nanoseconds) observed across all paths at the last
    /// on_select call. Used as the normalisation baseline in on_ack fallback
    /// context construction.
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
    /// Initial exploration coefficient α₀.  alpha(n) = max(alpha_floor, alpha_init/√(n+1)).
    /// Default: 1.0.  Increase to explore more aggressively; decrease for faster exploitation.
    pub alpha_init: f64,
    /// Minimum exploration coefficient (floor).  Keeps a residual probe budget per path.
    /// Default: 0.15.  Set to 0.0 for pure exploitation; set high to approximate Round Robin.
    pub alpha_floor: f64,
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
    /// Used to compute per-second actual throughput (delta bytes × 8 / 1e6 = Mbps).
    prev_sent_bytes: Vec<u64>,
    /// Last context used when each path was scored for selection.
    /// ACK-time updates reuse this context to keep LinUCB credit assignment consistent.
    last_contexts: Vec<Option<[f64; D]>>,
    /// Timestamp of the last time each path was selected by on_select.
    /// Used together with `last_forget_time` to detect path quiescence.
    last_select_time: Vec<Option<Instant>>,
    /// Timestamp of the last time the idle-forgetting decay was applied to a path.
    /// Prevents the per-call decay from being applied repeatedly within one window.
    last_forget_time: Vec<Option<Instant>>,
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
            alpha_init: 1.0,
            alpha_floor: 0.15,
            ema_rtt_ns: Vec::new(),
            ack_counts: Vec::new(),
            metrics_jsonl: Vec::new(),
            start_unix_secs,
            prev_sent_bytes: Vec::new(),
            last_contexts: Vec::new(),
            last_select_time: Vec::new(),
            last_forget_time: Vec::new(),
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

    /// Apply one decay step to an arm's (A, b) toward identity / zero.
    /// Used to restore exploration uncertainty after a path has been idle.
    fn forget_arm(arm: &mut ArmState, lambda: f64) {
        for i in 0..D {
            for j in 0..D {
                arm.a[i][j] *= lambda;
            }
            arm.a[i][i] += 1.0 - lambda;
            arm.b[i] *= lambda;
        }
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
        // (pid, rtt_ns, bytes_in_flight, cwnd, loss_rate, pacing_rate_bps, sent_bytes)
        let mut raw: Vec<(usize, u128, usize, u64, f64, u64, u64)> = Vec::new();

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
            let sent_bytes = path.recovery.stats.sent_bytes;
            if rtt_ns < min_rtt_ns {
                min_rtt_ns = rtt_ns;
            }
            if pacing_bps > max_pacing_bps {
                max_pacing_bps = pacing_bps;
            }
            raw.push((pid, rtt_ns, bytes_in_flight, cwnd, loss_rate, pacing_bps, sent_bytes));
        }

        if raw.is_empty() {
            return Err(Error::Done);
        }

        let min_rtt_ns = min_rtt_ns.max(1);
        self.last_min_rtt_ns = min_rtt_ns;

        // Capture a single timestamp used for idle-forgetting checks (below)
        // and for the per-second logging snapshot.
        let now = Instant::now();

        // Pick the arm with the highest UCB score and collect per-path scores
        // for logging.
        let mut best_pid = raw[0].0;
        let mut best_score = f64::NEG_INFINITY;
        // (pid, rtt_us, cwnd_pressure, reward_est, explore_bonus, ucb, context_x)
        let mut score_rows: Vec<(usize, u64, f64, f64, f64, f64, [f64; D])> = Vec::new();

        for &(pid, rtt_ns, bytes_in_flight, cwnd, loss_rate, pacing_bps, _sent_bytes) in &raw {
            self.ensure_arm(pid);

            // ── Idle-forgetting ────────────────────────────────────────
            // If this path has been idle longer than IDLE_FORGET_THRESHOLD_SECS
            // since both its last selection and its last forgetting step, decay
            // its (A, b) once toward (I, 0).  This restores exploration
            // uncertainty so the path can be re-probed after long inactivity.
            let last_select = self.last_select_time.get(pid).and_then(|t| *t);
            let last_forget = self.last_forget_time.get(pid).and_then(|t| *t);
            let idle_secs = last_select
                .map(|t| now.duration_since(t).as_secs_f64())
                .unwrap_or(f64::INFINITY);
            let since_forget = last_forget
                .map(|t| now.duration_since(t).as_secs_f64())
                .unwrap_or(f64::INFINITY);
            if idle_secs > IDLE_FORGET_THRESHOLD_SECS
                && since_forget > IDLE_FORGET_THRESHOLD_SECS
            {
                if let Some(arm) = self.arms[pid].as_mut() {
                    Self::forget_arm(arm, IDLE_FORGET_LAMBDA);
                }
                if pid >= self.last_forget_time.len() {
                    self.last_forget_time.resize(pid + 1, None);
                }
                self.last_forget_time[pid] = Some(now);
            }

            let x = Self::make_context(
                rtt_ns,
                min_rtt_ns,
                bytes_in_flight,
                cwnd,
                loss_rate,
                pacing_bps,
                max_pacing_bps,
            );
            let arm = self.arms[pid].as_ref().unwrap();
            // Dynamic alpha: 1/sqrt(n+1). Starts at 1.0 (heavy exploration),
            // decays as the arm accumulates ACKs, with a floor to keep probing paths.
            let n = self.ack_counts.get(pid).copied().unwrap_or(0);
            let alpha = (self.alpha_init / ((n + 1) as f64).sqrt()).max(self.alpha_floor);
            let (est, bonus) = Self::ucb_parts(arm, x, alpha);
            let ucb = est + bonus;
            let cwnd_pressure = x[1];
            score_rows.push((pid, (rtt_ns / 1_000) as u64, cwnd_pressure, est, bonus, ucb, x));
            if pid >= self.last_contexts.len() {
                self.last_contexts.resize(pid + 1, None);
            }
            self.last_contexts[pid] = Some(x);
            if ucb > best_score {
                best_score = ucb;
                best_pid = pid;
            }
        }

        // --- Logging and counters ---

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

        // Record the selection timestamp for quiescence on future on_select calls.
        if best_pid >= self.last_select_time.len() {
            self.last_select_time.resize(best_pid + 1, None);
        }
        self.last_select_time[best_pid] = Some(now);

        // Cache per-path addresses for the final summary.
        // Prefer local_addr, but fall back to remote_addr when local is
        // unspecified (server bound to 0.0.0.0 / ::).
        for &(pid, _, _, _, _, _, _) in &raw {
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

            // ── wandb JSONL metric line ────────────────────────────────────────
            // Flat JSON object per second.  Metric names use "p{pid}." prefix
            // so wandb groups them by path in the UI.
            // Features: x0=rtt_norm, x1=cwnd_p, x2=loss_rate, x3=bw_norm, x4=bias
            // Theta:    th0..th4 = learned LinUCB weights for each feature.
            //
            // Alpha sweep fields (for cross-run comparison):
            //   cfg.alpha_init  — the alpha_init value configured for this run
            //   cfg.alpha_floor — the alpha_floor value configured for this run
            //
            // Path dominance fields:
            //   dominant_path   — pid of the path with the highest selection share
            //   path_delta      — |p0.pct - p1.pct|, measures how skewed the split is
            //                     (100 = all traffic on one path, 0 = perfect split)
            {
                let step = self.metrics_jsonl.len() as u64;
                let timestamp = self.start_unix_secs + elapsed_s;
                let feat_names = ["rtt_norm", "cwnd_p", "loss_rate", "bw_norm", "bias"];

                // Compute dominant path and path_delta for the α-sweep plot.
                let dominant_path = score_rows
                    .iter()
                    .map(|&(pid, _, _, _, _, _, _)| {
                        let cnt = self.window_counts.get(pid).copied().unwrap_or(0);
                        (pid, cnt)
                    })
                    .max_by_key(|&(_, cnt)| cnt)
                    .map(|(pid, _)| pid)
                    .unwrap_or(0);
                let pcts: Vec<f64> = score_rows
                    .iter()
                    .map(|&(pid, _, _, _, _, _, _)| {
                        let cnt = self.window_counts.get(pid).copied().unwrap_or(0);
                        cnt as f64 * 100.0 / total as f64
                    })
                    .collect();
                let path_delta = if pcts.len() >= 2 {
                    (pcts[0] - pcts[1]).abs()
                } else {
                    100.0
                };

                let mut jline = format!(
                    "{{\"_step\":{step},\"_timestamp\":{timestamp},\"t\":{elapsed_s},\
                     \"cfg.alpha_init\":{:.4},\"cfg.alpha_floor\":{:.4},\
                     \"dominant_path\":{dominant_path},\"path_delta\":{path_delta:.2}",
                    self.alpha_init, self.alpha_floor,
                );
                for &(pid, rtt_us, _cp, reward_est, explore_bonus, _ucb, x) in &score_rows {
                    let cnt = self.window_counts.get(pid).copied().unwrap_or(0);
                    let pct = cnt as f64 * 100.0 / total as f64;
                    let n = self.ack_counts.get(pid).copied().unwrap_or(0);
                    // Per-path throughput: delta sent bytes over the last second → Mbps.
                    let sent_now = raw.iter()
                        .find(|&&(p, ..)| p == pid)
                        .map(|&(_, _, _, _, _, _, s)| s)
                        .unwrap_or(0);
                    if pid >= self.prev_sent_bytes.len() {
                        self.prev_sent_bytes.resize(pid + 1, 0);
                    }
                    let delta_bytes = sent_now.saturating_sub(self.prev_sent_bytes[pid]);
                    let throughput_mbps = (delta_bytes as f64 * 8.0) / 1_000_000.0;
                    self.prev_sent_bytes[pid] = sent_now;
                    let theta = if let Some(Some(arm)) = self.arms.get(pid) {
                        mat_vec(mat_inv(arm.a), arm.b)
                    } else {
                        [0.0_f64; D]
                    };
                    let alpha_logged = (self.alpha_init / ((n + 1) as f64).sqrt()).max(self.alpha_floor);
                    jline.push_str(&format!(
                        ",\"p{pid}.pct\":{pct:.2},\"p{pid}.rtt_us\":{rtt_us},\
                         \"p{pid}.throughput_mbps\":{throughput_mbps:.3},\
                         \"p{pid}.reward\":{reward_est:.4},\"p{pid}.bonus\":{explore_bonus:.4},\
                         \"p{pid}.n\":{n},\"p{pid}.alpha\":{alpha_logged:.4}",
                    ));
                    for (i, xi) in x.iter().enumerate() {
                        jline.push_str(&format!(",\"p{pid}.x_{}\":{xi:.4}", feat_names[i]));
                    }
                    for (i, ti) in theta.iter().enumerate() {
                        jline.push_str(&format!(",\"p{pid}.th_{}\":{ti:.4}", feat_names[i]));
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

        // ── RTT-jump adaptation ───────────────────────────────────────────────
        // If the EMA RTT has changed by more than 30% since the previous ACK,
        // the path quality has shifted significantly (delay added or removed).
        // Forget stale history so the model can re-learn after changing path
        // conditions rather than relying on old observations.
        self.ensure_arm(path_id);
        if self.ack_counts[path_id] > 8 {
            let rtt_change = (self.ema_rtt_ns[path_id] - old_ema) / old_ema;
            if rtt_change.abs() > 0.30 {
                let arm = self.arms[path_id].as_mut().unwrap();
                let lambda = 0.85_f64;
                for i in 0..D {
                    for j in 0..D {
                        arm.a[i][j] *= lambda;
                    }
                    arm.b[i] *= lambda;
                    arm.a[i][i] += 1.0 - lambda;
                }
            }
        }

        let x = self
            .last_contexts
            .get(path_id)
            .and_then(|ctx| *ctx)
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
                    path.recovery.congestion.pacing_rate().unwrap_or(0),
                    0,
                )
            });

        let reward = (-x[0]).exp() * (1.0 - x[1]);

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
            // Final model estimate + exploration bonus using a neutral context.
            let (est_str, bonus_str) = if let Some(Some(arm)) = self.arms.get(pid) {
                // Neutral context: rtt_norm=1.0 (best), cwnd_pressure=0.0,
                // loss_rate=0.0, bw_norm=1.0 (fastest), bias=1.0.
                let x = [1.0_f64, 0.0_f64, 0.0_f64, 1.0_f64, 1.0_f64];
                let a_inv = mat_inv(arm.a);
                let theta = mat_vec(a_inv, arm.b);
                let est = dot(theta, x);
                let n = self.ack_counts.get(pid).copied().unwrap_or(0);
                let alpha = (self.alpha_init / ((n + 1) as f64).sqrt()).max(self.alpha_floor);
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