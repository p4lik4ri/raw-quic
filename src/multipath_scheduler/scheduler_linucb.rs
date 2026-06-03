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
///   x[0] = ema_rtt / min_ema_rtt
///   x[1] = bytes_in_flight / cwnd
///   x[2] = lost_pkts / sent_pkts
///   x[3] = 1.0 (was bandwidth ratio; removed to prevent pacing feedback loop)
///   x[4] = 1.0 (bias)
const D: usize = 5;

/// Idle threshold before a path is considered stale.
const IDLE_FORGET_THRESHOLD_SECS: f64 = 2.0;

/// Forgetting factor for idle paths.
const IDLE_FORGET_LAMBDA: f64 = 0.92;

/// EMA smoothing for cwnd pressure.
/// 0.30 reacts to congestion roughly 3× faster than the previous 0.10,
/// so the model sees real-time path quality rather than a heavily lagged average.
const CWND_P_EMA_ALPHA: f64 = 0.30;

/// Exploration boost when path traffic share collapses.
///
/// If a path's traffic share falls below:
///     FAIRNESS_MIN_SHARE_PCT
///
/// then its exploration bonus is multiplied by:
///     FAIRNESS_BOOST
///
/// This prevents total path starvation while still allowing
/// the scheduler to strongly prefer better paths.
const FAIRNESS_MIN_SHARE_PCT: f64 = 5.0;
const FAIRNESS_BOOST: f64 = 2.0;

/// Per-arm state for LinUCB.
struct ArmState {
    a: [[f64; D]; D],
    b: [f64; D],
}

impl ArmState {
    fn new() -> Self {
        let mut a = [[0.0_f64; D]; D];
        for i in 0..D {
            a[i][i] = 1.0;
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
/// where the reward rₚ = exp(-rtt_norm).
/// cwnd_pressure is kept as a context feature so the model can learn its
/// correlation with path quality, but excluded from the reward label to
/// prevent the small-BDP starvation loop on fast-RTT paths.
///
/// # Exploration parameters
///
/// alpha(n) = max(alpha_floor, alpha_init / sqrt(n + 1))
///
/// - `alpha_init`  — initial exploration weight (default 1.0). Higher values
///   cause the scheduler to explore undersampled paths more aggressively at
///   the start. Setting this very large approximates Round Robin.
/// - `alpha_floor` — minimum exploration weight, derived as `alpha_init * 0.15`.
///   Scales with alpha_init so that raising --linucb-alpha increases both the
///   initial burst of exploration and the steady-state probe budget.
pub struct LinUCBScheduler {
    arms: Vec<Option<ArmState>>,
    last_min_rtt_ns: u128,

    last_selected: Option<usize>,
    last_log: Option<Instant>,

    window_counts: Vec<u64>,
    window_total: u64,

    total_counts: Vec<u64>,
    total_selections: u64,

    path_addrs: Vec<Option<String>>,
    snapshots: Vec<String>,
    start_time: Instant,

    pub alpha_init: f64,
    pub alpha_floor: f64,

    ema_rtt_ns: Vec<f64>,
    ack_counts: Vec<u64>,

    metrics_jsonl: Vec<String>,

    last_contexts: Vec<Option<[f64; D]>>,
    last_select_time: Vec<Option<Instant>>,
    last_forget_time: Vec<Option<Instant>>,
    ema_cwnd_pressure: Vec<Option<f64>>,
    forget_counts: Vec<u64>,

    last_window_sent: Vec<u64>,
    /// Snapshot of `lost_count` at each per-second window boundary, used to
    /// compute a windowed (non-cumulative) loss rate.
    last_window_lost: Vec<u64>,
    /// EMA-smoothed per-path loss rate computed from the last 1-second window.
    /// Updated once per second alongside `last_window_sent`.
    ema_loss_rate: Vec<f64>,
}

impl LinUCBScheduler {
    pub fn new(conf: &MultipathConfig) -> Self {
        let now = Instant::now();

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

            alpha_init: conf.linucb_alpha,
            // 0.05× gives a very small steady-state probe budget so that
            // low alpha_linucb → near-pure exploitation (5G dominant) and
            // high alpha_linucb → significant exploration (satellite sampled).
            // The 2× fairness boost below 5% share still prevents total starvation.
            alpha_floor: conf.linucb_alpha * 0.05,

            ema_rtt_ns: Vec::new(),
            ack_counts: Vec::new(),

            metrics_jsonl: Vec::new(),

            last_contexts: Vec::new(),
            last_select_time: Vec::new(),
            last_forget_time: Vec::new(),
            ema_cwnd_pressure: Vec::new(),
            forget_counts: Vec::new(),

            last_window_sent: Vec::new(),
            last_window_lost: Vec::new(),
            ema_loss_rate:    Vec::new(),
        }
    }

    fn ensure_arm(&mut self, path_id: usize) {
        if path_id >= self.arms.len() {
            self.arms.resize_with(path_id + 1, || None);
        }

        if self.arms[path_id].is_none() {
            self.arms[path_id] = Some(ArmState::new());
        }
    }

    fn make_context(
        rtt_ns: u128,
        min_rtt_ns: u128,
        cwnd_pressure: f64,
        loss_rate: f64,
        pacing_rate_bps: u64,
        max_pacing_bps: u64,
    ) -> [f64; D] {
        let rtt_norm = if min_rtt_ns > 0 {
            (rtt_ns as f64 / min_rtt_ns as f64).clamp(1.0, 4.0)
        } else {
            1.0
        };

        // bw_norm is fixed at 1.0 (neutral) to break the pacing feedback loop:
        // previously, starving a path caused its BBR pacing estimate to drop,
        // raising bw_norm, making the model rank it as worse — a self-reinforcing
        // starvation cycle. RTT and loss_rate are sufficient quality signals.
        let bw_norm = 1.0_f64;

        [
            rtt_norm,
            cwnd_pressure.clamp(0.0, 1.0),
            loss_rate.clamp(0.0, 1.0),
            bw_norm,
            1.0,
        ]
    }

    fn forget_arm(arm: &mut ArmState, lambda: f64) {
        for i in 0..D {
            for j in 0..D {
                arm.a[i][j] *= lambda;
            }

            arm.a[i][i] += 1.0 - lambda;
            arm.b[i] *= lambda;
        }
    }

    fn ucb_parts(arm: &ArmState, x: [f64; D], alpha: f64) -> (f64, f64) {
        let a_inv = mat_inv(arm.a);

        let theta = mat_vec(a_inv, arm.b);

        let reward_est = dot(theta, x);

        let explore_bonus =
            alpha * quadratic(a_inv, x).max(0.0).sqrt();

        (reward_est, explore_bonus)
    }

    fn update_arm(
        arm: &mut ArmState,
        x: [f64; D],
        reward: f64,
    ) {
        for i in 0..D {
            for j in 0..D {
                arm.a[i][j] += x[i] * x[j];
            }
        }

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
        // (pid, rtt_ns, bytes_in_flight, cwnd, loss_rate, pacing_rate_bps,
        //  sent_count, lost_count)
        let mut raw: Vec<(usize, u128, usize, u64, f64, u64, u64, u64)> = Vec::new();

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
            if rtt_ns < min_rtt_ns {
                min_rtt_ns = rtt_ns;
            }
            if pacing_bps > max_pacing_bps {
                max_pacing_bps = pacing_bps;
            }
            raw.push((pid, rtt_ns, bytes_in_flight, cwnd, loss_rate, pacing_bps,
                      sent as u64, lost as u64));
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

        for &(pid, rtt_ns, bytes_in_flight, cwnd, loss_rate, pacing_bps, _sent, _lost) in &raw {
            self.ensure_arm(pid);

            // ── cwnd_pressure EMA smoothing ───────────────────────────
            // Raw `bytes_in_flight / cwnd` is bursty packet-by-packet;
            // smooth it per path so the reward term (1 - cwnd_p) does not
            // drive scheduler oscillation on cwnd noise.
            let raw_cwnd_p = if cwnd > 0 {
                (bytes_in_flight as f64 / cwnd as f64).min(1.0)
            } else {
                1.0
            };
            if pid >= self.ema_cwnd_pressure.len() {
                self.ema_cwnd_pressure.resize(pid + 1, None);
            }
            let smoothed_cwnd_p = match self.ema_cwnd_pressure[pid] {
                Some(prev) => CWND_P_EMA_ALPHA * raw_cwnd_p
                    + (1.0 - CWND_P_EMA_ALPHA) * prev,
                None => raw_cwnd_p,
            };
            self.ema_cwnd_pressure[pid] = Some(smoothed_cwnd_p);

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
                if pid >= self.forget_counts.len() {
                    self.forget_counts.resize(pid + 1, 0);
                }
                self.forget_counts[pid] += 1;
            }

            let x = Self::make_context(
                rtt_ns,
                min_rtt_ns,
                smoothed_cwnd_p,
                // Use the windowed EMA loss rate when available; fall back to the
                // cumulative rate for the first second before any snapshot exists.
                self.ema_loss_rate.get(pid).copied().unwrap_or(loss_rate),
                pacing_bps,
                max_pacing_bps,
            );
            let arm =
                self.arms[pid].as_ref().unwrap();

            let n = self
                .ack_counts
                .get(pid)
                .copied()
                .unwrap_or(0);

            let alpha = (self.alpha_init
                / ((n + 1) as f64).sqrt())
            .max(self.alpha_floor);

            let (est, mut bonus) =
                Self::ucb_parts(arm, x, alpha);

            // ----------------------------------------------------
            // Fairness exploration boost
            // ----------------------------------------------------
            // If a path has nearly vanished from traffic share,
            // artificially boost its exploration term so it
            // periodically gets re-sampled.
            //
            // This avoids permanent starvation after temporary
            // congestion or RTT spikes.
            // ----------------------------------------------------
            let current_share_pct = if self.window_total > 0 {
                self.window_counts
                    .get(pid)
                    .copied()
                    .unwrap_or(0) as f64
                    * 100.0
                    / self.window_total as f64
            } else {
                100.0
            };

            if current_share_pct
                < FAIRNESS_MIN_SHARE_PCT
            {
                bonus *= FAIRNESS_BOOST;
            }

            let ucb = est + bonus;

            score_rows.push((
                pid,
                (rtt_ns / 1_000) as u64,
                x[1],
                est,
                bonus,
                ucb,
                x,
            ));
            if pid >= self.last_contexts.len() {
                self.last_contexts
                    .resize(pid + 1, None);
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
        for &(pid, _, _, _, _, _, _, _) in &raw {
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
            //
            // LinUCB internals (selection-time):
            //   p{pid}.pct           — % selections in this 1-second window
            //   p{pid}.rtt_ms        — EMA RTT in ms
            //   p{pid}.reward        — reward estimate θᵀx (exploitation term)
            //   p{pid}.bonus         — exploration bonus α√(xᵀA⁻¹x)
            //   p{pid}.ucb_total     — reward + bonus (the score being maximised)
            //   p{pid}.explore_ratio — bonus / |ucb_total| (0=exploit, 1=explore)
            //   p{pid}.a_trace       — sum of A's diagonal (effective sample count;
            //                          drops at forgetting events)
            //   p{pid}.forget_count  — cumulative count of forgetting events
            //                          (idle + RTT-jump); step-function counter
            //   p{pid}.n             — cumulative ACK count for this arm
            //   p{pid}.alpha         — current α value for this arm
            //   p{pid}.x_*           — context features (rtt_norm, cwnd_p, loss_rate)
            //                          x_bias and x_bw_norm are always 1.0; omitted.
            //   p{pid}.th_*          — 5 learned LinUCB weights including th_bias
            //
            // Network state (throughput, congestion, loss):
            //   p{pid}.pacing_mbps        — pacing rate in Mbps (CC-allowed)
            //   p{pid}.delivered_mbps     — approx delivered Mbps this window
            //                                (sent_per_sec × 1380 B × 8 / 1e6)
            //   p{pid}.sent_per_sec       — packets sent in this 1-second window
            //   p{pid}.traffic_share_pct  — % of total packets sent on this path
            //                                in this window (actual traffic split,
            //                                vs `pct` which is the *selection* split)
            //   p{pid}.bif_kb             — bytes in flight (KiB)
            //   p{pid}.cwnd_kb            — congestion window (KiB)
            //   p{pid}.sent               — cumulative packets sent
            //   p{pid}.lost               — cumulative packets lost
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

                // ── Per-window traffic deltas ──────────────────────────
                // Compute packets sent in this 1-second window per path,
                // and the global total, so each path's traffic share can be
                // reported.  On a path's first appearance we treat the
                // "previous" snapshot as the current sent count, yielding a
                // delta of 0 (avoids reporting cumulative as instantaneous).
                const AVG_PKT_SIZE_BYTES: f64 = 1380.0;
                let mut path_delta_sent: Vec<u64> = Vec::new();
                let mut total_delta_sent: u64 = 0;
                for &(pid, _, _, _, _, _, sent_p, _) in &raw {
                    let prev = if pid < self.last_window_sent.len() {
                        self.last_window_sent[pid]
                    } else {
                        sent_p
                    };
                    let delta = sent_p.saturating_sub(prev);
                    if pid >= path_delta_sent.len() {
                        path_delta_sent.resize(pid + 1, 0);
                    }
                    path_delta_sent[pid] = delta;
                    total_delta_sent = total_delta_sent.saturating_add(delta);
                }

                let mut jline = format!(
                    "{{\"_step\":{step},\"t\":{elapsed_s},\
                     \"cfg.alpha_init\":{:.4},\"cfg.alpha_floor\":{:.4},\
                     \"dominant_path\":{dominant_path},\"path_delta\":{path_delta:.2}",
                    self.alpha_init, self.alpha_floor,
                );
                for &(pid, rtt_us, _cp, reward_est, explore_bonus, _ucb, x) in &score_rows {
                    let cnt = self.window_counts.get(pid).copied().unwrap_or(0);
                    let pct = cnt as f64 * 100.0 / total as f64;
                    let n = self.ack_counts.get(pid).copied().unwrap_or(0);
                    let theta = if let Some(Some(arm)) = self.arms.get(pid) {
                        mat_vec(mat_inv(arm.a), arm.b)
                    } else {
                        [0.0_f64; D]
                    };
                    let alpha_logged = (self.alpha_init / ((n + 1) as f64).sqrt()).max(self.alpha_floor);

                    // Look up the raw network stats for this pid (throughput/congestion/loss).
                    let (bif_b, cwnd_b, pacing_bps_p, sent_p, lost_p) = raw
                        .iter()
                        .find(|r| r.0 == pid)
                        .map(|r| (r.2 as u64, r.3, r.5, r.6, r.7))
                        .unwrap_or((0, 0, 0, 0, 0));
                    let pacing_mbps = pacing_bps_p as f64 * 8.0 / 1_000_000.0;
                    let bif_kb = bif_b as f64 / 1024.0;
                    let cwnd_kb = cwnd_b as f64 / 1024.0;
                    let rtt_ms = rtt_us as f64 / 1000.0;

                    // Composite LinUCB diagnostics.
                    //   ucb_total     — the full score: reward_est + explore_bonus
                    //   explore_ratio — fraction of UCB coming from exploration
                    //                   (1.0 = pure exploration, 0.0 = pure exploit)
                    //   a_trace       — sum of A's diagonal; effective sample count.
                    //                   Drops sharply at forgetting events, grows with ACKs.
                    let ucb_total = reward_est + explore_bonus;
                    let explore_ratio = if ucb_total.abs() > 1e-12 {
                        explore_bonus / ucb_total.abs()
                    } else {
                        1.0
                    };
                    let a_trace = if let Some(Some(arm)) = self.arms.get(pid) {
                        (0..D).map(|i| arm.a[i][i]).sum::<f64>()
                    } else {
                        0.0
                    };
                    let forget_count = self.forget_counts.get(pid).copied().unwrap_or(0);

                    // Per-window traffic.  `sent_per_sec` is packets/sec for
                    // this 1-second window.  `delivered_mbps` approximates the
                    // delivered Mbps assuming an average QUIC packet of 1380 B.
                    // `traffic_share_pct` is the fraction of *bytes* (here:
                    // packets) carried by this path in this window.
                    let sent_per_sec = path_delta_sent.get(pid).copied().unwrap_or(0);
                    let delivered_mbps =
                        sent_per_sec as f64 * AVG_PKT_SIZE_BYTES * 8.0 / 1_000_000.0;
                    let traffic_share_pct = if total_delta_sent > 0 {
                        sent_per_sec as f64 * 100.0 / total_delta_sent as f64
                    } else {
                        0.0
                    };

                    jline.push_str(&format!(
                        ",\"p{pid}.pct\":{pct:.2},\
                         \"p{pid}.rtt_ms\":{rtt_ms:.3},\
                         \"p{pid}.reward\":{reward_est:.4},\"p{pid}.bonus\":{explore_bonus:.4},\
                         \"p{pid}.ucb_total\":{ucb_total:.4},\
                         \"p{pid}.explore_ratio\":{explore_ratio:.4},\
                         \"p{pid}.a_trace\":{a_trace:.3},\
                         \"p{pid}.forget_count\":{forget_count},\
                         \"p{pid}.n\":{n},\"p{pid}.alpha\":{alpha_logged:.4},\
                         \"p{pid}.pacing_mbps\":{pacing_mbps:.3},\
                         \"p{pid}.delivered_mbps\":{delivered_mbps:.3},\
                         \"p{pid}.sent_per_sec\":{sent_per_sec},\
                         \"p{pid}.traffic_share_pct\":{traffic_share_pct:.2},\
                         \"p{pid}.bif_kb\":{bif_kb:.2},\"p{pid}.cwnd_kb\":{cwnd_kb:.2},\
                         \"p{pid}.sent\":{sent_p},\"p{pid}.lost\":{lost_p}",
                    ));
                    // x_bias is always 1.0 (constant intercept); skip to reduce clutter.
                    for (i, xi) in x.iter().enumerate() {
                        if feat_names[i] == "bias" { continue; }
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
            // Snapshot current `sent_count` and `lost_count` so the next window's
            // sent_per_sec / delivered_mbps / traffic_share_pct and windowed loss
            // rate can be computed.
            for &(pid, _, _, _, _, _, sent_p, lost_p) in &raw {
                // Read previous snapshots BEFORE overwriting them.
                let prev_sent = if pid < self.last_window_sent.len() { self.last_window_sent[pid] } else { sent_p };
                let prev_lost = if pid < self.last_window_lost.len() { self.last_window_lost[pid] } else { lost_p };

                // Windowed loss rate: fraction of packets lost in this 1-second window,
                // EMA-smoothed (α=0.5) to dampen statistical noise from short bursts.
                let d_sent = sent_p.saturating_sub(prev_sent);
                let d_lost = lost_p.saturating_sub(prev_lost);
                let window_loss = if d_sent > 0 { (d_lost as f64 / d_sent as f64).min(1.0) } else { 0.0 };
                if pid >= self.ema_loss_rate.len() {
                    self.ema_loss_rate.resize(pid + 1, 0.0);
                }
                self.ema_loss_rate[pid] = 0.5 * window_loss + 0.5 * self.ema_loss_rate[pid];

                // Now update the snapshots for next window.
                if pid >= self.last_window_sent.len() {
                    self.last_window_sent.resize(pid + 1, 0);
                }
                self.last_window_sent[pid] = sent_p;
                if pid >= self.last_window_lost.len() {
                    self.last_window_lost.resize(pid + 1, 0);
                }
                self.last_window_lost[pid] = lost_p;
            }
        }

        Ok(best_pid)
    }

    /// Update the model for the path that received an ACK.
    fn on_ack(&mut self, _now: Instant, path_id: usize, paths: &mut PathMap) {
        let path = match paths.get_mut(path_id) {
            Ok(p) => p,
            Err(_) => return,
        };

        let rtt_ns = path
            .recovery
            .rtt
            .latest_rtt()
            .as_nanos()
            .max(1);

        if path_id >= self.ema_rtt_ns.len() {
            self.ema_rtt_ns
                .resize(path_id + 1, rtt_ns as f64);

            self.ack_counts.resize(path_id + 1, 0);
        }

        let old_ema = self.ema_rtt_ns[path_id];

        self.ack_counts[path_id] += 1;

        let ema_alpha =
            if self.ack_counts[path_id] <= 8 {
                0.5_f64
            } else {
                0.2_f64
            };

        self.ema_rtt_ns[path_id] =
            ema_alpha * rtt_ns as f64
                + (1.0 - ema_alpha) * old_ema;

        let ema_rtt_ns =
            self.ema_rtt_ns[path_id] as u128;

        self.ensure_arm(path_id);

        if self.ack_counts[path_id] > 8 {
            let rtt_change =
                (self.ema_rtt_ns[path_id] - old_ema)
                    / old_ema;

            if rtt_change.abs() > 0.30 {
                let arm =
                    self.arms[path_id].as_mut().unwrap();

                let lambda = 0.85_f64;

                for i in 0..D {
                    for j in 0..D {
                        arm.a[i][j] *= lambda;
                    }

                    arm.b[i] *= lambda;

                    arm.a[i][i] += 1.0 - lambda;
                }

                if path_id >= self.forget_counts.len() {
                    self.forget_counts
                        .resize(path_id + 1, 0);
                }

                self.forget_counts[path_id] += 1;
            }
        }

        let x = self
            .last_contexts
            .get(path_id)
            .and_then(|ctx| *ctx)
            .unwrap_or_else(|| {
                let cwnd =
                    path.recovery.congestion.congestion_window();

                let bif =
                    path.recovery.bytes_in_flight;

                let raw_cp = if cwnd > 0 {
                    (bif as f64 / cwnd as f64)
                        .min(1.0)
                } else {
                    1.0
                };

                let cp = self
                    .ema_cwnd_pressure
                    .get(path_id)
                    .and_then(|c| *c)
                    .unwrap_or(raw_cp);

                Self::make_context(
                    ema_rtt_ns,
                    self.last_min_rtt_ns,
                    cp,
                    {
                        let sent =
                            path.recovery.stats.sent_count;

                        let lost =
                            path.recovery.stats.lost_count;

                        if sent > 0 {
                            lost as f64 / sent as f64
                        } else {
                            0.0
                        }
                    },
                    path.recovery
                        .congestion
                        .pacing_rate()
                        .unwrap_or(0),
                    0,
                )
            });

        // Shifted RTT reward: exp(-(rtt_norm - 1)).
        // The best path (rtt_norm=1.0) now earns reward 1.0 instead of
        // exp(-1)≈0.37.  This ~3× stronger signal makes θ diverge faster
        // between paths, so even a small change in alpha_linucb produces a
        // visible shift in the exploitation/exploration balance.
        let reward = (-(x[0] - 1.0)).exp();

        let arm =
            self.arms[path_id].as_mut().unwrap();

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

fn mat_inv(
    m: [[f64; D]; D],
) -> [[f64; D]; D] {
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

fn mat_vec(
    m: [[f64; D]; D],
    v: [f64; D],
) -> [f64; D] {
    let mut r = [0.0_f64; D];
    for i in 0..D {
        for j in 0..D {
            r[i] += m[i][j] * v[j];
        }
    }
    r
}

fn dot(a: [f64; D], b: [f64; D]) -> f64 {
    let mut s = 0.0_f64;
    for i in 0..D {
        s += a[i] * b[i];
    }
    s
}

fn quadratic(
    m: [[f64; D]; D],
    x: [f64; D],
) -> f64 {
    dot(x, mat_vec(m, x))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multipath_scheduler::tests::*;

    #[test]
    fn linucb_single_path() -> Result<()> {
        let mut t = MultipathTester::new()?;

        let mut s =
            LinUCBScheduler::new(&Default::default());

        assert_eq!(
            s.on_select(
                &mut t.paths,
                &mut t.spaces,
                &mut t.streams
            )?,
            0
        );

        Ok(())
    }

    #[test]
    fn linucb_multi_path_selects_valid() -> Result<()> {
        let mut t = MultipathTester::new()?;

        t.add_path(
            "127.0.0.1:443",
            "127.0.0.2:8443",
            50,
        )?;

        t.add_path(
            "127.0.0.1:443",
            "127.0.0.3:8443",
            150,
        )?;

        let mut s =
            LinUCBScheduler::new(&Default::default());

        let pid = s.on_select(
            &mut t.paths,
            &mut t.spaces,
            &mut t.streams,
        )?;

        assert!(pid <= 2);

        Ok(())
    }

    #[test]
    fn mat_inv_identity() {
        let mut id = [[0.0_f64; D]; D];

        for i in 0..D {
            id[i][i] = 1.0;
        }

        let inv = mat_inv(id);

        for i in 0..D {
            for j in 0..D {
                let expected =
                    if i == j { 1.0 } else { 0.0 };

                assert!(
                    (inv[i][j] - expected).abs()
                        < 1e-10
                );
            }
        }
    }
}