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
///   x[0] = min_ema_rtt / ema_rtt          (latency score; 1.0 for best path)
///   x[1] = 1 - bytes_in_flight / cwnd     (cwnd headroom score in [0, 1])
///   x[2] = 1 - lost_pkts / sent_pkts      (reliability score in [0, 1])
///   x[3] = pacing / max_pacing            (relative pacing score in [0, 1])
///   x[4] = 1.0                            (bias / intercept term)
const D: usize = 5;

/// Pending selections that do not receive feedback by this timeout are treated
/// as failed decisions and trained once with zero reward.
const STALE_DECISION_TIMEOUT: Duration = Duration::from_secs(1);

/// RTT change threshold for triggering RTT-jump reset.
/// If RTT changes by more than 30%, we reset the model to adapt to new conditions.
const RTT_JUMP_THRESHOLD: f64 = 0.30;

/// Idle forgetting threshold - if a path is idle for this duration, apply forgetting.
const IDLE_FORGET_THRESHOLD: Duration = Duration::from_secs(10);

/// Forgetting factor for idle paths - decays the A matrix toward identity.
const IDLE_FORGET_FACTOR: f64 = 0.95;

/// Per-arm (per-path) state for the LinUCB algorithm.
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
/// where the reward rₚ is a bounded score derived from latency, reliability,
/// and delivery feedback.
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

    /// Unix timestamp (seconds) at scheduler creation, used to compute per-step
    /// `_timestamp` required by wandb to render time-series charts.
    start_unix_secs: u64,
    /// Cumulative sent-bytes snapshot from the previous 1-second window.
    /// Used to compute per-second actual throughput (delta bytes × 8 / 1e6 Mbps).
    prev_sent_bytes: Vec<u64>,
    /// Pending selected context for each path.
    /// Stored so on_ack can update the model with the exact features that won
    /// selection, and so stalled paths can receive one zero-reward update.
    pending_context: Vec<Option<[f64; D]>>,
    /// Time each pending context was selected.
    pending_since: Vec<Option<Instant>>,

    last_window_sent: Vec<u64>,
    last_max_pacing_bps: u64,
    /// Last EMA RTT for each path, used for RTT-jump detection.
    last_ema_rtt_ns: Vec<f64>,
    /// Last selection time for each path, used for idle forgetting.
    last_selection_time: Vec<Option<Instant>>,
    /// Last observed `acked_count` per path, used for delivery delta reward.
    prev_acked_count: Vec<u64>,
    /// Last observed `sent_count` per path, paired with `prev_acked_count`.
    prev_sent_count: Vec<u64>,
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
            pending_context: Vec::new(),
            pending_since: Vec::new(),

            last_window_sent: Vec::new(),
            last_max_pacing_bps: 0,
            last_ema_rtt_ns: Vec::new(),
            last_selection_time: Vec::new(),
            prev_acked_count: Vec::new(),
            prev_sent_count: Vec::new(),
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
        bytes_in_flight: usize,
        cwnd: u64,
        loss_rate: f64,
        pacing_rate_bytes_per_sec: u64,
        max_pacing_bytes_per_sec: u64,
    ) -> [f64; D] {
        let latency_score = if rtt_ns > 0 && min_rtt_ns > 0 {
            (min_rtt_ns as f64 / rtt_ns as f64).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let cwnd_headroom_score = if cwnd > 0 {
            1.0 - (bytes_in_flight as f64 / cwnd as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let reliability_score = 1.0 - loss_rate.clamp(0.0, 1.0);
        let relative_pacing_score =
            if pacing_rate_bytes_per_sec > 0 && max_pacing_bytes_per_sec > 0 {
                (pacing_rate_bytes_per_sec as f64 / max_pacing_bytes_per_sec as f64)
                    .clamp(0.0, 1.0)
            } else {
                0.0
            };
        [
            latency_score,
            cwnd_headroom_score,
            reliability_score,
            relative_pacing_score,
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

    /// Dynamic exploration coefficient based on selections, not ACKs.
    /// Decays with selection count from `alpha_init` and is clamped above by
    /// `alpha_floor` so the exploration bonus never collapses entirely.
    fn selection_alpha(&self, selection_count: u64) -> f64 {
        let decayed = self.alpha_init / ((selection_count + 1) as f64).sqrt();
        decayed.max(self.alpha_floor)
    }

    /// Reward used for ACK feedback.
    ///
    /// r = delivery × exp(-(rtt_norm - 1)) × sqrt(1 - cwnd_pressure)
    ///
    /// where:
    ///   delivery      = Δacked / max(Δsent, 1)
    ///   rtt_norm      = clamp(ema_rtt / min_rtt, 1, 4)
    ///   cwnd_pressure = clamp(bytes_in_flight / cwnd, 0, 1)
    fn observed_reward(
        delta_acked: u64,
        delta_sent: u64,
        ema_rtt_ns: f64,
        min_rtt_ns: u128,
        bif: usize,
        cwnd: u64,
    ) -> f64 {
        let delivery = if delta_sent > 0 {
            (delta_acked as f64 / delta_sent as f64).clamp(0.0, 1.0)
        } else {
            0.5
        };

        let rtt_norm = (ema_rtt_ns / min_rtt_ns.max(1) as f64).clamp(1.0, 4.0);
        let cwnd_pressure = if cwnd > 0 {
            (bif as f64 / cwnd as f64).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let latency_term = (-(rtt_norm - 1.0)).exp();
        let congestion_term = (1.0 - cwnd_pressure).max(0.0).sqrt();

        (delivery * latency_term * congestion_term).clamp(0.0, 1.0)
    }

    fn ensure_pending_state(&mut self, path_id: usize) {
        if path_id >= self.pending_context.len() {
            self.pending_context.resize(path_id + 1, None);
        }
        if path_id >= self.pending_since.len() {
            self.pending_since.resize(path_id + 1, None);
        }
    }

    fn mark_pending_context(&mut self, path_id: usize, x: [f64; D], now: Instant) {
        self.ensure_pending_state(path_id);
        // Always overwrite with the latest selection context so that the next
        // ACK trains on the most recent decision basis. Previously, repeated
        // selections of the same path before an ACK arrived would drop the
        // newer contexts, causing many decisions to never be trained.
        self.pending_context[path_id] = Some(x);
        // Only refresh the pending timestamp on the first selection; otherwise
        // a busy path could indefinitely defer the stale-decision timeout.
        if self.pending_since[path_id].is_none() {
            self.pending_since[path_id] = Some(now);
        }
    }

    fn take_pending_context(&mut self, path_id: usize) -> Option<[f64; D]> {
        if path_id >= self.pending_context.len() {
            return None;
        }
        if path_id < self.pending_since.len() {
            self.pending_since[path_id] = None;
        }
        self.pending_context[path_id].take()
    }

    fn penalize_stale_pending(&mut self, now: Instant, skip_path_id: Option<usize>) {
        for path_id in 0..self.pending_context.len() {
            if skip_path_id == Some(path_id) {
                continue;
            }

            let is_stale = self
                .pending_since
                .get(path_id)
                .and_then(|t| *t)
                .map(|t| {
                    now.checked_duration_since(t).unwrap_or_default()
                        >= STALE_DECISION_TIMEOUT
                })
                .unwrap_or(false);

            if !is_stale {
                continue;
            }

            if let Some(x) = self.pending_context[path_id].take() {
                self.ensure_arm(path_id);
                let arm = self.arms[path_id].as_mut().unwrap();
                Self::update_arm(arm, x, 0.0);
            }

            if path_id < self.pending_since.len() {
                self.pending_since[path_id] = None;
            }
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
        let now = Instant::now();
        self.penalize_stale_pending(now, None);

        // Collect per-path stats via iter_mut (can_send() takes &mut self).
        // Values are extracted before mutating self.arms to satisfy the borrow checker.
        let mut min_rtt_ns: u128 = u128::MAX;
        let mut max_pacing_bytes_per_sec: u64 = 0;
        // (pid, rtt_ns, bytes_in_flight, cwnd, loss_rate, pacing_rate_bytes_per_sec,
        //  sent_bytes_total, lost_count_total, sent_count_total)
        let mut raw: Vec<(usize, u128, usize, u64, f64, u64, u64, u64, u64)> = Vec::new();

        for (pid, path) in paths.iter_mut() {
            if !path.active() || !path.recovery.can_send() {
                continue;
            }
            let rtt_ns = if self.ack_counts.get(pid).copied().unwrap_or(0) > 0
                && pid < self.ema_rtt_ns.len()
            {
                self.ema_rtt_ns[pid] as u128
            } else {
                path.recovery.rtt.smoothed_rtt().as_nanos()
            };
            let bytes_in_flight = path.recovery.bytes_in_flight;
            let cwnd = path.recovery.congestion.congestion_window();
            let sent = path.recovery.stats.sent_count;
            let lost = path.recovery.stats.lost_count;
            let loss_rate = if sent > 0 { lost as f64 / sent as f64 } else { 0.0 };
            let pacing_rate_bytes_per_sec = path.recovery.congestion.pacing_rate().unwrap_or(0);
            let sent_bytes_total = path.recovery.stats.sent_bytes;
            if rtt_ns < min_rtt_ns {
                min_rtt_ns = rtt_ns;
            }
            if pacing_rate_bytes_per_sec > max_pacing_bytes_per_sec {
                max_pacing_bytes_per_sec = pacing_rate_bytes_per_sec;
            }
            raw.push((
                pid,
                rtt_ns,
                bytes_in_flight,
                cwnd,
                loss_rate,
                pacing_rate_bytes_per_sec,
                sent_bytes_total,
                lost,
                sent,
            ));
        }

        if raw.is_empty() {
            return Err(Error::Done);
        }

        let min_rtt_ns = min_rtt_ns.max(1);
        self.last_min_rtt_ns = min_rtt_ns;
        self.last_max_pacing_bps = max_pacing_bytes_per_sec;

        // ── Idle forgetting ─────────────────────────────────────────────────────
        // If a path has been idle for more than IDLE_FORGET_THRESHOLD, decay its
        // A matrix toward identity to re-trigger exploration on that path.
        for &(pid, ..) in &raw {
            if pid >= self.last_selection_time.len() {
                self.last_selection_time.resize(pid + 1, None);
            }
            if let Some(last_time) = self.last_selection_time[pid] {
                if now.duration_since(last_time) > IDLE_FORGET_THRESHOLD {
                    self.ensure_arm(pid);
                    let arm = self.arms[pid].as_mut().unwrap();
                    // Decay A matrix toward identity: A ← λA + (1-λ)I
                    for i in 0..D {
                        for j in 0..D {
                            arm.a[i][j] = IDLE_FORGET_FACTOR * arm.a[i][j];
                        }
                        arm.a[i][i] += 1.0 - IDLE_FORGET_FACTOR;
                    }
                }
            }
        }

        // Pick the arm with the highest UCB score and collect per-path scores
        // for logging.
        let mut best_pid = raw[0].0;
        let mut best_score = f64::NEG_INFINITY;
        // (pid, rtt_us, cwnd_headroom_score, reward_est, explore_bonus, ucb, context_x)
        let mut score_rows: Vec<(usize, u64, f64, f64, f64, f64, [f64; D])> = Vec::new();

        for &(
            pid,
            rtt_ns,
            bytes_in_flight,
            cwnd,
            loss_rate,
            pacing_rate_bytes_per_sec,
            ..
        ) in &raw
        {
            let x = Self::make_context(
                rtt_ns,
                min_rtt_ns,
                bytes_in_flight,
                cwnd,
                loss_rate,
                pacing_rate_bytes_per_sec,
                max_pacing_bytes_per_sec,
            );
            self.ensure_arm(pid);
            let arm = self.arms[pid].as_ref().unwrap();
            // Dynamic alpha: 1/sqrt(n+1). Starts at 1.0 (heavy exploration),
            // decays as the arm is selected, so slow-ACKing bad paths do not
            // keep high exploration forever.
            let n = self.total_counts.get(pid).copied().unwrap_or(0);
            let alpha = self.selection_alpha(n);
            let (est, bonus) = Self::ucb_parts(arm, x, alpha);
            let ucb = est + bonus;
            let cwnd_headroom_score = x[1];
            score_rows.push((
                pid,
                (rtt_ns / 1_000) as u64,
                cwnd_headroom_score,
                est,
                bonus,
                ucb,
                x,
            ));
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

        if let Some((_, _, _, _, _, _, x)) = score_rows.iter().find(|row| row.0 == best_pid) {
            self.mark_pending_context(best_pid, *x, now);
        }

        // Update last selection time for the selected path (used for idle forgetting)
        if best_pid >= self.last_selection_time.len() {
            self.last_selection_time.resize(best_pid + 1, None);
        }
        self.last_selection_time[best_pid] = Some(now);

        // Cache per-path addresses for the final summary.
        // Prefer local_addr, but fall back to remote_addr when local is
        // unspecified (server bound to 0.0.0.0 / ::).
        for &(pid, ..) in &raw {
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
            //   p{pid}.x_*           — context features (rtt_norm, cwnd_p, loss_rate, bw_norm)
            //                          x_bias is omitted (always 1.0).
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
                let timestamp = self.start_unix_secs + elapsed_s;
                // Feature / weight names in context-vector order:
                //   x0=latency_score  x1=cwnd_headroom  x2=reliability_score
                //   x3=relative_pacing_score  x4=bias
                let feat_names = [
                    "latency_score",
                    "cwnd_headroom",
                    "reliability_score",
                    "relative_pacing_score",
                    "bias",
                ];
                let mut jline = format!(
                    "{{\"_step\":{step},\"_timestamp\":{timestamp},\"t\":{elapsed_s}"
                );

                // ── Per-window traffic deltas ──────────────────────────
                // Compute bytes sent in this 1-second window per path,
                // and the global total, so each path's traffic share can be
                // reported.  On a path's first appearance we treat the
                // "previous" snapshot as the current sent count, yielding a
                // delta of 0 (avoids reporting cumulative as instantaneous).
                let mut path_delta_sent: Vec<u64> = Vec::new();
                let mut total_delta_sent: u64 = 0;
                for &(pid, _, _, _, _, _, sent_bytes_p, _, _) in &raw {
                    let prev = if pid < self.prev_sent_bytes.len() {
                        self.prev_sent_bytes[pid]
                    } else {
                        sent_bytes_p
                    };
                    let delta = sent_bytes_p.saturating_sub(prev);
                    if pid >= path_delta_sent.len() {
                        path_delta_sent.resize(pid + 1, 0);
                    }
                    path_delta_sent[pid] = delta;
                    total_delta_sent = total_delta_sent.saturating_add(delta);
                }

                for &(pid, _rtt_us, _cp, reward_est, explore_bonus, _ucb, x) in &score_rows {
                    let cnt = self.window_counts.get(pid).copied().unwrap_or(0);
                    let pct = cnt as f64 * 100.0 / total as f64;
                    let n = self.total_counts.get(pid).copied().unwrap_or(0);
                    let alpha = self.selection_alpha(n);
                    let theta = if let Some(Some(arm)) = self.arms.get(pid) {
                        mat_vec(mat_inv(arm.a), arm.b)
                    } else {
                        [0.0_f64; D]
                    };

                    // Look up the raw network stats for this pid (throughput/congestion/loss).
                    let (
                        bytes_in_flight,
                        cwnd,
                        pacing_rate_bytes_per_sec,
                        sent_bytes_total,
                        lost_total,
                        sent_total,
                    ) =
                        raw.iter().find(|r| r.0 == pid)
                            .map(|r| (r.2, r.3, r.5, r.6, r.7, r.8))
                            .unwrap_or((0, 0, 0, 0, 0, 0));
                    // Per-second actual throughput: delta sent_bytes since last window.
                    let delta_bytes = path_delta_sent.get(pid).copied().unwrap_or(0);
                    let throughput_mbps = delta_bytes as f64 * 8.0 / 1_000_000.0;
                    let pacing_mbps = pacing_rate_bytes_per_sec as f64 * 8.0 / 1_000_000.0;
                    let cwnd_kb = cwnd as f64 / 1024.0;
                    let inflight_kb = bytes_in_flight as f64 / 1024.0;
                    // Report baseline path RTT (close to ping) using min_rtt,
                    // and keep latest RTT as a separate metric for queueing spikes.
                    let (rtt_ms, latest_rtt_ms) = if let Ok(path) = paths.get(pid) {
                        (
                            path.recovery.rtt.min_rtt().as_micros() as f64 / 1000.0,
                            path.recovery.rtt.latest_rtt().as_micros() as f64 / 1000.0,
                        )
                    } else {
                        (0.0, 0.0)
                    };
                    // Throughput-branch compatible aliases.
                    let rtt_norm = if x[0] > 1e-12 { (1.0 / x[0]).clamp(1.0, 20.0) } else { 20.0 };
                    let cwnd_p = (1.0 - x[1]).clamp(0.0, 1.0);
                    let loss_rate = (1.0 - x[2]).clamp(0.0, 1.0);
                    let bw_norm = x[3].clamp(0.0, 1.0);
                    let ucb_total = reward_est + explore_bonus;
                    // Normalize by current alpha so explore_ratio is comparable
                    // across runs with different alpha_init values.
                    // Equals 1.0 when bonus == alpha (full exploration) and
                    // decays toward alpha_floor/alpha at steady state.
                    let explore_ratio = (explore_bonus / (alpha + 1e-12)).clamp(0.0, 1.0);
                    let a_trace = if let Some(Some(arm)) = self.arms.get(pid) {
                        (0..D).map(|i| arm.a[i][i]).sum::<f64>()
                    } else {
                        0.0
                    };
                    // Bytes-based traffic share: actual bytes sent on this path
                    // in this window as a percentage of total bytes across all paths.
                    let traffic_share_pct = if total_delta_sent > 0 {
                        delta_bytes as f64 * 100.0 / total_delta_sent as f64
                    } else {
                        0.0
                    };
                    // traffic / latency / throughput
                    jline.push_str(&format!(
                        ",\"traffic/path{pid}_pct\":{pct:.2}\
                         ,\"latency/path{pid}_rtt_ms\":{rtt_ms:.3}\
                        ,\"latency/path{pid}_latest_rtt_ms\":{latest_rtt_ms:.3}\
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
                    let reward_est_log = reward_est.max(0.0);
                    jline.push_str(&format!(
                        ",\"linucb/path{pid}_reward\":{reward_est_log:.4}\
                         ,\"linucb/path{pid}_explore_bonus\":{explore_bonus:.4}\
                         ,\"linucb/path{pid}_samples\":{n}",
                    ));
                    // throughput-branch schema aliases (p{pid}.*)
                    let rtt_us = (rtt_ms * 1000.0) as u64;
                    jline.push_str(&format!(
                        ",\"p{pid}.pct\":{pct:.2}\
                         ,\"p{pid}.rtt_us\":{rtt_us}\
                         ,\"p{pid}.rtt_ms\":{rtt_ms:.3}\
                         ,\"p{pid}.reward\":{reward_est_log:.4}\
                         ,\"p{pid}.bonus\":{explore_bonus:.4}\
                         ,\"p{pid}.ucb_total\":{ucb_total:.4}\
                         ,\"p{pid}.explore_ratio\":{explore_ratio:.4}\
                         ,\"p{pid}.a_trace\":{a_trace:.3}\
                         ,\"p{pid}.forget_count\":0\
                         ,\"p{pid}.n\":{n}\
                         ,\"p{pid}.alpha\":{alpha:.4}\
                         ,\"p{pid}.pacing_mbps\":{pacing_mbps:.3}\
                         ,\"p{pid}.delivered_mbps\":{throughput_mbps:.3}\
                         ,\"p{pid}.sent_per_sec\":0\
                         ,\"p{pid}.traffic_share_pct\":{traffic_share_pct:.2}\
                         ,\"p{pid}.bif_kb\":{inflight_kb:.2}\
                         ,\"p{pid}.cwnd_kb\":{cwnd_kb:.2}\
                         ,\"p{pid}.sent\":{sent_total}\
                         ,\"p{pid}.lost\":{lost_total}\
                         ,\"p{pid}.x_rtt_norm\":{rtt_norm:.4}\
                         ,\"p{pid}.x_cwnd_p\":{cwnd_p:.4}\
                         ,\"p{pid}.x_loss_rate\":{loss_rate:.4}\
                         ,\"p{pid}.x_bw_norm\":{bw_norm:.4}\
                         ,\"p{pid}.x_bias\":1.0000\
                         ,\"p{pid}.th_rtt_norm\":{:.4}\
                         ,\"p{pid}.th_cwnd_p\":{:.4}\
                         ,\"p{pid}.th_loss_rate\":{:.4}\
                         ,\"p{pid}.th_bw_norm\":{:.4}\
                         ,\"p{pid}.th_bias\":{:.4}",
                        theta[0], theta[1], theta[2], theta[3], theta[4],
                    ));
                    // context features (x vector)
                    for (i, xi) in x.iter().enumerate() {
                        jline.push_str(&format!(
                            ",\"features/path{pid}_{}\":{xi:.4}",
                            feat_names[i]
                        ));
                    }
                    // learned weights (theta)
                    for (i, ti) in theta.iter().enumerate() {
                        jline.push_str(&format!(
                            ",\"weights/path{pid}_{}\":{ti:.4}",
                            feat_names[i]
                        ));
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
            // Snapshot current `sent_bytes` so the next window's
            // throughput_mbps can be computed.
            for &(pid, _, _, _, _, _, sent_bytes_p, _, _) in &raw {
                if pid >= self.prev_sent_bytes.len() {
                    self.prev_sent_bytes.resize(pid + 1, 0);
                }
                self.prev_sent_bytes[pid] = sent_bytes_p;
            }
        }

        Ok(best_pid)
    }

    /// Update the model for the path that received an ACK.
    fn on_ack(&mut self, now: Instant, path_id: usize, paths: &mut PathMap) {
        self.penalize_stale_pending(now, Some(path_id));

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
        if self.ack_counts[path_id] > 8 && old_ema > 0.0 {
            let rtt_change = (self.ema_rtt_ns[path_id] - old_ema) / old_ema;
            if rtt_change.abs() > RTT_JUMP_THRESHOLD {
                let arm = self.arms[path_id].as_mut().unwrap();
                for i in 0..D {
                    arm.a[i][i] += 10.0;
                }
            }
        }

        // Only train decisions that are still pending. A stale timeout consumes
        // the context and already applies the zero-reward update.
        let x = match self.take_pending_context(path_id) {
            Some(x) => x,
            None => return,
        };

        // Delta delivery over the most recent ACK interval.
        let acked_total = path.recovery.stats.acked_count;
        let sent_total = path.recovery.stats.sent_count;
        if path_id >= self.prev_acked_count.len() {
            self.prev_acked_count.resize(path_id + 1, 0);
        }
        if path_id >= self.prev_sent_count.len() {
            self.prev_sent_count.resize(path_id + 1, 0);
        }
        let delta_acked = acked_total.saturating_sub(self.prev_acked_count[path_id]);
        let delta_sent = sent_total.saturating_sub(self.prev_sent_count[path_id]);
        self.prev_acked_count[path_id] = acked_total;
        self.prev_sent_count[path_id] = sent_total;

        let reward = Self::observed_reward(
            delta_acked,
            delta_sent,
            self.ema_rtt_ns[path_id],
            self.last_min_rtt_ns,
            path.recovery.bytes_in_flight,
            path.recovery.congestion.congestion_window(),
        );

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
            // Final model estimate + exploration bonus using an ideal context:
            // latency=1.0, cwnd_headroom=1.0, reliability=1.0,
            // relative_pacing=1.0, bias=1.0.
            let (est_str, bonus_str) = if let Some(Some(arm)) = self.arms.get(pid) {
                let x = [1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64, 1.0_f64];
                let a_inv = mat_inv(arm.a);
                let theta = mat_vec(a_inv, arm.b);
                let est = dot(theta, x);
                let n = self.total_counts.get(pid).copied().unwrap_or(0);
                let alpha = self.selection_alpha(n);
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
    use std::time::{Duration, Instant};

    fn set_arm_theta(s: &mut LinUCBScheduler, path_id: usize, theta: [f64; D]) {
        s.ensure_arm(path_id);
        let arm = s.arms[path_id].as_mut().unwrap();
        arm.a = [[0.0_f64; D]; D];
        for i in 0..D {
            arm.a[i][i] = 1.0;
        }
        arm.b = theta;
    }

    #[test]
    fn linucb_single_path() -> Result<()> {
        let mut t = MultipathTester::new()?;
        let mut s = LinUCBScheduler::new(&Default::default());
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
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn linucb_features_are_higher_is_better() {
        let good = LinUCBScheduler::make_context(50, 50, 0, 12_000, 0.0, 4_000, 4_000);
        let bad = LinUCBScheduler::make_context(200, 50, 12_000, 12_000, 0.75, 0, 4_000);

        assert!(good[0] > bad[0], "lower RTT should score higher");
        assert!(good[1] > bad[1], "more cwnd headroom should score higher");
        assert!(good[2] > bad[2], "lower loss should score higher");
        assert!(good[3] > bad[3], "higher relative pacing should score higher");
        assert_eq!(bad[3], 0.0, "zero pacing rate should not look neutral");
        assert_eq!(good[4], 1.0);
    }

    #[test]
    fn linucb_ucb_winner_is_selected() -> Result<()> {
        let mut t = MultipathTester::new()?;
        t.add_path("127.0.0.1:443", "127.0.0.2:8443", 50)?;

        let mut s = LinUCBScheduler::new(&Default::default());
        s.total_counts.resize(2, 10_000);
        set_arm_theta(&mut s, 0, [-10.0, -10.0, -10.0, -10.0, -10.0]);
        set_arm_theta(&mut s, 1, [10.0, 10.0, 10.0, 10.0, 10.0]);

        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 1);
        Ok(())
    }

    #[test]
    fn linucb_bad_path_stops_receiving_after_low_rewards() -> Result<()> {
        let mut t = MultipathTester::new()?;
        t.add_path("127.0.0.1:443", "127.0.0.2:8443", 50)?;

        let mut s = LinUCBScheduler::new(&Default::default());
        s.ensure_arm(0);
        s.ensure_arm(1);

        let bad_x = [0.25, 0.2, 0.1, 0.2, 1.0];
        let good_x = [1.0, 1.0, 1.0, 1.0, 1.0];
        for _ in 0..12 {
            LinUCBScheduler::update_arm(s.arms[0].as_mut().unwrap(), bad_x, 0.0);
            LinUCBScheduler::update_arm(s.arms[1].as_mut().unwrap(), good_x, 1.0);
        }
        s.total_counts.resize(2, 100);

        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 1);
        Ok(())
    }

    #[test]
    fn linucb_exploration_decays_with_selections_not_acks() {
        let mut s = LinUCBScheduler::new(&Default::default());
        s.ack_counts.resize(1, 0);
        s.total_counts.resize(1, 99);

        let high_exploration = s.selection_alpha(0);
        let decayed_exploration = s.selection_alpha(s.total_counts[0]);

        assert_eq!(s.ack_counts[0], 0);
        assert!(decayed_exploration <= high_exploration);
        // Floor (alpha_floor = 0.15) kicks in well before n=99.
        assert!((decayed_exploration - s.alpha_floor).abs() < 1e-12);
    }

    #[test]
    fn linucb_stalled_path_receives_zero_reward() {
        let mut s = LinUCBScheduler::new(&Default::default());
        let now = Instant::now();
        let stale_since = now
            .checked_sub(STALE_DECISION_TIMEOUT + Duration::from_millis(1))
            .unwrap();

        s.mark_pending_context(0, [1.0, 1.0, 1.0, 1.0, 1.0], stale_since);
        s.penalize_stale_pending(now, None);

        let arm = s.arms[0].as_ref().unwrap();
        assert!(s.pending_context[0].is_none());
        assert!(s.pending_since[0].is_none());
        assert!(arm.a[0][0] > 1.0);
        assert_eq!(arm.b, [0.0_f64; D]);
    }

    #[test]
    fn linucb_on_ack_updates_pending_context_once() -> Result<()> {
        let mut t = MultipathTester::new()?;
        let mut s = LinUCBScheduler::new(&Default::default());
        let pid = s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?;

        {
            let path = t.paths.get_mut(pid)?;
            path.recovery.stats.sent_count = 10;
            path.recovery.stats.acked_count = 8;
        }

        let before = s.arms[pid].as_ref().unwrap().b;
        s.on_ack(Instant::now(), pid, &mut t.paths);
        let after = s.arms[pid].as_ref().unwrap().b;

        assert_ne!(before, after);
        assert!(s.pending_context[pid].is_none());
        Ok(())
    }

    #[test]
    fn linucb_slow_ack_path_cannot_extend_warmup_forever() -> Result<()> {
        let mut t = MultipathTester::new()?;
        t.add_path("127.0.0.1:443", "127.0.0.2:8443", 50)?;

        let mut s = LinUCBScheduler::new(&Default::default());
        s.ack_counts.resize(2, 0);
        s.ack_counts[0] = 0;
        s.ack_counts[1] = 64;
        s.ema_rtt_ns.resize(2, 50_000_000.0);
        s.ema_rtt_ns[0] = 200_000_000.0;
        s.ema_rtt_ns[1] = 50_000_000.0;
        s.total_counts.resize(2, 100);
        set_arm_theta(&mut s, 0, [-5.0, -5.0, -5.0, -5.0, -5.0]);
        set_arm_theta(&mut s, 1, [5.0, 5.0, 5.0, 5.0, 5.0]);

        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 1);
        Ok(())
    }
}