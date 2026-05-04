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
///   x[0] = srtt / min_srtt_across_paths  (normalised RTT; 1.0 for best path)
///   x[1] = bytes_in_flight / cwnd        (congestion window utilisation ∈ [0, 1])
///   x[2] = 1.0                            (bias / intercept term)
const D: usize = 3;

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
    /// Exploration coefficient α.  Larger values increase exploration.
    alpha: f64,
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
    /// Deficit credits for weighted round-robin path selection.
    /// Each scheduling round adds softmax(UCB) weight to every active path;
    /// the path with the most credits is chosen and loses one credit.
    credits: Vec<f64>,
}

impl LinUCBScheduler {
    pub fn new(_conf: &MultipathConfig) -> Self {
        let now = Instant::now();
        LinUCBScheduler {
            alpha: 0.5,
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
            credits: Vec::new(),
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

    /// Build the context vector for a path, given the minimum RTT across all
    /// active paths (in nanoseconds).
    fn make_context(
        rtt_ns: u128,
        min_rtt_ns: u128,
        bytes_in_flight: usize,
        cwnd: u64,
    ) -> [f64; D] {
        let rtt_norm = if min_rtt_ns > 0 {
            // Clamp to [1.0, 4.0] so a single very-high-RTT measurement cannot
            // produce a reward of −30 that overwhelms the learned model.
            (rtt_ns as f64 / min_rtt_ns as f64).clamp(1.0, 4.0)
        } else {
            1.0
        };
        let cwnd_pressure = if cwnd > 0 {
            (bytes_in_flight as f64 / cwnd as f64).min(1.0)
        } else {
            1.0
        };
        [rtt_norm, cwnd_pressure, 1.0]
    }

    /// Decompose the UCB score into (reward_estimate, exploration_bonus).
    fn ucb_parts(arm: &ArmState, x: [f64; D], alpha: f64) -> (f64, f64) {
        let a_inv = mat_inv_3(arm.a);
        let theta = mat_vec_3(a_inv, arm.b);
        let reward_est = dot3(theta, x);
        let explore_bonus = alpha * quadratic_3(a_inv, x).max(0.0).sqrt();
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
        let mut raw: Vec<(usize, u128, usize, u64)> = Vec::new();

        for (pid, path) in paths.iter_mut() {
            if !path.active() || !path.recovery.can_send() {
                continue;
            }
            // Use EMA RTT (learned from ACKs) if available; otherwise fall back
            // to smoothed_rtt. This avoids scoring based on the historical
            // min_rtt which is anchored to the sub-ms handshake and never
            // reflects a permanently bad path.
            let rtt_ns = if self.ack_counts.get(pid).copied().unwrap_or(0) > 0 {
                self.ema_rtt_ns[pid] as u128
            } else {
                path.recovery.rtt.smoothed_rtt().as_nanos()
            };
            let bytes_in_flight = path.recovery.bytes_in_flight;
            let cwnd = path.recovery.congestion.congestion_window();
            if rtt_ns < min_rtt_ns {
                min_rtt_ns = rtt_ns;
            }
            raw.push((pid, rtt_ns, bytes_in_flight, cwnd));
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
        // (pid, rtt_us, cwnd_pressure, reward_est, explore_bonus, ucb)
        let mut score_rows: Vec<(usize, u64, f64, f64, f64, f64)> = Vec::new();

        for &(pid, rtt_ns, bytes_in_flight, cwnd) in &raw {
            let x = Self::make_context(rtt_ns, min_rtt_ns, bytes_in_flight, cwnd);
            self.ensure_arm(pid);
            let arm = self.arms[pid].as_ref().unwrap();
            let (est, bonus) = Self::ucb_parts(arm, x, self.alpha);
            let ucb = est + bonus;
            let cwnd_pressure = x[1];
            score_rows.push((pid, (rtt_ns / 1_000) as u64, cwnd_pressure, est, bonus, ucb));
            if ucb > best_score {
                best_score = ucb;
                best_pid = pid;
            }
        }

        // Deficit weighted round-robin using softmax of UCB scores.
        // Temperature 0.3: near-equal paths split traffic; large UCB gaps
        // degenerate to argmax (bad paths get ~0% weight).
        const SPLIT_TAU: f64 = 0.3;
        let max_ucb = best_score;
        let weights: Vec<f64> = score_rows
            .iter()
            .map(|&(.., ucb)| ((ucb - max_ucb) / SPLIT_TAU).exp())
            .collect();
        let total_w: f64 = weights.iter().sum();

        // Grow credits to cover all active path ids.
        let max_active_pid = score_rows.iter().map(|&(pid, ..)| pid).max().unwrap_or(0);
        if max_active_pid >= self.credits.len() {
            self.credits.resize(max_active_pid + 1, 0.0);
        }
        for (i, &(pid, ..)) in score_rows.iter().enumerate() {
            self.credits[pid] += weights[i] / total_w;
        }
        // Choose path with most credits among active paths.
        let best_pid = score_rows
            .iter()
            .map(|&(pid, ..)| pid)
            .max_by(|&a, &b| {
                self.credits[a]
                    .partial_cmp(&self.credits[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(best_pid);
        self.credits[best_pid] -= 1.0;

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
        for &(pid, _, _, _) in &raw {
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
        self.ack_counts[path_id] += 1;
        let alpha = if self.ack_counts[path_id] <= 8 { 0.5_f64 } else { 0.2_f64 };
        self.ema_rtt_ns[path_id] =
            alpha * rtt_ns as f64 + (1.0 - alpha) * self.ema_rtt_ns[path_id];
        let ema_rtt_ns = self.ema_rtt_ns[path_id] as u128;

        let x = Self::make_context(
            ema_rtt_ns,
            self.last_min_rtt_ns, // global minimum from last scheduling decision
            path.recovery.bytes_in_flight,
            path.recovery.congestion.congestion_window(),
        );

        // Reward = 1 − rtt_norm.
        // Best path (rtt_norm = 1.0) → reward = 0.0.
        // Worse paths (rtt_norm > 1.0) → negative reward, discouraging selection.
        // cwnd_pressure in x[1] further penalises congested paths via theta.
        let reward = 1.0 - x[0];

        self.ensure_arm(path_id);
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
            // Final model estimate using a neutral context (rtt_norm=1.0, pressure=0.0, bias=1.0).
            let est_str = if let Some(Some(arm)) = self.arms.get(pid) {
                let x = [1.0_f64, 0.0_f64, 1.0_f64];
                let a_inv = mat_inv_3(arm.a);
                let theta = mat_vec_3(a_inv, arm.b);
                let est = dot3(theta, x);
                format!("{est:+.3}")
            } else {
                "n/a".into()
            };
            out.push_str(&format!(
                "    path[{pid}] {addr}  selections={cnt} ({pct}%)  est(neutral)={est_str}\n"
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
}

// ─── 3×3 linear algebra helpers ──────────────────────────────────────────────
//
// These operate on plain arrays to avoid external dependencies.

/// Compute the inverse of a 3×3 matrix using Cramer's rule.
///
/// Returns the identity matrix when the determinant is near zero to avoid
/// numerical blow-up during early exploration.
fn mat_inv_3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-15 {
        let mut r = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            r[i][i] = 1.0;
        }
        return r;
    }

    let inv = 1.0 / det;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ]
}

/// Multiply a 3×3 matrix by a 3-vector.
fn mat_vec_3(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    let mut r = [0.0_f64; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i] += m[i][j] * v[j];
        }
    }
    r
}

/// Dot product of two 3-vectors.
fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Quadratic form xᵀ M x for a 3×3 matrix M and 3-vector x.
fn quadratic_3(m: [[f64; 3]; 3], x: [f64; 3]) -> f64 {
    let mx = mat_vec_3(m, x);
    dot3(x, mx)
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
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = mat_inv_3(id);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((inv[i][j] - expected).abs() < 1e-10);
            }
        }
    }
}
