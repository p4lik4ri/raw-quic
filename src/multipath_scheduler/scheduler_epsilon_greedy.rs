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

use rand::Rng;

use crate::connection::path::PathMap;
use crate::connection::space::PacketNumSpaceMap;
use crate::connection::stream::StreamMap;
use crate::multipath_scheduler::traffic_metrics::TrafficMetricsCollector;
use crate::multipath_scheduler::MultipathScheduler;
use crate::Error;
use crate::MultipathConfig;
use crate::Result;

/// Number of context features used by the linear reward model.
///
/// Features per path are expressed as "goodness" scores, so larger usually
/// means better:
///   x[0] = min_rtt / rtt                  (global latency score, 1.0 for best RTT)
///   x[1] = 1 - bytes_in_flight / cwnd     (congestion-window headroom)
///   x[2] = 1 - ewma_recent_loss_rate      (recent reliability score)
///   x[3] = pacing_rate / max_pacing_rate  (relative pacing-rate score)
///   x[4] = 1.0                            (bias / intercept term)
const D: usize = 5;

/// Initial probability of choosing a random sendable path instead of the path
/// with the highest predicted reward.
const EPSILON_INITIAL: f64 = 0.20;

/// Exploration floor. A small non-zero epsilon lets the scheduler keep probing
/// path quality after the model has converged.
const EPSILON_MIN: f64 = 0.02;

/// Number of model updates over which epsilon decays toward EPSILON_MIN.
const EPSILON_DECAY_UPDATES: f64 = 2_000.0;

/// Force each available path to receive initial selections before normal
/// epsilon-greedy exploitation starts. This is based on selections, not ACKs,
/// so bursts before the first ACK still cover every path.
const WARMUP_SELECTIONS_PER_PATH: u64 = 5;

/// Flush feedback after this many observed ACK/loss outcomes on a path.
const FEEDBACK_WINDOW_OUTCOMES: u64 = 32;

/// Also flush a non-empty feedback window when it gets old enough.
const FEEDBACK_WINDOW_DURATION: Duration = Duration::from_millis(250);

/// If selected packets on a path produce no ACK/loss feedback for this long,
/// train the pending context once with zero reward.
const STALE_DECISION_TIMEOUT: Duration = Duration::from_secs(1);

/// Base learning rate for the online linear update. The effective rate decays
/// as each path accumulates model updates.
const LEARNING_RATE: f64 = 0.05;

/// Small L2 penalty used by the online update to keep weights bounded.
const L2_REG: f64 = 0.0001;

/// Smoothing factor for recent loss used as a selection-time feature.
const LOSS_EWMA_ALPHA: f64 = 0.20;

#[derive(Clone)]
struct ArmState {
    theta: [f64; D],
    samples: u64,
    selections: u64,

    /// Contexts selected since the last feedback update. The arm is trained on
    /// their average when ACK/loss counter deltas close the feedback window.
    pending_context_sum: [f64; D],
    pending_decisions: u64,
    pending_since: Option<Instant>,

    /// Last cumulative counters observed by on_ack.
    counters_initialized: bool,
    prev_acked_pkts: u64,
    prev_lost_pkts: u64,
    prev_sent_bytes: u64,
    prev_acked_bytes: u64,
    prev_lost_bytes: u64,

    /// Counter deltas accumulated in the current feedback window.
    window_acked_pkts: u64,
    window_lost_pkts: u64,
    window_sent_bytes: u64,
    window_acked_bytes: u64,
    window_lost_bytes: u64,
    window_start: Option<Instant>,

    base_rtt_ns: u128,
    loss_ewma: f64,
    loss_ewma_init: bool,
}

impl ArmState {
    fn new() -> Self {
        Self {
            theta: [0.0; D],
            samples: 0,
            selections: 0,
            pending_context_sum: [0.0; D],
            pending_decisions: 0,
            pending_since: None,
            counters_initialized: false,
            prev_acked_pkts: 0,
            prev_lost_pkts: 0,
            prev_sent_bytes: 0,
            prev_acked_bytes: 0,
            prev_lost_bytes: 0,
            window_acked_pkts: 0,
            window_lost_pkts: 0,
            window_sent_bytes: 0,
            window_acked_bytes: 0,
            window_lost_bytes: 0,
            window_start: None,
            base_rtt_ns: 0,
            loss_ewma: 0.0,
            loss_ewma_init: false,
        }
    }

    fn push_pending_context(&mut self, x: [f64; D]) {
        for i in 0..D {
            self.pending_context_sum[i] += x[i];
        }
        if self.pending_since.is_none() {
            self.pending_since = Some(Instant::now());
        }
        self.pending_decisions += 1;
        self.selections += 1;
    }

    fn take_pending_context(&mut self) -> Option<[f64; D]> {
        if self.pending_decisions == 0 {
            return None;
        }

        let mut x = [0.0; D];
        for i in 0..D {
            x[i] = self.pending_context_sum[i] / self.pending_decisions as f64;
            self.pending_context_sum[i] = 0.0;
        }
        self.pending_decisions = 0;
        self.pending_since = None;
        Some(x)
    }

    fn reset_feedback_window(&mut self, now: Instant) {
        self.window_acked_pkts = 0;
        self.window_lost_pkts = 0;
        self.window_sent_bytes = 0;
        self.window_acked_bytes = 0;
        self.window_lost_bytes = 0;
        self.window_start = Some(now);
    }
}

#[derive(Clone, Copy)]
struct Candidate {
    pid: usize,
    x: [f64; D],
    prediction: f64,
}

/// EpsilonGreedyScheduler implements an online contextual bandit scheduler.
///
/// on_select computes per-path contexts, chooses a path with epsilon-greedy,
/// and saves the selected context as a pending decision. on_ack only consumes
/// ACK/loss counter deltas. A fixed feedback window then converts those
/// observed deltas into a reward and updates the selected arm's linear model.
pub struct EpsilonGreedyScheduler {
    arms: Vec<Option<ArmState>>,
    last_min_rtt_ns: u128,
    metrics: TrafficMetricsCollector,

    /// Per-path selection counts accumulated between internal log snapshots.
    selection_window_counts: Vec<u64>,

    /// Total selections accumulated between internal log snapshots.
    selection_window_total: u64,

    /// Last time a selection snapshot was emitted.
    last_internal_log: Option<Instant>,

    /// Extra JSONL records used to document scheduler experiments.
    internal_metrics_jsonl: Vec<String>,
}

impl EpsilonGreedyScheduler {
    pub fn new(_conf: &MultipathConfig) -> Self {
        Self {
            arms: Vec::new(),
            last_min_rtt_ns: 1,
            metrics: TrafficMetricsCollector::new(),
            selection_window_counts: Vec::new(),
            selection_window_total: 0,
            last_internal_log: None,
            internal_metrics_jsonl: Vec::new(),
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
        recent_loss_rate: f64,
        pacing_rate_bytes_per_sec: u64,
        max_pacing_rate_bytes_per_sec: u64,
    ) -> [f64; D] {
        let latency_score = if rtt_ns > 0 && min_rtt_ns > 0 {
            (min_rtt_ns as f64 / rtt_ns as f64).clamp(0.25, 1.0)
        } else {
            1.0
        };

        let cwnd_pressure = if cwnd > 0 {
            (bytes_in_flight as f64 / cwnd as f64).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let cwnd_headroom = 1.0 - cwnd_pressure;
        let reliability_score = 1.0 - recent_loss_rate.clamp(0.0, 1.0);

        let relative_pacing_rate_score = if max_pacing_rate_bytes_per_sec > 0 {
            (pacing_rate_bytes_per_sec as f64 / max_pacing_rate_bytes_per_sec as f64)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };

        [
            latency_score,
            cwnd_headroom,
            reliability_score,
            relative_pacing_rate_score,
            1.0,
        ]
    }

    fn predict(theta: [f64; D], x: [f64; D]) -> f64 {
        dot(theta, x)
    }

    fn update_arm(arm: &mut ArmState, x: [f64; D], reward: f64) {
        let prediction = Self::predict(arm.theta, x);
        let error = reward - prediction;
        let eta = LEARNING_RATE / ((arm.samples + 1) as f64).sqrt();

        for i in 0..D {
            arm.theta[i] += eta * (error * x[i] - L2_REG * arm.theta[i]);
        }
        arm.samples += 1;
    }

    fn observed_reward(
        acked_pkts: u64,
        lost_pkts: u64,
        sent_bytes: u64,
        acked_bytes: u64,
        _lost_bytes: u64,
        window_duration: Duration,
        rtt_ns: u128,
        path_base_rtt_ns: u128,
        pacing_rate_bytes_per_sec: u64,
        max_pacing_rate_bytes_per_sec: u64,
        bytes_in_flight: usize,
        cwnd: u64,
    ) -> Option<f64> {
        let pkt_outcomes = acked_pkts.saturating_add(lost_pkts);
        if pkt_outcomes == 0 || sent_bytes == 0 {
            return None;
        }

        let loss_rate = lost_pkts as f64 / pkt_outcomes as f64;
        let reliability_score = (-20.0 * loss_rate.clamp(0.0, 1.0)).exp();

        let delivery_efficiency = (acked_bytes as f64 / sent_bytes as f64).clamp(0.0, 1.0);
        let served_bytes_per_sec = sent_bytes as f64 / window_duration.as_secs_f64().max(0.001);
        // congestion_control::pacing_rate() returns bytes/sec internally. qlog
        // multiplies it by 8 only when exporting bps, so keep reward units in
        // bytes/sec here.
        let served_rate_baseline =
            max_pacing_rate_bytes_per_sec.max(pacing_rate_bytes_per_sec);
        let served_rate_score = if served_rate_baseline > 0 {
            (served_bytes_per_sec / served_rate_baseline as f64).clamp(0.0, 1.0)
        } else {
            1.0
        };
        let goodput_score = delivery_efficiency * served_rate_score;

        // Context x[0] uses global min RTT for cross-path selection. Reward
        // uses a path-local base RTT to penalize queueing/inflation on the
        // selected path without punishing inherently high-delay paths forever.
        let latency_score = if rtt_ns > 0 && path_base_rtt_ns > 0 {
            (path_base_rtt_ns as f64 / rtt_ns as f64).clamp(0.25, 1.0)
        } else {
            1.0
        };

        let queue_score = if cwnd > 0 {
            let saturation = (bytes_in_flight as f64 / cwnd as f64).clamp(0.0, 1.0);
            1.0 - saturation * saturation
        } else {
            0.0
        };

        Some(
            (0.45 * goodput_score
                + 0.25 * reliability_score
                + 0.20 * latency_score
                + 0.10 * queue_score)
                .clamp(0.0, 1.0),
        )
    }

    fn update_loss_ewma(arm: &mut ArmState, acked_pkts: u64, lost_pkts: u64) {
        let outcomes = acked_pkts.saturating_add(lost_pkts);
        if outcomes == 0 {
            return;
        }

        let instant_loss = (lost_pkts as f64 / outcomes as f64).clamp(0.0, 1.0);
        if arm.loss_ewma_init {
            arm.loss_ewma = LOSS_EWMA_ALPHA * instant_loss
                + (1.0 - LOSS_EWMA_ALPHA) * arm.loss_ewma;
        } else {
            arm.loss_ewma = instant_loss;
            arm.loss_ewma_init = true;
        }
    }

    fn record_feedback_counters(
        arm: &mut ArmState,
        now: Instant,
        acked_pkts: u64,
        lost_pkts: u64,
        sent_bytes: u64,
        acked_bytes: u64,
        lost_bytes: u64,
        rtt_ns: u128,
        pacing_rate_bytes_per_sec: u64,
        max_pacing_rate_bytes_per_sec: u64,
        bytes_in_flight: usize,
        cwnd: u64,
    ) -> Option<([f64; D], f64)> {
        if !arm.counters_initialized {
            arm.prev_acked_pkts = acked_pkts;
            arm.prev_lost_pkts = lost_pkts;
            arm.prev_sent_bytes = sent_bytes;
            arm.prev_acked_bytes = acked_bytes;
            arm.prev_lost_bytes = lost_bytes;
            if arm.base_rtt_ns == 0 || rtt_ns < arm.base_rtt_ns {
                arm.base_rtt_ns = rtt_ns;
            }
            arm.counters_initialized = true;
            return None;
        }

        let delta_acked_pkts = acked_pkts.saturating_sub(arm.prev_acked_pkts);
        let delta_lost_pkts = lost_pkts.saturating_sub(arm.prev_lost_pkts);
        let delta_sent_bytes = sent_bytes.saturating_sub(arm.prev_sent_bytes);
        let delta_acked_bytes = acked_bytes.saturating_sub(arm.prev_acked_bytes);
        let delta_lost_bytes = lost_bytes.saturating_sub(arm.prev_lost_bytes);

        arm.prev_acked_pkts = acked_pkts;
        arm.prev_lost_pkts = lost_pkts;
        arm.prev_sent_bytes = sent_bytes;
        arm.prev_acked_bytes = acked_bytes;
        arm.prev_lost_bytes = lost_bytes;

        let delta_outcomes = delta_acked_pkts.saturating_add(delta_lost_pkts);
        if delta_outcomes == 0 {
            return None;
        }

        if arm.base_rtt_ns == 0 || rtt_ns < arm.base_rtt_ns {
            arm.base_rtt_ns = rtt_ns;
        }
        if arm.window_start.is_none() {
            arm.window_start = Some(now);
        }

        arm.window_acked_pkts = arm.window_acked_pkts.saturating_add(delta_acked_pkts);
        arm.window_lost_pkts = arm.window_lost_pkts.saturating_add(delta_lost_pkts);
        arm.window_sent_bytes = arm.window_sent_bytes.saturating_add(delta_sent_bytes);
        arm.window_acked_bytes = arm.window_acked_bytes.saturating_add(delta_acked_bytes);
        arm.window_lost_bytes = arm.window_lost_bytes.saturating_add(delta_lost_bytes);
        Self::update_loss_ewma(arm, delta_acked_pkts, delta_lost_pkts);

        let window_outcomes = arm.window_acked_pkts.saturating_add(arm.window_lost_pkts);
        let window_old_enough = arm
            .window_start
            .map(|start| now.duration_since(start) >= FEEDBACK_WINDOW_DURATION)
            .unwrap_or(false);

        if window_outcomes < FEEDBACK_WINDOW_OUTCOMES && !window_old_enough {
            return None;
        }

        let window_duration = arm
            .window_start
            .map(|start| now.duration_since(start))
            .unwrap_or_default();
        let reward = Self::observed_reward(
            arm.window_acked_pkts,
            arm.window_lost_pkts,
            arm.window_sent_bytes,
            arm.window_acked_bytes,
            arm.window_lost_bytes,
            window_duration,
            rtt_ns,
            arm.base_rtt_ns,
            pacing_rate_bytes_per_sec,
            max_pacing_rate_bytes_per_sec,
            bytes_in_flight,
            cwnd,
        )?;
        let context = arm.take_pending_context();
        arm.reset_feedback_window(now);

        context.map(|x| (x, reward))
    }

    fn total_samples(&self) -> u64 {
        self.arms
            .iter()
            .filter_map(|arm| arm.as_ref().map(|arm| arm.samples))
            .sum()
    }

    fn effective_epsilon(&self) -> f64 {
        let t = self.total_samples() as f64;
        let decay = (-t / EPSILON_DECAY_UPDATES).exp();
        (EPSILON_MIN + (EPSILON_INITIAL - EPSILON_MIN) * decay)
            .clamp(EPSILON_MIN, EPSILON_INITIAL)
    }

    fn train_stale_pending_decisions(&mut self, now: Instant) {
        for path_id in 0..self.arms.len() {
            let stale_update = {
                let Some(arm) = self.arms[path_id].as_mut() else {
                    continue;
                };

                let is_stale = arm
                    .pending_since
                    .map(|since| now.duration_since(since) > STALE_DECISION_TIMEOUT)
                    .unwrap_or(false);
                if !is_stale {
                    continue;
                }

                let x = arm.take_pending_context();
                if let Some(x) = x {
                    Self::update_arm(arm, x, 0.0);
                    arm.reset_feedback_window(now);
                    Some(x)
                } else {
                    arm.reset_feedback_window(now);
                    None
                }
            };

            if let Some(x) = stale_update {
                self.log_model_update(now, path_id, x, 0.0, "stale_timeout");
            }
        }
    }

    fn log_selection_snapshot(
        &mut self,
        now: Instant,
        selected_pid: usize,
        candidates: &[Candidate],
    ) {
        let do_log = self
            .last_internal_log
            .map(|t| now.duration_since(t) >= Duration::from_secs(1))
            .unwrap_or(true);

        if !do_log {
            return;
        }

        let elapsed_s = now.duration_since(self.metrics.start_time).as_secs();
        let timestamp = self.metrics.start_unix_secs + elapsed_s;
        let step = self.internal_metrics_jsonl.len() as u64;
        let total = self.selection_window_total.max(1);
        let epsilon = self.effective_epsilon();

        let mut jline = format!(
            "{{\"_step\":{step},\"_timestamp\":{timestamp},\"t\":{elapsed_s},\
             \"epsilon_greedy/event\":\"selection_snapshot\",\
             \"epsilon_greedy/selected_path\":{selected_pid},\
             \"epsilon_greedy/epsilon\":{epsilon:.6}"
        );

        for c in candidates {
            let pid = c.pid;
            let arm = self.arms[pid].as_ref().unwrap();
            let cnt = self.selection_window_counts.get(pid).copied().unwrap_or(0);
            let pct = cnt as f64 * 100.0 / total as f64;

            jline.push_str(&format!(
                ",\"epsilon_greedy/path{pid}_selection_pct\":{pct:.2}\
                 ,\"epsilon_greedy/path{pid}_prediction\":{:.6}\
                 ,\"epsilon_greedy/path{pid}_samples\":{}\
                 ,\"epsilon_greedy/path{pid}_selections\":{}\
                 ,\"epsilon_greedy/path{pid}_pending\":{}\
                 ,\"epsilon_greedy/path{pid}_loss_ewma\":{:.6}\
                 ,\"epsilon_greedy/path{pid}_x_latency\":{:.6}\
                 ,\"epsilon_greedy/path{pid}_x_cwnd\":{:.6}\
                 ,\"epsilon_greedy/path{pid}_x_reliability\":{:.6}\
                 ,\"epsilon_greedy/path{pid}_x_pacing\":{:.6}\
                 ,\"epsilon_greedy/path{pid}_theta_latency\":{:.6}\
                 ,\"epsilon_greedy/path{pid}_theta_cwnd\":{:.6}\
                 ,\"epsilon_greedy/path{pid}_theta_reliability\":{:.6}\
                 ,\"epsilon_greedy/path{pid}_theta_pacing\":{:.6}\
                 ,\"epsilon_greedy/path{pid}_theta_bias\":{:.6}",
                c.prediction,
                arm.samples,
                arm.selections,
                arm.pending_decisions,
                arm.loss_ewma,
                c.x[0],
                c.x[1],
                c.x[2],
                c.x[3],
                arm.theta[0],
                arm.theta[1],
                arm.theta[2],
                arm.theta[3],
                arm.theta[4],
            ));
        }

        jline.push('}');
        self.internal_metrics_jsonl.push(jline);

        self.last_internal_log = Some(now);
        for c in &mut self.selection_window_counts {
            *c = 0;
        }
        self.selection_window_total = 0;
    }

    fn log_model_update(
        &mut self,
        now: Instant,
        path_id: usize,
        x: [f64; D],
        reward: f64,
        source: &str,
    ) {
        let Some(arm) = self.arms[path_id].as_ref() else {
            return;
        };

        let elapsed_s = now.duration_since(self.metrics.start_time).as_secs();
        let timestamp = self.metrics.start_unix_secs + elapsed_s;
        let step = self.internal_metrics_jsonl.len() as u64;
        let epsilon = self.effective_epsilon();

        self.internal_metrics_jsonl.push(format!(
            "{{\"_step\":{step},\"_timestamp\":{timestamp},\"t\":{elapsed_s},\
             \"epsilon_greedy/event\":\"model_update\",\
             \"epsilon_greedy/update_source\":\"{source}\",\
             \"epsilon_greedy/path\":{path_id},\
             \"epsilon_greedy/reward\":{reward:.6},\
             \"epsilon_greedy/epsilon\":{epsilon:.6},\
             \"epsilon_greedy/x_latency\":{:.6},\
             \"epsilon_greedy/x_cwnd\":{:.6},\
             \"epsilon_greedy/x_reliability\":{:.6},\
             \"epsilon_greedy/x_pacing\":{:.6},\
             \"epsilon_greedy/x_bias\":{:.6},\
             \"epsilon_greedy/samples\":{},\
             \"epsilon_greedy/selections\":{},\
             \"epsilon_greedy/pending\":{},\
             \"epsilon_greedy/loss_ewma\":{:.6},\
             \"epsilon_greedy/theta_latency\":{:.6},\
             \"epsilon_greedy/theta_cwnd\":{:.6},\
             \"epsilon_greedy/theta_reliability\":{:.6},\
             \"epsilon_greedy/theta_pacing\":{:.6},\
             \"epsilon_greedy/theta_bias\":{:.6}}}",
            x[0],
            x[1],
            x[2],
            x[3],
            x[4],
            arm.samples,
            arm.selections,
            arm.pending_decisions,
            arm.loss_ewma,
            arm.theta[0],
            arm.theta[1],
            arm.theta[2],
            arm.theta[3],
            arm.theta[4],
        ));
    }
}

impl MultipathScheduler for EpsilonGreedyScheduler {
    fn on_select(
        &mut self,
        paths: &mut PathMap,
        _spaces: &mut PacketNumSpaceMap,
        _streams: &mut StreamMap,
    ) -> Result<usize> {
        self.train_stale_pending_decisions(Instant::now());

        let mut min_rtt_ns = u128::MAX;
        let mut max_pacing_rate_bytes_per_sec = 0;
        let mut raw = Vec::new();

        for (pid, path) in paths.iter_mut() {
            if !path.active() || !path.recovery.can_send() {
                continue;
            }

            let rtt_ns = path.recovery.rtt.smoothed_rtt().as_nanos().max(1);
            let bytes_in_flight = path.recovery.bytes_in_flight;
            let cwnd = path.recovery.congestion.congestion_window();
            let pacing_rate_bytes_per_sec = path.recovery.congestion.pacing_rate().unwrap_or(0);

            min_rtt_ns = min_rtt_ns.min(rtt_ns);
            max_pacing_rate_bytes_per_sec =
                max_pacing_rate_bytes_per_sec.max(pacing_rate_bytes_per_sec);
            raw.push((
                pid,
                rtt_ns,
                bytes_in_flight,
                cwnd,
                pacing_rate_bytes_per_sec,
            ));
        }

        if raw.is_empty() {
            return Err(Error::Done);
        }

        let min_rtt_ns = min_rtt_ns.max(1);
        self.last_min_rtt_ns = min_rtt_ns;

        let mut candidates = Vec::with_capacity(raw.len());
        for (pid, rtt_ns, bytes_in_flight, cwnd, pacing_rate_bytes_per_sec) in raw {
            self.ensure_arm(pid);
            let arm = self.arms[pid].as_ref().unwrap();
            let x = Self::make_context(
                rtt_ns,
                min_rtt_ns,
                bytes_in_flight,
                cwnd,
                arm.loss_ewma,
                pacing_rate_bytes_per_sec,
                max_pacing_rate_bytes_per_sec,
            );
            candidates.push(Candidate {
                pid,
                x,
                prediction: Self::predict(arm.theta, x),
            });
        }

        let mut rng = rand::thread_rng();
        let min_selections = candidates
            .iter()
            .filter_map(|c| self.arms[c.pid].as_ref().map(|arm| arm.selections))
            .min()
            .unwrap_or(0);

        let selected = if min_selections < WARMUP_SELECTIONS_PER_PATH {
            let under_sampled: Vec<Candidate> = candidates
                .iter()
                .copied()
                .filter(|c| {
                    self.arms[c.pid]
                        .as_ref()
                        .map(|arm| arm.selections == min_selections)
                        .unwrap_or(false)
                })
                .collect();
            under_sampled[rng.gen_range(0..under_sampled.len())]
        } else if candidates.len() > 1 && rng.gen_bool(self.effective_epsilon()) {
            candidates[rng.gen_range(0..candidates.len())]
        } else {
            let best_prediction = candidates
                .iter()
                .map(|c| c.prediction)
                .fold(f64::NEG_INFINITY, f64::max);
            let tied_best: Vec<Candidate> = candidates
                .iter()
                .copied()
                .filter(|c| (c.prediction - best_prediction).abs() < f64::EPSILON)
                .collect();
            tied_best[rng.gen_range(0..tied_best.len())]
        };

        let arm = self.arms[selected.pid].as_mut().unwrap();
        arm.push_pending_context(selected.x);

        if selected.pid >= self.selection_window_counts.len() {
            self.selection_window_counts.resize(selected.pid + 1, 0);
        }
        self.selection_window_counts[selected.pid] += 1;
        self.selection_window_total += 1;

        let now = Instant::now();
        self.log_selection_snapshot(now, selected.pid, &candidates);

        self.metrics.record(selected.pid, paths);
        Ok(selected.pid)
    }

    fn on_ack(&mut self, now: Instant, path_id: usize, paths: &mut PathMap) {
        self.train_stale_pending_decisions(now);

        let mut max_pacing_rate_bytes_per_sec = 0;
        let mut feedback = None;

        for (pid, path) in paths.iter_mut() {
            let rtt_ns = path.recovery.rtt.smoothed_rtt().as_nanos().max(1);
            let pacing_rate_bytes_per_sec = path.recovery.congestion.pacing_rate().unwrap_or(0);

            if path.active() {
                max_pacing_rate_bytes_per_sec =
                    max_pacing_rate_bytes_per_sec.max(pacing_rate_bytes_per_sec);
            }

            if pid == path_id {
                feedback = Some((
                    path.recovery.stats.acked_count,
                    path.recovery.stats.lost_count,
                    path.recovery.stats.sent_bytes,
                    path.recovery.stats.acked_bytes,
                    path.recovery.stats.lost_bytes,
                    rtt_ns,
                    pacing_rate_bytes_per_sec,
                    path.recovery.bytes_in_flight,
                    path.recovery.congestion.congestion_window(),
                ));
            }
        }

        let Some((
            acked_pkts,
            lost_pkts,
            sent_bytes,
            acked_bytes,
            lost_bytes,
            rtt_ns,
            pacing_rate_bytes_per_sec,
            bytes_in_flight,
            cwnd,
        )) = feedback
        else {
            return;
        };

        self.ensure_arm(path_id);
        let update = {
            let arm = self.arms[path_id].as_mut().unwrap();
            Self::record_feedback_counters(
                arm,
                now,
                acked_pkts,
                lost_pkts,
                sent_bytes,
                acked_bytes,
                lost_bytes,
                rtt_ns,
                pacing_rate_bytes_per_sec,
                max_pacing_rate_bytes_per_sec,
                bytes_in_flight,
                cwnd,
            )
        };

        if let Some((x, reward)) = update {
            {
                let arm = self.arms[path_id].as_mut().unwrap();
                Self::update_arm(arm, x, reward);
            }

            self.log_model_update(now, path_id, x, reward, "ack_feedback");
        }
    }

    fn scheduler_summary(&self) -> Option<String> {
        if self.arms.iter().all(|arm| {
            arm.as_ref()
                .map(|arm| arm.samples == 0)
                .unwrap_or(true)
        }) {
            return None;
        }

        let mut out = format!(
            "Epsilon-greedy scheduler summary\n  epsilon_current={:.4}\n",
            self.effective_epsilon(),
        );
        for (pid, arm) in self.arms.iter().enumerate() {
            let Some(arm) = arm else {
                continue;
            };
            if arm.samples == 0 {
                continue;
            }
            out.push_str(&format!(
                "  path[{pid}] updates={} selections={} pending={} loss_ewma={:.4} theta=[{:.4}, {:.4}, {:.4}, {:.4}, {:.4}]\n",
                arm.samples,
                arm.selections,
                arm.pending_decisions,
                arm.loss_ewma,
                arm.theta[0],
                arm.theta[1],
                arm.theta[2],
                arm.theta[3],
                arm.theta[4],
            ));
        }
        Some(out)
    }

    fn scheduler_metrics_jsonl(&self) -> Vec<String> {
        let mut metrics = self.metrics.metrics.clone();
        metrics.extend(self.internal_metrics_jsonl.clone());
        metrics
    }
}

fn dot(a: [f64; D], b: [f64; D]) -> f64 {
    let mut s = 0.0;
    for i in 0..D {
        s += a[i] * b[i];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multipath_scheduler::tests::*;

    #[test]
    fn epsilon_greedy_single_path() -> Result<()> {
        let mut t = MultipathTester::new()?;
        let mut s = EpsilonGreedyScheduler::new(&Default::default());

        assert_eq!(s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?, 0);
        Ok(())
    }

    #[test]
    fn epsilon_greedy_multi_path_selects_valid_path() -> Result<()> {
        let mut t = MultipathTester::new()?;
        t.add_path("127.0.0.1:443", "127.0.0.2:8443", 50)?;
        t.add_path("127.0.0.1:443", "127.0.0.3:8443", 150)?;

        let mut s = EpsilonGreedyScheduler::new(&Default::default());
        let pid = s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?;
        assert!(pid <= 2);
        Ok(())
    }

    #[test]
    fn epsilon_greedy_warmup_uses_selection_counts() -> Result<()> {
        let mut t = MultipathTester::new()?;
        t.add_path("127.0.0.1:443", "127.0.0.2:8443", 50)?;
        t.add_path("127.0.0.1:443", "127.0.0.3:8443", 150)?;

        let mut s = EpsilonGreedyScheduler::new(&Default::default());
        for _ in 0..3 {
            s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?;
        }

        for pid in 0..3 {
            assert_eq!(s.arms[pid].as_ref().unwrap().selections, 1);
        }
        Ok(())
    }

    #[test]
    fn epsilon_greedy_window_feedback_updates_model() -> Result<()> {
        let mut t = MultipathTester::new()?;
        let mut s = EpsilonGreedyScheduler::new(&Default::default());

        let pid = s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?;
        s.on_ack(Instant::now(), pid, &mut t.paths);
        {
            let path = t.paths.get_mut(pid)?;
            path.recovery.stats.acked_count += FEEDBACK_WINDOW_OUTCOMES;
            path.recovery.stats.sent_bytes += FEEDBACK_WINDOW_OUTCOMES * 1200;
            path.recovery.stats.acked_bytes += FEEDBACK_WINDOW_OUTCOMES * 1200;
        }
        s.on_ack(Instant::now(), pid, &mut t.paths);

        let arm = s.arms[pid].as_ref().unwrap();
        assert_eq!(arm.samples, 1);
        assert_eq!(arm.pending_decisions, 0);
        assert!(arm.theta.iter().any(|v| *v != 0.0));
        Ok(())
    }

    #[test]
    fn first_feedback_only_initializes_counters() -> Result<()> {
        let mut t = MultipathTester::new()?;
        let mut s = EpsilonGreedyScheduler::new(&Default::default());

        let pid = s.on_select(&mut t.paths, &mut t.spaces, &mut t.streams)?;
        {
            let path = t.paths.get_mut(pid)?;
            path.recovery.stats.acked_count += 100;
            path.recovery.stats.sent_bytes += 100_000;
            path.recovery.stats.acked_bytes += 100_000;
        }
        s.on_ack(Instant::now(), pid, &mut t.paths);

        let arm = s.arms[pid].as_ref().unwrap();
        assert!(arm.counters_initialized);
        assert_eq!(arm.samples, 0);
        assert_eq!(arm.pending_decisions, 1);
        Ok(())
    }

    #[test]
    fn invalid_reward_keeps_pending_context() {
        let now = Instant::now();
        let mut arm = ArmState::new();
        arm.push_pending_context([1.0, 1.0, 1.0, 1.0, 1.0]);
        arm.counters_initialized = true;

        let update = EpsilonGreedyScheduler::record_feedback_counters(
            &mut arm,
            now,
            FEEDBACK_WINDOW_OUTCOMES,
            0,
            0,
            FEEDBACK_WINDOW_OUTCOMES * 1200,
            0,
            100,
            1_000_000,
            1_000_000,
            0,
            10_000,
        );

        assert!(update.is_none());
        assert_eq!(arm.pending_decisions, 1);
    }

    #[test]
    fn stale_pending_decision_trains_zero_reward() {
        let mut s = EpsilonGreedyScheduler::new(&Default::default());
        s.ensure_arm(0);
        {
            let arm = s.arms[0].as_mut().unwrap();
            arm.push_pending_context([1.0, 0.0, 1.0, 1.0, 1.0]);
            arm.pending_since =
                Some(Instant::now() - STALE_DECISION_TIMEOUT - STALE_DECISION_TIMEOUT);
        }

        s.train_stale_pending_decisions(Instant::now());

        let arm = s.arms[0].as_ref().unwrap();
        assert_eq!(arm.samples, 1);
        assert_eq!(arm.pending_decisions, 0);
    }

    #[test]
    fn goodness_context_has_expected_direction() {
        let x_fast = EpsilonGreedyScheduler::make_context(50, 50, 0, 10_000, 0.0, 100, 100);
        let x_slow = EpsilonGreedyScheduler::make_context(200, 50, 9_000, 10_000, 0.5, 25, 100);

        assert!(x_fast[0] > x_slow[0]);
        assert!(x_fast[1] > x_slow[1]);
        assert!(x_fast[2] > x_slow[2]);
        assert!(x_fast[3] > x_slow[3]);
    }

    #[test]
    fn observed_reward_prefers_goodput_and_low_latency() {
        let fast_reward = EpsilonGreedyScheduler::observed_reward(
            32,
            0,
            32 * 1200,
            32 * 1200,
            0,
            Duration::from_millis(100),
            50,
            50,
            400_000,
            400_000,
            0,
            10_000,
        )
        .unwrap();
        let slow_reward = EpsilonGreedyScheduler::observed_reward(
            32,
            0,
            32 * 1200,
            8 * 1200,
            0,
            Duration::from_millis(100),
            200,
            50,
            400_000,
            400_000,
            9_000,
            10_000,
        )
        .unwrap();

        assert!(fast_reward > slow_reward);
    }

    #[test]
    fn context_zero_pacing_rate_scores_zero() {
        let x = EpsilonGreedyScheduler::make_context(50, 50, 0, 10_000, 0.0, 0, 100);

        assert_eq!(x[3], 0.0);
    }

    #[test]
    fn observed_reward_uses_nonlinear_loss_penalty() {
        let no_loss = EpsilonGreedyScheduler::observed_reward(
            100,
            0,
            100_000,
            100_000,
            0,
            Duration::from_millis(100),
            100,
            100,
            1_000_000,
            1_000_000,
            0,
            10_000,
        )
        .unwrap();
        let ten_percent_loss = EpsilonGreedyScheduler::observed_reward(
            90,
            10,
            100_000,
            90_000,
            10_000,
            Duration::from_millis(100),
            100,
            100,
            1_000_000,
            1_000_000,
            0,
            10_000,
        )
        .unwrap();

        assert!(no_loss - ten_percent_loss > 0.15);
    }

    #[test]
    fn observed_reward_does_not_reward_idle_path_from_queue_score() {
        let reward = EpsilonGreedyScheduler::observed_reward(
            0,
            0,
            0,
            0,
            0,
            Duration::from_millis(100),
            100,
            100,
            1_000_000,
            1_000_000,
            0,
            10_000,
        );

        assert!(reward.is_none());
    }

    #[test]
    fn observed_reward_uses_path_local_base_rtt() {
        let near_base = EpsilonGreedyScheduler::observed_reward(
            32,
            0,
            32_000,
            32_000,
            0,
            Duration::from_millis(100),
            200,
            200,
            320_000,
            320_000,
            0,
            10_000,
        )
        .unwrap();
        let above_base = EpsilonGreedyScheduler::observed_reward(
            32,
            0,
            32_000,
            32_000,
            0,
            Duration::from_millis(100),
            200,
            100,
            320_000,
            320_000,
            0,
            10_000,
        )
        .unwrap();

        assert!(near_base > above_base);
    }

    #[test]
    fn observed_goodput_is_normalized_by_sent_bytes() {
        let efficient = EpsilonGreedyScheduler::observed_reward(
            32,
            0,
            32_000,
            32_000,
            0,
            Duration::from_millis(100),
            100,
            100,
            1_000,
            1_000,
            0,
            10_000,
        )
        .unwrap();
        let inefficient = EpsilonGreedyScheduler::observed_reward(
            32,
            0,
            64_000,
            32_000,
            0,
            Duration::from_millis(100),
            100,
            100,
            1_000,
            1_000,
            0,
            10_000,
        )
        .unwrap();

        assert!(efficient > inefficient);
    }
}
