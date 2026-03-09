use std::collections::VecDeque;
use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::state::OUTPUT_CAP;

// ─────────────────────────────────── formatting ───────────────────────────────

/// Format a float like Python: bare `0` for zero, minimal digits otherwise.
pub fn fmt_float(v: f64) -> String {
    if v == 0.0 {
        "0".to_string()
    } else {
        format!("{v:?}")
    }
}

// ─────────────────────────────────── line parser ──────────────────────────────

/// Parse a per-second interval line into a `{timestamp, throughput, jitter, packetLoss}` sample.
///
/// Accepted formats:
///   Client/server sender row:  `0.00-1.00 s  59.60 MB  500.00 Mbits/sec  52051`
///   Client/server receiver row: `0.00-1.00 s  59.60 MB  500.00 Mbits/sec  0.009 ms  0/52106 (0%)`
///
/// Summary rows (last token = "sender"/"receiver") are rejected.
pub fn parse_interval_line(line: &str) -> Option<serde_json::Value> {
    let trimmed = line.trim();
    // Skip separator / header / iperf3-style ID lines
    if trimmed.starts_with('-') || trimmed.starts_with('[') { return None; }

    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    if parts.len() < 7 { return None; }

    // Structural validation: N.NN-N.NN s ... MB ... Mbits/sec ...
    if parts[1] != "s" || parts[3] != "MB" || parts[5] != "Mbits/sec" { return None; }
    if !parts[0].contains('-') { return None; }

    // Skip summary lines
    let last = *parts.last().unwrap();
    if last == "sender" || last == "receiver" { return None; }

    let bitrate_mbps: f64 = parts[4].parse().ok()?;

    // Receiver row has jitter at [6] and loss at [9]: `... 0.009 ms  0/52106 (0%)`
    let (jitter_ms, loss_pct) = if parts.len() >= 10 && parts[7] == "ms" {
        let jitter: f64 = parts[6].parse().ok()?;
        let pct_s = parts[9].trim_matches(|c: char| c == '(' || c == ')' || c == '%');
        let pct: f64   = pct_s.parse().unwrap_or(0.0);
        (jitter, pct)
    } else {
        (0.0_f64, 0.0_f64)
    };

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);

    Some(serde_json::json!({
        "timestamp":  ts,
        "throughput": bitrate_mbps,
        "jitter":     jitter_ms,
        "packetLoss": loss_pct,
    }))
}

// ─────────────────────────────────── spawner ──────────────────────────────────

/// Spawn `cmd`, piping stdout and stderr into `output_buf` (ring buffer).
///
/// If `result_store` is `Some`, every parsed per-second interval line is
/// appended there; the store is **cleared** on each call so it holds only
/// the current session's samples.
pub async fn spawn_and_capture(
    mut cmd:      Command,
    output_buf:   Arc<Mutex<VecDeque<String>>>,
    result_store: Option<Arc<Mutex<Vec<serde_json::Value>>>>,
    source:       &'static str,
) -> std::io::Result<Child> {
    use std::process::Stdio;
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    {
        let std_cmd = cmd.as_std();
        let args: Vec<_> = std_cmd.get_args().collect();
        log::info!("[{source}] run: {:?} {}",
            std_cmd.get_program(),
            args.iter().map(|a| a.to_string_lossy()).collect::<Vec<_>>().join(" "),
        );
        if let Some(cwd) = std_cmd.get_current_dir() {
            log::info!("[{source}] cwd: {}", cwd.display());
        }
    }

    // Clear previous session samples before starting the new one.
    if let Some(ref store) = result_store {
        store.lock().await.clear();
    }

    let mut child = cmd.spawn()?;
    log::info!("[{source}] started pid={:?}", child.id());

    // stdout → parse interval samples + ring buffer
    if let Some(stdout) = child.stdout.take() {
        let buf = Arc::clone(&output_buf);
        let res = result_store.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                log::debug!("[{source}] stdout: {line}");
                if let Some(ref store) = res {
                    if let Some(sample) = parse_interval_line(&line) {
                        store.lock().await.push(sample);
                    }
                }
                let mut lock = buf.lock().await;
                if lock.len() >= OUTPUT_CAP { lock.pop_front(); }
                lock.push_back(line);
            }
            log::info!("[{source}] stdout stream ended");
        });
    }

    // stderr → always visible at INFO
    if let Some(stderr) = child.stderr.take() {
        let buf = Arc::clone(&output_buf);
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                log::info!("[{source}] stderr: {line}");
                let mut lock = buf.lock().await;
                if lock.len() >= OUTPUT_CAP { lock.pop_front(); }
                lock.push_back(line);
            }
        });
    }

    Ok(child)
}
