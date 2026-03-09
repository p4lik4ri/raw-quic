use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;

use serde::Serialize;
use tokio::process::Child;
use tokio::sync::Mutex;

pub const OUTPUT_CAP: usize = 1000;

// ─────────────────────────────────── process state ────────────────────────────

#[derive(Debug, Serialize)]
pub struct ProcessStatus {
    pub running: bool,
    pub pid:     Option<u32>,
    /// Last up-to OUTPUT_CAP lines of stdout+stderr combined.
    pub output:  Vec<String>,
}

pub struct ProcessState {
    pub child:  Option<Child>,
    pub pid:    Option<u32>,
    pub output: Arc<Mutex<VecDeque<String>>>,
}

impl ProcessState {
    pub fn new() -> Self {
        Self {
            child:  None,
            pid:    None,
            output: Arc::new(Mutex::new(VecDeque::new())),
        }
    }

    /// Returns true if the child is still alive (polls without blocking).
    pub fn is_running(&mut self) -> bool {
        match &mut self.child {
            None => false,
            Some(child) => match child.try_wait() {
                Ok(Some(_)) => {
                    self.child = None;
                    self.pid   = None;
                    false
                }
                Ok(None) => true,
                Err(_)   => false,
            },
        }
    }

    pub async fn status_snapshot(&mut self) -> ProcessStatus {
        let running = self.is_running();
        let output  = self.output.lock().await;
        ProcessStatus {
            running,
            pid:    self.pid,
            output: output.iter().cloned().collect(),
        }
    }

    pub async fn stop(&mut self) -> Result<(), String> {
        match &mut self.child {
            None        => Err("not running".into()),
            Some(child) => {
                child.kill().await.map_err(|e| e.to_string())?;
                self.child = None;
                self.pid   = None;
                Ok(())
            }
        }
    }
}

// ─────────────────────────────────── app state ────────────────────────────────

pub struct AppState {
    pub server:  Mutex<ProcessState>,
    pub client:  Mutex<ProcessState>,
    pub bin_dir: PathBuf,
    /// Per-second interval samples from the most recent client session.
    /// Cleared automatically on each new /client/start call.
    pub last_client: Arc<Mutex<Vec<serde_json::Value>>>,
}
