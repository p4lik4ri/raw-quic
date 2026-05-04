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

//! Weights & Biases (wandb) logger for LinUCB per-second metrics.
//!
//! Flow:
//!  1. `WandbLogger::new()` — resolve entity from API key, create a new run.
//!  2. `WandbLogger::upload_history()` — POST the JSONL metric history once at
//!     connection close (batch upload, no streaming dependency required).

use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// Always prints to stderr regardless of log level so you always see what's happening.
macro_rules! wlog {
    ($($arg:tt)*) => { eprintln!("[wandb] {}", format!($($arg)*)) };
}

/// Handles a single wandb run for one QUIC connection.
pub struct WandbLogger {
    client: Client,
    api_key: String,
    /// Wandb username / org resolved from the API key.
    entity: String,
    project: String,
    /// Run identifier (also used as the run name on wandb).
    run_name: String,
    /// Public URL for the wandb run dashboard.
    pub run_url: String,
}

impl WandbLogger {
    const GRAPHQL: &'static str = "https://api.wandb.ai/graphql";

    /// Create a new wandb run.
    ///
    /// Returns `None` with a descriptive stderr message if any step fails,
    /// so callers can treat wandb as optional and continue normally.
    pub fn new(api_key: &str, project: &str) -> Option<Self> {
        wlog!("initializing for project={project}");

        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .map_err(|e| wlog!("ERROR: failed to build HTTP client: {e}"))
            .ok()?;

        // ── Step 1: resolve entity (username) from API key ────────────────────
        wlog!("step 1/4: resolving entity from API key ...");
        let raw = match client
            .post(Self::GRAPHQL)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&json!({"query": "{viewer{entity}}"}))
            .send()
        {
            Ok(r) => r,
            Err(e) => { wlog!("ERROR: viewer query network error: {e}"); return None; }
        };
        let status = raw.status();
        let body = raw.text().unwrap_or_default();
        wlog!("  viewer response HTTP {status}: {body}");
        let resp: Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(e) => { wlog!("ERROR: could not parse viewer JSON: {e}"); return None; }
        };

        let entity = match resp["data"]["viewer"]["entity"].as_str() {
            Some(s) => s.to_string(),
            None => {
                wlog!("ERROR: entity not found in viewer response. Full: {resp}");
                return None;
            }
        };
        wlog!("  entity resolved: {entity}");

        // ── Step 2: create run via upsertBucket mutation ──────────────────────
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let run_name = format!("tquic-linucb-{ts}");
        wlog!("step 2/4: creating run \"{run_name}\" in {entity}/{project} ...");

        let raw = match client
            .post(Self::GRAPHQL)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&json!({
                "query": "mutation UpsertBucket($name: String, $project: String, $entity: String) \
                          { upsertBucket(input: {name: $name, projectName: $project, entityName: $entity}) \
                            { bucket { id name } } }",
                "variables": {
                    "name": run_name,
                    "project": project,
                    "entity": entity,
                }
            }))
            .send()
        {
            Ok(r) => r,
            Err(e) => { wlog!("ERROR: upsertBucket network error: {e}"); return None; }
        };
        let status = raw.status();
        let body = raw.text().unwrap_or_default();
        wlog!("  upsertBucket response HTTP {status}: {body}");
        let resp: Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(e) => { wlog!("ERROR: could not parse upsertBucket JSON: {e}"); return None; }
        };

        if let Some(errors) = resp["errors"].as_array() {
            if !errors.is_empty() {
                wlog!("ERROR: upsertBucket returned errors: {errors:?}");
                return None;
            }
        }

        let run_url = format!("https://wandb.ai/{entity}/{project}/runs/{run_name}");
        wlog!("step 2/4 OK: run created → {run_url}");

        Some(WandbLogger {
            client,
            api_key: api_key.to_string(),
            entity,
            project: project.to_string(),
            run_name,
            run_url,
        })
    }

    /// Upload per-second metric lines (JSONL) to the wandb run history.
    ///
    /// Each element of `jsonl_lines` must be a valid JSON object string.
    /// Returns `true` on success.
    pub fn upload_history(&self, jsonl_lines: &[String]) -> bool {
        if jsonl_lines.is_empty() {
            wlog!("upload_history: no metrics to upload, skipping");
            return true;
        }

        wlog!(
            "step 3/4: uploading {} metric steps to {} ...",
            jsonl_lines.len(),
            self.run_url
        );

        let content = jsonl_lines.join("\n");
        let md5_hex = format!("{:x}", md5::compute(content.as_bytes()));
        wlog!("  JSONL size={} bytes  md5={md5_hex}", content.len());

        // ── Step 3: request a presigned upload URL ────────────────────────────
        let file_batch_url = format!(
            "https://api.wandb.ai/files/{}/{}/{}/file_batch",
            self.entity, self.project, self.run_name
        );
        wlog!("  POST {file_batch_url}");

        let raw = match self
            .client
            .post(&file_batch_url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&json!({
                "files": {
                    "wandb-history.jsonl": {
                        "md5": md5_hex,
                        "content-type": "application/octet-stream"
                    }
                }
            }))
            .send()
        {
            Ok(r) => r,
            Err(e) => { wlog!("ERROR: file_batch network error: {e}"); return false; }
        };
        let status = raw.status();
        let body = raw.text().unwrap_or_default();
        wlog!("  file_batch response HTTP {status}: {body}");

        let resp: Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(e) => { wlog!("ERROR: could not parse file_batch JSON: {e}"); return false; }
        };

        let upload_url = match resp["files"]["wandb-history.jsonl"]["uploadUrl"].as_str() {
            Some(u) => u.to_string(),
            None => {
                wlog!("ERROR: no uploadUrl in response. Full: {resp}");
                return false;
            }
        };

        // ── Step 4: upload the JSONL content via presigned URL ────────────────
        wlog!("step 4/4: PUT presigned URL ...");
        match self
            .client
            .put(&upload_url)
            .header("Content-Type", "application/octet-stream")
            .body(content)
            .send()
        {
            Ok(r) => {
                let status = r.status();
                let body = r.text().unwrap_or_default();
                if status.is_success() {
                    wlog!(
                        "SUCCESS: {} steps uploaded → {}",
                        jsonl_lines.len(),
                        self.run_url
                    );
                    true
                } else {
                    wlog!("ERROR: presigned PUT returned HTTP {status}: {body}");
                    false
                }
            }
            Err(e) => { wlog!("ERROR: presigned PUT network error: {e}"); false }
        }
    }
}
