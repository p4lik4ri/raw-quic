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
//!
//! The wandb REST API used here:
//!  - GraphQL `{viewer{entity}}` — resolve username from API key.
//!  - GraphQL `upsertBucket` mutation — create a new run.
//!  - `POST /files/{entity}/{project}/{run}/file_batch` — obtain a presigned
//!    upload URL for `wandb-history.jsonl`.
//!  - `PUT {presigned_url}` — upload the JSONL history file.

use log::{info, warn};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
    /// Returns `None` and logs a warning if any network call fails, so callers
    /// can treat wandb as optional and continue normally.
    pub fn new(api_key: &str, project: &str) -> Option<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(15))
            .build()
            .map_err(|e| warn!("wandb: failed to create HTTP client: {e}"))
            .ok()?;

        // ── Step 1: resolve entity (username) from API key ────────────────────
        let resp: Value = client
            .post(Self::GRAPHQL)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&json!({"query": "{viewer{entity}}"}))
            .send()
            .and_then(|r| r.json())
            .map_err(|e| warn!("wandb: viewer query failed: {e}"))
            .ok()?;

        let entity = resp["data"]["viewer"]["entity"]
            .as_str()
            .map(|s| s.to_string())
            .or_else(|| {
                warn!("wandb: could not read entity from viewer response: {resp}");
                None
            })?;

        // ── Step 2: create run via upsertBucket mutation ──────────────────────
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let run_name = format!("tquic-linucb-{ts}");

        let resp: Value = client
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
            .and_then(|r| r.json())
            .map_err(|e| warn!("wandb: upsertBucket failed: {e}"))
            .ok()?;

        if resp["errors"].is_array() {
            warn!("wandb: upsertBucket returned errors: {}", resp["errors"]);
            return None;
        }

        let run_url = format!("https://wandb.ai/{entity}/{project}/runs/{run_name}");
        info!("wandb run created: {run_url}");

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
    /// Each element of `jsonl_lines` must be a valid JSON object string whose
    /// keys are metric names.  The `_step` key is used by wandb for the x-axis.
    ///
    /// Returns `true` on success.
    pub fn upload_history(&self, jsonl_lines: &[String]) -> bool {
        if jsonl_lines.is_empty() {
            return true;
        }

        let content = jsonl_lines.join("\n");
        let md5_hex = format!("{:x}", md5::compute(content.as_bytes()));

        // ── Step 3: request a presigned upload URL ────────────────────────────
        let file_batch_url = format!(
            "https://api.wandb.ai/files/{}/{}/{}/file_batch",
            self.entity, self.project, self.run_name
        );

        let resp: Value = match self
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
            .and_then(|r| r.json())
        {
            Ok(v) => v,
            Err(e) => {
                warn!("wandb: file_batch request failed: {e}");
                return false;
            }
        };

        let upload_url = match resp["files"]["wandb-history.jsonl"]["uploadUrl"].as_str() {
            Some(u) => u.to_string(),
            None => {
                warn!("wandb: no uploadUrl in file_batch response: {resp}");
                return false;
            }
        };

        // ── Step 4: upload the JSONL content via presigned URL ────────────────
        match self
            .client
            .put(&upload_url)
            .header("Content-Type", "application/octet-stream")
            .body(content)
            .send()
        {
            Ok(_) => {
                info!(
                    "wandb: {} metrics steps uploaded → {}",
                    jsonl_lines.len(),
                    self.run_url
                );
                true
            }
            Err(e) => {
                warn!("wandb: history upload failed: {e}");
                false
            }
        }
    }
}
