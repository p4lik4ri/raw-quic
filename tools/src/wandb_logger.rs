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

//! Weights & Biases REST uploader for LinUCB per-second metrics.
//!
//! Runs entirely inside the tquic_client Rust binary — no Python, no venv.
//!
//! Upload flow:
//!   1. POST /graphql  {viewer{entity}}  → resolve username from API key
//!   2. POST /graphql  upsertBucket      → create a new run
//!   3. POST /graphql  createRunFiles    → get presigned S3 upload URL
//!   4. PUT  <presigned S3 URL>          → upload wandb-history.jsonl

use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

macro_rules! wlog {
    ($($arg:tt)*) => { eprintln!("[wandb] {}", format!($($arg)*)) };
}

pub struct WandbLogger {
    client:   Client,
    api_key:  String,
    entity:   String,
    project:  String,
    run_name: String,
    /// Public URL shown in the dashboard.
    pub run_url: String,
}

impl WandbLogger {
    const GRAPHQL: &'static str = "https://api.wandb.ai/graphql";

    /// Initialise a new wandb run.  Returns `None` on any error so callers
    /// can treat wandb as fully optional and continue normally.
    pub fn new(api_key: &str, project: &str, scheduler: &str) -> Option<Self> {
        wlog!("initializing for project={project} scheduler={scheduler}");

        let client = Client::builder()
            .timeout(Duration::from_secs(20))
            .build()
            .map_err(|e| wlog!("ERROR building HTTP client: {e}"))
            .ok()?;

        // ── Step 1: resolve entity from API key ───────────────────────────────
        wlog!("step 1/4: resolving entity ...");
        let raw = client
            .post(Self::GRAPHQL)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&json!({"query": "{viewer{entity}}"}))
            .send()
            .map_err(|e| wlog!("ERROR viewer query: {e}"))
            .ok()?;
        let body = raw.text().unwrap_or_default();
        wlog!("  viewer: {body}");
        let resp: Value = serde_json::from_str(&body)
            .map_err(|e| wlog!("ERROR parse viewer: {e}"))
            .ok()?;
        let entity = resp["data"]["viewer"]["entity"]
            .as_str()
            .map(|s| s.to_string())
            .or_else(|| { wlog!("ERROR entity not found"); None })?;
        wlog!("  entity={entity}");

        // ── Step 2: create run ────────────────────────────────────────────────
        // Use milliseconds for the timestamp to avoid duplicate-key collisions
        // when two tests finish within the same second.
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let run_name = format!("tquic-{scheduler}-{ts}");
        wlog!("step 2/4: creating run \"{run_name}\" ...");

        let raw = client
            .post(Self::GRAPHQL)
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&json!({
                "query": "mutation UpsertBucket($name:String,$project:String,$entity:String) \
                          { upsertBucket(input:{name:$name,modelName:$project,entityName:$entity}) \
                            { bucket { id name } } }",
                "variables": { "name": run_name, "project": project, "entity": entity }
            }))
            .send()
            .map_err(|e| wlog!("ERROR upsertBucket: {e}"))
            .ok()?;
        let body = raw.text().unwrap_or_default();
        wlog!("  upsertBucket: {body}");
        let resp: Value = serde_json::from_str(&body)
            .map_err(|e| wlog!("ERROR parse upsertBucket: {e}"))
            .ok()?;
        if let Some(errs) = resp["errors"].as_array() {
            if !errs.is_empty() {
                wlog!("ERROR upsertBucket errors: {errs:?}");
                return None;
            }
        }

        let run_url = format!("https://wandb.ai/{entity}/{project}/runs/{run_name}");
        wlog!("step 2/3 OK → {run_url}");

        Some(WandbLogger {
            client,
            api_key: api_key.to_string(),
            entity,
            project: project.to_string(),
            run_name,
            run_url,
        })
    }

    /// Upload per-second JSONL lines to this run's history file and mark the
    /// run as finished, using the wandb file-stream API in one POST.
    ///
    /// The file-stream endpoint is the same mechanism used by the wandb Python
    /// SDK.  Sending `complete:true, exitcode:0` is what causes wandb to
    /// finalize the run and render all charts.
    pub fn upload_history(&self, jsonl_lines: &[String]) -> bool {
        if jsonl_lines.is_empty() {
            return true;
        }
        wlog!(
            "step 3/3: streaming {} steps to wandb file-stream ...",
            jsonl_lines.len()
        );

        // The file-stream endpoint accepts a JSON body with a `files` map and
        // optional `complete` / `exitcode` fields.
        // Each content line must end with '\n'.
        let lines: Vec<String> = jsonl_lines.iter()
            .map(|l| format!("{l}\n"))
            .collect();

        let stream_url = format!(
            "https://api.wandb.ai/files/{}/{}/{}/file_stream",
            self.entity, self.project, self.run_name
        );

        let body = json!({
            "files": {
                "wandb-history.jsonl": {
                    "offset":  0,
                    "content": lines
                }
            },
            "complete": true,
            "exitcode": 0
        });

        match self.client
            .post(&stream_url)
            // File-stream uses HTTP Basic auth: username="api", password=api_key
            // (same as the wandb Python SDK — Bearer works for GraphQL but not here)
            .basic_auth("api", Some(&self.api_key))
            .json(&body)
            .send()
        {
            Ok(r) if r.status().is_success() => {
                wlog!("SUCCESS: {} steps → {}", jsonl_lines.len(), self.run_url);
                true
            }
            Ok(r) => {
                let status = r.status();
                let text = r.text().unwrap_or_default();
                wlog!("ERROR file-stream HTTP {status}: {text}");
                false
            }
            Err(e) => { wlog!("ERROR file-stream network: {e}"); false }
        }
    }
}
