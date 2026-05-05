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
    pub fn new(api_key: &str, project: &str) -> Option<Self> {
        wlog!("initializing for project={project}");

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
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let run_name = format!("tquic-linucb-{ts}");
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
        wlog!("step 2/4 OK → {run_url}");

        Some(WandbLogger {
            client,
            api_key: api_key.to_string(),
            entity,
            project: project.to_string(),
            run_name,
            run_url,
        })
    }

    /// Upload per-second JSONL lines to this run's history file.
    /// Each element must be a valid JSON object string.
    pub fn upload_history(&self, jsonl_lines: &[String]) -> bool {
        if jsonl_lines.is_empty() {
            return true;
        }
        let content = jsonl_lines.join("\n");
        wlog!(
            "step 3/4: requesting upload URL ({} steps, {} bytes) ...",
            jsonl_lines.len(), content.len()
        );

        // ── Step 3: get presigned URL via createRunFiles ──────────────────────
        let raw = match self.client
            .post(Self::GRAPHQL)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&json!({
                "query": "mutation CreateRunFiles(\
                    $entityName:String!,$projectName:String!,$runName:String!,$files:[String]!) \
                    { createRunFiles(entityName:$entityName,projectName:$projectName,\
                        runName:$runName,files:$files) \
                      { uploadHeaders files { name url(upload:true) } } }",
                "variables": {
                    "entityName":  self.entity,
                    "projectName": self.project,
                    "runName":     self.run_name,
                    "files":       ["wandb-history.jsonl"]
                }
            }))
            .send()
        {
            Ok(r) => r,
            Err(e) => { wlog!("ERROR createRunFiles network: {e}"); return false; }
        };
        let body = raw.text().unwrap_or_default();
        wlog!("  createRunFiles: {body}");
        let resp: Value = match serde_json::from_str(&body) {
            Ok(v) => v,
            Err(e) => { wlog!("ERROR parse createRunFiles: {e}"); return false; }
        };
        if let Some(errs) = resp["errors"].as_array() {
            if !errs.is_empty() { wlog!("ERROR createRunFiles errors: {errs:?}"); return false; }
        }
        let upload_url = match resp["data"]["createRunFiles"]["files"]
            .as_array()
            .and_then(|a| a.first())
            .and_then(|f| f["url"].as_str())
        {
            Some(u) => u.to_string(),
            None => { wlog!("ERROR no upload url. Full: {resp}"); return false; }
        };
        let extra_headers: Vec<(String, String)> = resp["data"]["createRunFiles"]["uploadHeaders"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|h| h.as_str())
                    .filter_map(|s| s.split_once(':')
                        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string())))
                    .collect()
            })
            .unwrap_or_default();

        // ── Step 4: PUT to presigned S3 URL ───────────────────────────────────
        wlog!("step 4/4: uploading to S3 ...");
        let mut req = self.client
            .put(&upload_url)
            .header("Content-Type", "application/octet-stream");
        for (k, v) in &extra_headers {
            req = req.header(k, v);
        }
        match req.body(content).send() {
            Ok(r) if r.status().is_success() => {
                wlog!("SUCCESS: {} steps → {}", jsonl_lines.len(), self.run_url);
                true
            }
            Ok(r) => {
                let s = r.status();
                let b = r.text().unwrap_or_default();
                wlog!("ERROR S3 PUT HTTP {s}: {b}");
                false
            }
            Err(e) => { wlog!("ERROR S3 PUT network: {e}"); false }
        }
    }
}
