#!/usr/bin/env python3
"""Upload tquic LinUCB per-second metrics to Weights & Biases.

Usage
-----
    python tquic_wandb_upload.py /tmp/tquic-metrics-<timestamp>.jsonl

The script reads the JSONL file produced by tquic_client (one JSON object
per second) and logs every line as a wandb step.  Three groups of metrics
are logged, corresponding to the three plots you want in the dashboard:

  Plot 1 — UCB score decomposition
    p<N>.reward  : LinUCB reward estimate  (θᵀx)
    p<N>.bonus   : exploration bonus       (α √(xᵀA⁻¹x))

  Plot 2 — Context features (what the scheduler observed each second)
    p<N>.x_rtt_norm   : normalised RTT
    p<N>.x_cwnd_p     : CWND pressure
    p<N>.x_loss_rate  : packet loss rate
    p<N>.x_bw_norm    : normalised bandwidth
    p<N>.rtt_us       : raw smoothed RTT in microseconds
    p<N>.pct          : % of packets sent on this path

  Plot 3 — Learned θ weights (how the model evolves over time)
    p<N>.th_rtt_norm
    p<N>.th_cwnd_p
    p<N>.th_loss_rate
    p<N>.th_bw_norm
    p<N>.th_bias

Requirements
------------
    pip install wandb
"""

import json
import sys
import os
import wandb


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: tquic_wandb_upload.py <metrics.jsonl> [run_name]")
        sys.exit(1)

    metrics_path = sys.argv[1]
    run_name = sys.argv[2] if len(sys.argv) > 2 else None

    # Read all JSONL lines up-front so we know the step count.
    with open(metrics_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        print(f"ERROR: {metrics_path} is empty")
        sys.exit(1)

    rows = [json.loads(line) for line in lines]

    # Detect which path IDs are present (p0, p1, …).
    path_ids: list[int] = []
    for key in rows[0]:
        if key.startswith("p") and ".reward" in key:
            pid = int(key.split(".")[0][1:])
            if pid not in path_ids:
                path_ids.append(pid)
    path_ids.sort()
    print(f"Found {len(rows)} steps, {len(path_ids)} paths: {path_ids}")

    # ── Feature / theta names must match scheduler_linucb.rs ─────────────────
    feat_names = ["rtt_norm", "cwnd_p", "loss_rate", "bw_norm", "bias"]

    # ── Initialise wandb run ──────────────────────────────────────────────────
    run = wandb.init(
        entity="asakel64-ncsrd",
        project="quic",
        name=run_name or os.path.splitext(os.path.basename(metrics_path))[0],
        config={
            "scheduler": "LinUCB",
            "n_paths": len(path_ids),
            "n_steps": len(rows),
            "source_file": metrics_path,
        },
    )

    # ── Log each second as a wandb step ──────────────────────────────────────
    for row in rows:
        step = row.get("_step", None)
        log_dict: dict = {"t": row.get("t", step)}

        for pid in path_ids:
            prefix = f"p{pid}"

            # Plot 1: UCB score decomposition
            if f"{prefix}.reward" in row:
                log_dict[f"{prefix}.reward"] = row[f"{prefix}.reward"]
            if f"{prefix}.bonus" in row:
                log_dict[f"{prefix}.bonus"] = row[f"{prefix}.bonus"]

            # Plot 2: context features
            for fn in feat_names:
                key = f"{prefix}.x_{fn}"
                if key in row:
                    log_dict[key] = row[key]
            if f"{prefix}.rtt_us" in row:
                log_dict[f"{prefix}.rtt_us"] = row[f"{prefix}.rtt_us"]
            if f"{prefix}.pct" in row:
                log_dict[f"{prefix}.pct"] = row[f"{prefix}.pct"]

            # Plot 3: learned theta weights
            for fn in feat_names:
                key = f"{prefix}.th_{fn}"
                if key in row:
                    log_dict[key] = row[key]

        wandb.log(log_dict, step=step)

    run.finish()
    print(f"\nDone! View run at: {run.url}")


if __name__ == "__main__":
    main()
