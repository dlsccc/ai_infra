#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping.")
    return config


def build_dry_run_metrics(config: dict) -> dict:
    batch_size = int(config.get("batch_size", 1))
    precision = str(config.get("precision", "fp32")).lower()

    precision_factor = {
        "fp32": 1.00,
        "fp16": 0.72,
        "int8": 0.55,
    }.get(precision, 1.00)

    base_latency = 12.0 * max(batch_size, 1)
    p50 = round(base_latency * precision_factor, 3)
    p95 = round(p50 * 1.18, 3)
    throughput = round((1000.0 / p50) * max(batch_size, 1), 3)
    peak_gpu_memory_gb = round(0.6 + 0.08 * max(batch_size, 1), 3)

    return {
        "mode": "dry_run",
        "p50_latency_ms": p50,
        "p95_latency_ms": p95,
        "throughput_tokens_per_s": throughput,
        "peak_gpu_memory_gb": peak_gpu_memory_gb,
    }


def write_outputs(config: dict, metrics: dict) -> Path:
    exp_name = str(config.get("experiment_name", "unnamed_experiment"))
    output_root = Path(str(config.get("output_dir", f"experiments/runs/{exp_name}")))
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = run_dir / "config_snapshot.yaml"
    metrics_json = run_dir / "metrics.json"
    summary_md = run_dir / "summary.md"

    with config_snapshot.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

    with metrics_json.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    summary = (
        "# Dry-run Benchmark Summary\n\n"
        f"- experiment: `{exp_name}`\n"
        f"- run_id: `{run_id}`\n"
        f"- mode: `{metrics['mode']}`\n"
        f"- p50_latency_ms: `{metrics['p50_latency_ms']}`\n"
        f"- p95_latency_ms: `{metrics['p95_latency_ms']}`\n"
        f"- throughput_tokens_per_s: `{metrics['throughput_tokens_per_s']}`\n"
        f"- peak_gpu_memory_gb: `{metrics['peak_gpu_memory_gb']}`\n"
    )
    summary_md.write_text(summary, encoding="utf-8")

    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Week0 dry-run benchmark runner")
    parser.add_argument("--config", required=True, type=Path, help="Path to YAML config")
    args = parser.parse_args()

    config = load_config(args.config)
    metrics = build_dry_run_metrics(config)
    run_dir = write_outputs(config, metrics)

    print(f"[OK] Dry-run benchmark finished. Output: {run_dir}")


if __name__ == "__main__":
    main()
