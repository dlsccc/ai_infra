from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_bench.metrics.server_metrics import (  # noqa: E402
    parse_prometheus_metrics,
    summarize_metric_delta,
)
from scripts.benchmark.run_backend_baseline import main as run_backend_main  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--metrics-url")
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / config["output_dir"] / args.variant
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    variant_config = _filter_config_to_variant(config, args.variant)
    variant_config["output_dir"] = str(output_dir.relative_to(ROOT))

    if args.warmup_requests > 0:
        warmup_config = _make_warmup_config(variant_config, args.warmup_requests)
        warmup_config["output_dir"] = str((output_dir / "warmup").relative_to(ROOT))
        _run_with_temp_config(warmup_config)

    before_raw = _fetch_metrics(args.metrics_url, output_dir, "before") if args.metrics_url else ""
    if before_raw:
        (output_dir / "metrics_before.prom").write_text(before_raw, encoding="utf-8")

    _run_with_temp_config(variant_config)

    after_raw = _fetch_metrics(args.metrics_url, output_dir, "after") if args.metrics_url else ""
    if after_raw:
        (output_dir / "metrics_after.prom").write_text(after_raw, encoding="utf-8")
        before = parse_prometheus_metrics(before_raw)
        after = parse_prometheus_metrics(after_raw)
        summary = summarize_metric_delta(before, after)
        (output_dir / "metrics_delta.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )

    print(f"Wrote variant run to {output_dir}")


def _filter_config_to_variant(config: dict[str, Any], variant: str) -> dict[str, Any]:
    filtered = dict(config)
    workloads = []
    for workload in config.get("workloads", []):
        item = dict(workload)
        if item.get("type") in {"context_compiler_ablation", "context_compiler_realistic_ablation"}:
            item["variants"] = [variant]
        workloads.append(item)
    filtered["workloads"] = workloads
    return filtered


def _make_warmup_config(config: dict[str, Any], warmup_requests: int) -> dict[str, Any]:
    warmup = dict(config)
    warmup["repeat"] = 1
    warmup_workloads = []
    remaining = warmup_requests
    for workload in config.get("workloads", []):
        item = dict(workload)
        original_count = int(item.get("count", 1))
        count = max(1, min(original_count, remaining))
        item["count"] = count
        warmup_workloads.append(item)
        remaining -= count
        if remaining <= 0:
            break
    warmup["workloads"] = warmup_workloads
    return warmup


def _run_with_temp_config(config: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)
        temp_path = Path(handle.name)

    old_argv = sys.argv[:]
    try:
        sys.argv = ["run_backend_baseline.py", "--config", str(temp_path)]
        run_backend_main()
    finally:
        sys.argv = old_argv
        temp_path.unlink(missing_ok=True)


def _fetch_metrics(metrics_url: str | None, output_dir: Path, phase: str) -> str:
    if not metrics_url:
        return ""
    try:
        response = httpx.get(metrics_url, timeout=10.0)
        response.raise_for_status()
        return response.text
    except httpx.HTTPError as exc:
        message = (
            f"Metrics fetch failed during {phase}: {metrics_url}\n"
            f"{type(exc).__name__}: {exc}\n"
            "Continuing without backend metrics. Do not report actual cache hit rate for this run.\n"
        )
        (output_dir / f"metrics_{phase}_error.txt").write_text(message, encoding="utf-8")
        print(f"WARNING: {message.strip()}")
        return ""


if __name__ == "__main__":
    main()
