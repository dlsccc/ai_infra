from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_bench.metrics.server_metrics import (  # noqa: E402
    parse_prometheus_metrics,
    summarize_metric_delta,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = ROOT / run_root
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(run_root)
    output.write_text(render_markdown(rows, run_root), encoding="utf-8")
    write_csv(output.with_suffix(".csv"), rows)
    print(f"Wrote summary to {output}")
    print(f"Wrote CSV to {output.with_suffix('.csv')}")


def collect_rows(run_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for results_path in sorted(run_root.glob("*/*/results.json")):
        if "warmup" in results_path.parts:
            continue
        condition = results_path.parent.parent.name
        variant = results_path.parent.name
        payload = json.loads(results_path.read_text(encoding="utf-8"))
        requests = payload.get("requests", [])
        metrics_delta = _load_metrics_delta(results_path.parent)

        rows.append(
            {
                "condition": condition,
                "variant": variant,
                "request_count": len(requests),
                "mean_input_tokens": _mean(requests, "input_tokens"),
                "mean_output_tokens": _mean(requests, "output_tokens"),
                "mean_ttft_ms": _mean(requests, "ttft_ms"),
                "p95_ttft_ms": _p95(requests, "ttft_ms"),
                "mean_latency_ms": _mean(requests, "total_latency_ms"),
                "p95_latency_ms": _p95(requests, "total_latency_ms"),
                "mean_tpot_ms": _mean(requests, "tpot_ms"),
                "vllm_prefix_cache_hits_delta": metrics_delta.get("vllm_prefix_cache_hits_delta"),
                "vllm_prefix_cache_queries_delta": metrics_delta.get("vllm_prefix_cache_queries_delta"),
                "vllm_prefix_cache_hit_rate_delta": metrics_delta.get("vllm_prefix_cache_hit_rate_delta"),
                "sglang_cache_hit_rate": metrics_delta.get("sglang_cache_hit_rate"),
                "sglang_cached_tokens_delta": metrics_delta.get("sglang_cached_tokens_delta"),
                "sglang_prompt_tokens_delta": metrics_delta.get("sglang_prompt_tokens_delta"),
                "sglang_cached_token_ratio_delta": metrics_delta.get("sglang_cached_token_ratio_delta"),
            }
        )
    return rows


def _load_metrics_delta(run_dir: Path) -> dict[str, Any]:
    before = run_dir / "metrics_before.prom"
    after = run_dir / "metrics_after.prom"
    if before.exists() and after.exists():
        return summarize_metric_delta(
            parse_prometheus_metrics(before.read_text(encoding="utf-8")),
            parse_prometheus_metrics(after.read_text(encoding="utf-8")),
        )
    delta_path = run_dir / "metrics_delta.json"
    if delta_path.exists():
        return json.loads(delta_path.read_text(encoding="utf-8"))
    return {}


def render_markdown(rows: list[dict[str, Any]], run_root: Path) -> str:
    lines = [
        "# Context Compiler Run Summary",
        "",
        f"Run root: `{run_root}`",
        "",
        "| condition | variant | req | input tok | out tok | TTFT ms | p95 TTFT | latency ms | p95 latency | TPOT ms | backend cache ratio | source |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {condition} | {variant} | {request_count} | {mean_input_tokens:.1f} | "
            "{mean_output_tokens:.1f} | {mean_ttft_ms:.2f} | {p95_ttft_ms:.2f} | "
            "{mean_latency_ms:.2f} | {p95_latency_ms:.2f} | {mean_tpot_ms:.2f} | {hit_rate} | {hit_source} |".format(
                **row,
                hit_rate=_fmt_optional_ratio(_cache_ratio(row)[0]),
                hit_source=_cache_ratio(row)[1],
            )
        )
    lines.extend(
        [
            "",
            "说明：vLLM 使用 prefix_cache_hits_total / prefix_cache_queries_total 的差值计算 token-level hit ratio。",
            "SGLang 优先显示 cached_tokens_total / prompt_tokens_total 的差值比例；sglang:cache_hit_rate gauge 在本批实验中一直为 0，不作为主指标。",
            "",
        ]
    )
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else 0.0


def _p95(rows: list[dict[str, Any]], key: str) -> float:
    values = sorted(float(row[key]) for row in rows if row.get(key) is not None)
    if not values:
        return 0.0
    idx = min(len(values) - 1, int(len(values) * 0.95))
    return values[idx]


def _fmt_optional_ratio(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def _cache_ratio(row: dict[str, Any]) -> tuple[Any, str]:
    if row.get("vllm_prefix_cache_hit_rate_delta") is not None:
        return row.get("vllm_prefix_cache_hit_rate_delta"), "vLLM hits/queries"
    if row.get("sglang_cached_token_ratio_delta") is not None:
        return row.get("sglang_cached_token_ratio_delta"), "SGLang cached/prompt"
    if row.get("sglang_cache_hit_rate") is not None:
        return row.get("sglang_cache_hit_rate"), "SGLang gauge"
    return None, ""


if __name__ == "__main__":
    main()
