from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_bench.analysis.cacheability_metrics import (  # noqa: E402
    compute_cacheability_metrics,
    metrics_to_dict,
)
from agent_bench.optimizations.context_compiler import ContextSegment  # noqa: E402
from agent_bench.workloads.generator import generate_workloads  # noqa: E402
from agent_bench.workloads.token_utils import estimate_tokens  # noqa: E402


class EstimateTokenCounter:
    def count_prompt_tokens(self, prompt: str) -> int:
        return estimate_tokens(prompt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    output_dir = Path(args.output_dir) if args.output_dir else ROOT / config["output_dir"]
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = analyze_config(config)
    summary = summarize_rows(rows)
    results_path = output_dir / "results.json"
    latency_summary = summarize_results_by_variant(results_path) if results_path.exists() else []

    (output_dir / "context_compiler_metrics.json").write_text(
        json.dumps(
            {"rows": rows, "summary": summary, "latency_summary": latency_summary},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (output_dir / "context_compiler_summary.md").write_text(
        render_markdown(summary, latency_summary),
        encoding="utf-8",
    )
    print(f"Wrote context compiler metrics to {output_dir}")


def analyze_config(config: dict[str, Any]) -> list[dict[str, Any]]:
    token_counter = EstimateTokenCounter()
    rows: list[dict[str, Any]] = []
    previous_by_variant: dict[tuple[str, str], str] = {}

    for trace in generate_workloads(config):
        for request in trace.requests:
            variant = str(request.metadata.get("variant", "unknown"))
            trace_id = str(request.metadata.get("trace_id", "unknown"))
            key = (trace_id, variant)
            segments = _segments_from_metadata(request.metadata.get("context_segments", []))
            metrics = compute_cacheability_metrics(
                request.prompt,
                segments,
                token_counter,
                previous_prompt=previous_by_variant.get(key),
            )
            row = {
                "request_id": request.request_id,
                "trace_id": trace_id,
                "variant": variant,
                "turn": request.metadata.get("turn", 0),
                **metrics_to_dict(metrics),
                "prefix_overlap_ratio": request.metadata.get("prefix_overlap_ratio", 0.0),
                "segment_count": len(segments),
                "compiler": request.metadata.get("compiler_metadata", {}).get("compiler", "none"),
            }
            rows.append(row)
            previous_by_variant[key] = request.prompt
    return rows


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)

    summary: list[dict[str, Any]] = []
    for variant, variant_rows in sorted(grouped.items()):
        summary.append(
            {
                "variant": variant,
                "request_count": len(variant_rows),
                "mean_input_tokens": _mean(variant_rows, "input_tokens"),
                "mean_prefix_stability_ratio": _mean(variant_rows, "prefix_stability_ratio"),
                "mean_dynamic_pollution_ratio": _mean(variant_rows, "dynamic_pollution_ratio"),
                "mean_cache_reuse_opportunity": _mean(variant_rows, "cache_reuse_opportunity"),
                "mean_prefix_overlap_ratio": _mean(variant_rows, "prefix_overlap_ratio"),
            }
        )
    return summary


def summarize_results_by_variant(results_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in payload.get("requests", []):
        variant = str(row.get("metadata", {}).get("variant", "unknown"))
        grouped[variant].append(row)

    summary: list[dict[str, Any]] = []
    for variant, rows in sorted(grouped.items()):
        summary.append(
            {
                "variant": variant,
                "request_count": len(rows),
                "mean_input_tokens": _mean(rows, "input_tokens"),
                "mean_ttft_ms": _mean(rows, "ttft_ms"),
                "mean_latency_ms": _mean(rows, "total_latency_ms"),
            }
        )
    return summary


def render_markdown(
    summary: list[dict[str, Any]],
    latency_summary: list[dict[str, Any]],
) -> str:
    lines = [
        "# Context Compiler MVP Summary",
        "",
        "| variant | requests | input tokens | PSR | DPR | CRO | prefix overlap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            "| {variant} | {request_count} | {mean_input_tokens:.1f} | "
            "{mean_prefix_stability_ratio:.3f} | {mean_dynamic_pollution_ratio:.3f} | "
            "{mean_cache_reuse_opportunity:.3f} | {mean_prefix_overlap_ratio:.3f} |".format(
                **item
            )
        )
    lines.append("")
    lines.append("PSR = Prefix Stability Ratio; DPR = Dynamic Pollution Ratio; CRO = Cache Reuse Opportunity.")
    if latency_summary:
        lines.extend(
            [
                "",
                "## Latency Summary",
                "",
                "| variant | requests | backend input tokens | TTFT ms | latency ms |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for item in latency_summary:
            lines.append(
                "| {variant} | {request_count} | {mean_input_tokens:.1f} | "
                "{mean_ttft_ms:.2f} | {mean_latency_ms:.2f} |".format(**item)
            )
    return "\n".join(lines) + "\n"


def _segments_from_metadata(items: list[dict[str, Any]]) -> list[ContextSegment]:
    segments: list[ContextSegment] = []
    for item in items:
        segments.append(
            ContextSegment(
                name=str(item["name"]),
                text=str(item.get("text", "")),
                stability=str(item["stability"]),
                segment_type=str(item["segment_type"]),
                reuse_scope=str(item.get("reuse_scope", "request")),
                must_preserve=tuple(item.get("must_preserve", [])),
                source_pointer=item.get("source_pointer"),
            )
        )
    return segments


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.fmean(values) if values else 0.0


if __name__ == "__main__":
    main()
