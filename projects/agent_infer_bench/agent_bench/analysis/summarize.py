from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def make_markdown_summary(run_dir: Path) -> str:
    result_path = run_dir / "results.json"
    payload = load_result(result_path)
    metrics = payload["summary_metrics"]
    token_source_summary = payload.get("extra", {}).get("token_source_summary", {})
    lines = [
        "# Run Summary",
        "",
        f"- Experiment: `{payload['config'].get('experiment_name', 'unknown')}`",
        f"- Backend: `{payload['config'].get('backend', 'unknown')}`",
        f"- Requests: `{metrics['request_count']}`",
        f"- Mean latency: `{_fmt(metrics.get('mean_latency_ms'))} ms`",
        f"- P95 latency: `{_fmt(metrics.get('p95_latency_ms'))} ms`",
        f"- Mean TTFT: `{_fmt(metrics.get('mean_ttft_ms'))} ms`",
        f"- Tokens/s: `{_fmt(metrics.get('tokens_per_second'))}`",
        f"- Input token source(s): `{_fmt(token_source_summary.get('input'))}`",
        f"- Output token source(s): `{_fmt(token_source_summary.get('output'))}`",
        "",
    ]
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)
