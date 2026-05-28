from __future__ import annotations

import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_bench.backends.base import GenerationResult


def summarize_results(results: list[GenerationResult]) -> dict[str, Any]:
    latencies = [item.total_latency_ms for item in results]
    ttfts = [item.ttft_ms for item in results if item.ttft_ms is not None]
    decode_latencies = [item.decode_latency_ms for item in results if item.decode_latency_ms is not None]
    tpot_values = [item.tpot_ms for item in results if item.tpot_ms is not None]
    output_tokens = [item.output_tokens for item in results]
    total_latency_s = sum(latencies) / 1000.0
    total_output_tokens = sum(output_tokens)

    return {
        "request_count": len(results),
        "total_output_tokens": total_output_tokens,
        "mean_latency_ms": _mean(latencies),
        "p50_latency_ms": _quantile(latencies, 0.50),
        "p95_latency_ms": _quantile(latencies, 0.95),
        "mean_ttft_ms": _mean(ttfts) if ttfts else None,
        "p95_ttft_ms": _quantile(ttfts, 0.95) if ttfts else None,
        "mean_decode_latency_ms": _mean(decode_latencies) if decode_latencies else None,
        "p95_decode_latency_ms": _quantile(decode_latencies, 0.95) if decode_latencies else None,
        "mean_tpot_ms": _mean(tpot_values) if tpot_values else None,
        "p95_tpot_ms": _quantile(tpot_values, 0.95) if tpot_values else None,
        "tokens_per_second": total_output_tokens / total_latency_s if total_latency_s > 0 else None,
        "requests_per_second": len(results) / total_latency_s if total_latency_s > 0 else None,
    }


def write_run(
    output_dir: Path,
    config: dict[str, Any],
    results: list[GenerationResult],
    extra: dict[str, Any] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": config,
        "summary_metrics": summarize_results(results),
        "requests": [_result_to_dict(item) for item in results],
        "extra": extra or {},
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _result_to_dict(result: GenerationResult) -> dict[str, Any]:
    return {
        "request_id": result.request_id,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "ttft_ms": result.ttft_ms,
        "total_latency_ms": result.total_latency_ms,
        "decode_latency_ms": result.decode_latency_ms,
        "tpot_ms": result.tpot_ms,
        "metadata": result.metadata,
    }


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[index]
