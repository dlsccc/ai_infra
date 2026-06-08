from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


METRIC_LINE_RE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\{?[^}]*\}?\s+([-+0-9.eE]+)$")


@dataclass(frozen=True)
class MetricSnapshot:
    raw_text: str
    values: dict[str, float]


def parse_prometheus_metrics(raw_text: str) -> MetricSnapshot:
    values: dict[str, float] = {}
    for line in raw_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = METRIC_LINE_RE.match(line)
        if not match:
            continue
        name = match.group(1)
        value = float(match.group(2))
        values[name] = values.get(name, 0.0) + value
    return MetricSnapshot(raw_text=raw_text, values=values)


def summarize_metric_delta(
    before: MetricSnapshot,
    after: MetricSnapshot,
) -> dict[str, Any]:
    delta = {
        name: after.values.get(name, 0.0) - before.values.get(name, 0.0)
        for name in sorted(set(before.values) | set(after.values))
    }

    vllm_hits = _first_present(
        delta,
        [
            "vllm:prefix_cache_hits",
            "vllm:prefix_cache_hits_total",
            "vllm_gpu_prefix_cache_hits",
            "vllm:gpu_prefix_cache_hits_total",
        ],
    )
    vllm_queries = _first_present(
        delta,
        [
            "vllm:prefix_cache_queries",
            "vllm:prefix_cache_queries_total",
            "vllm_gpu_prefix_cache_queries",
            "vllm:gpu_prefix_cache_queries_total",
        ],
    )
    vllm_hit_rate = None
    if vllm_hits is not None and vllm_queries is not None and vllm_queries > 0:
        vllm_hit_rate = vllm_hits / vllm_queries

    sglang_cache_hit_rate = _first_present(
        after.values,
        ["sglang:cache_hit_rate", "sglang_cache_hit_rate"],
    )
    sglang_cached_tokens = _first_present(delta, ["sglang:cached_tokens_total"])
    sglang_prompt_tokens = _first_present(delta, ["sglang:prompt_tokens_total"])
    sglang_cached_token_ratio = None
    if (
        sglang_cached_tokens is not None
        and sglang_prompt_tokens is not None
        and sglang_prompt_tokens > 0
    ):
        sglang_cached_token_ratio = sglang_cached_tokens / sglang_prompt_tokens

    return {
        "delta": delta,
        "vllm_prefix_cache_hits_delta": vllm_hits,
        "vllm_prefix_cache_queries_delta": vllm_queries,
        "vllm_prefix_cache_hit_rate_delta": vllm_hit_rate,
        "sglang_cache_hit_rate": sglang_cache_hit_rate,
        "sglang_cached_tokens_delta": sglang_cached_tokens,
        "sglang_prompt_tokens_delta": sglang_prompt_tokens,
        "sglang_cached_token_ratio_delta": sglang_cached_token_ratio,
    }


def _first_present(values: dict[str, float], names: list[str]) -> float | None:
    for name in names:
        if name in values:
            return values[name]
    return None
