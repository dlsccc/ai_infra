from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agent_bench.optimizations.context_compiler import ContextSegment
from agent_bench.workloads.token_utils import common_prefix_tokens


class TextTokenCounter(Protocol):
    def count_prompt_tokens(self, prompt: str) -> int:
        """Count prompt tokens."""


@dataclass(frozen=True)
class CacheabilityMetrics:
    input_tokens: int
    stable_prefix_tokens: int
    dynamic_prefix_tokens: int
    prefix_stability_ratio: float
    dynamic_pollution_ratio: float
    cache_reuse_opportunity: float
    canonicalization_gain: float | None


def compute_cacheability_metrics(
    prompt: str,
    segments: list[ContextSegment],
    token_counter: TextTokenCounter,
    *,
    previous_prompt: str | None = None,
    baseline_prompt: str | None = None,
) -> CacheabilityMetrics:
    input_tokens = token_counter.count_prompt_tokens(prompt)
    stable_prefix_text, dynamic_prefix_text = _prefix_text_by_stability(segments)
    stable_prefix_tokens = token_counter.count_prompt_tokens(stable_prefix_text)
    dynamic_prefix_tokens = token_counter.count_prompt_tokens(dynamic_prefix_text)

    prefix_stability_ratio = _safe_ratio(stable_prefix_tokens, input_tokens)
    dynamic_pollution_ratio = _safe_ratio(dynamic_prefix_tokens, input_tokens)

    if previous_prompt:
        lcp = common_prefix_tokens(previous_prompt, prompt)
        denom = max(1, min(len(previous_prompt.split()), len(prompt.split())))
        cache_reuse_opportunity = lcp / denom
    else:
        cache_reuse_opportunity = 0.0

    canonicalization_gain = None
    if baseline_prompt is not None:
        baseline_lcp = common_prefix_tokens(baseline_prompt, prompt)
        denom = max(1, min(len(baseline_prompt.split()), len(prompt.split())))
        canonicalization_gain = baseline_lcp / denom

    return CacheabilityMetrics(
        input_tokens=input_tokens,
        stable_prefix_tokens=stable_prefix_tokens,
        dynamic_prefix_tokens=dynamic_prefix_tokens,
        prefix_stability_ratio=prefix_stability_ratio,
        dynamic_pollution_ratio=dynamic_pollution_ratio,
        cache_reuse_opportunity=cache_reuse_opportunity,
        canonicalization_gain=canonicalization_gain,
    )


def metrics_to_dict(metrics: CacheabilityMetrics) -> dict[str, float | int | None]:
    return {
        "input_tokens": metrics.input_tokens,
        "stable_prefix_tokens": metrics.stable_prefix_tokens,
        "dynamic_prefix_tokens": metrics.dynamic_prefix_tokens,
        "prefix_stability_ratio": metrics.prefix_stability_ratio,
        "dynamic_pollution_ratio": metrics.dynamic_pollution_ratio,
        "cache_reuse_opportunity": metrics.cache_reuse_opportunity,
        "canonicalization_gain": metrics.canonicalization_gain,
    }


def _prefix_text_by_stability(segments: list[ContextSegment]) -> tuple[str, str]:
    stable_parts: list[str] = []
    dynamic_parts: list[str] = []
    seen_dynamic = False
    for segment in segments:
        if segment.stability in {"turn_dynamic", "ephemeral"}:
            seen_dynamic = True
            dynamic_parts.append(segment.text)
            continue
        if seen_dynamic:
            dynamic_parts.append(segment.text)
        else:
            stable_parts.append(segment.text)
    return "\n".join(stable_parts), "\n".join(dynamic_parts)


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator

