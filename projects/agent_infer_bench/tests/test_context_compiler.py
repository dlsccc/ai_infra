from __future__ import annotations

from agent_bench.analysis.cacheability_metrics import compute_cacheability_metrics
from agent_bench.optimizations.context_compiler import ContextSegment, compile_context
from agent_bench.workloads.token_utils import estimate_tokens


class EstimateTokenCounter:
    def count_prompt_tokens(self, prompt: str) -> int:
        return estimate_tokens(prompt)


def test_context_compiler_moves_static_segments_before_ephemeral_segments() -> None:
    segments = [
        ContextSegment("runtime", "timestamp=dynamic", "ephemeral", "runtime"),
        ContextSegment("query", "current question", "turn_dynamic", "current_query"),
        ContextSegment("tools", "tool schema", "static", "tool_schema"),
        ContextSegment("system", "system prompt", "static", "system"),
    ]

    compiled = compile_context(segments)

    assert compiled.prompt.index("[SYSTEM:system]") < compiled.prompt.index("[RUNTIME:runtime]")
    assert compiled.prompt.index("[TOOL_SCHEMA:tools]") < compiled.prompt.index("[CURRENT_QUERY:query]")
    assert compiled.metadata["compiler"] == "context_compiler_mvp"


def test_observation_compression_is_recoverable() -> None:
    segments = [
        ContextSegment("system", "system prompt", "static", "system"),
        ContextSegment(
            "obs",
            " ".join(f"obs_{idx}" for idx in range(100)),
            "turn_dynamic",
            "observation",
            source_pointer="obs://trace/0",
        ),
    ]

    compiled = compile_context(segments, compress_observations=True, max_observation_words=20)

    assert "recoverable at obs://trace/0" in compiled.prompt
    assert len(compiled.prompt.split()) < 100


def test_cacheability_metrics_penalize_dynamic_prefix_pollution() -> None:
    token_counter = EstimateTokenCounter()
    good_segments = [
        ContextSegment("system", "stable stable stable", "static", "system"),
        ContextSegment("runtime", "dynamic", "ephemeral", "runtime"),
    ]
    bad_segments = [
        ContextSegment("runtime", "dynamic", "ephemeral", "runtime"),
        ContextSegment("system", "stable stable stable", "static", "system"),
    ]

    good = compute_cacheability_metrics("stable stable stable dynamic", good_segments, token_counter)
    bad = compute_cacheability_metrics("dynamic stable stable stable", bad_segments, token_counter)

    assert good.prefix_stability_ratio > bad.prefix_stability_ratio
    assert bad.dynamic_pollution_ratio > good.dynamic_pollution_ratio

