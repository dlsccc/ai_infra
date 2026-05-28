from __future__ import annotations

from collections import defaultdict
from typing import Any, Protocol

from agent_bench.workloads.generator import WorkloadTrace, generate_workloads


class PromptTokenCounter(Protocol):
    def count_prompt_tokens(self, prompt: str) -> int:
        """Count prompt tokens."""


def inspect_workloads(
    config: dict[str, Any],
    token_counter: PromptTokenCounter,
) -> dict[str, Any]:
    traces = generate_workloads(config)
    request_rows: list[dict[str, Any]] = []
    workload_groups: dict[str, list[int]] = defaultdict(list)

    for trace in traces:
        _append_trace_rows(trace, token_counter, request_rows, workload_groups)

    workload_summary = []
    for workload_type, token_counts in sorted(workload_groups.items()):
        workload_summary.append(
            {
                "workload_type": workload_type,
                "request_count": len(token_counts),
                "min_input_tokens": min(token_counts),
                "max_input_tokens": max(token_counts),
                "mean_input_tokens": sum(token_counts) / len(token_counts),
            }
        )

    return {
        "request_rows": request_rows,
        "workload_summary": workload_summary,
        "max_input_tokens": max((row["input_tokens"] for row in request_rows), default=0),
        "request_count": len(request_rows),
        "trace_count": len(traces),
    }


def _append_trace_rows(
    trace: WorkloadTrace,
    token_counter: PromptTokenCounter,
    request_rows: list[dict[str, Any]],
    workload_groups: dict[str, list[int]],
) -> None:
    for request in trace.requests:
        input_tokens = token_counter.count_prompt_tokens(request.prompt)
        row = {
            "request_id": request.request_id,
            "trace_id": request.metadata.get("trace_id"),
            "workload_type": request.metadata.get("workload_type", trace.workload_type),
            "turn": request.metadata.get("turn", 0),
            "expected_output_tokens": request.metadata.get("expected_output_tokens"),
            "prefix_overlap_tokens": request.metadata.get("prefix_overlap_tokens"),
            "prefix_overlap_ratio": request.metadata.get("prefix_overlap_ratio"),
            "input_tokens": input_tokens,
        }
        request_rows.append(row)
        workload_groups[row["workload_type"]].append(input_tokens)
