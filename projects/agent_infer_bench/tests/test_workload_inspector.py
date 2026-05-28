from __future__ import annotations

from agent_bench.analysis.workload_inspector import inspect_workloads


class _StubTokenCounter:
    def count_prompt_tokens(self, prompt: str) -> int:
        return len(prompt.split())


def test_inspect_workloads_collects_request_rows_and_summary() -> None:
    config = {
        "workloads": [
            {"type": "plain_chat", "count": 1, "prompt_tokens": 32, "output_tokens": 16},
            {
                "type": "multi_tool_serial",
                "count": 1,
                "system_tokens": 32,
                "tool_count": 2,
                "tool_tokens": 8,
                "turns": 3,
                "observation_tokens": 24,
                "output_tokens": 16,
            },
        ]
    }

    report = inspect_workloads(config, _StubTokenCounter())

    assert report["trace_count"] == 2
    assert report["request_count"] == 4
    assert report["max_input_tokens"] > 0
    assert len(report["request_rows"]) == 4
    workload_types = {row["workload_type"] for row in report["workload_summary"]}
    assert workload_types == {"plain_chat", "multi_tool_serial"}
