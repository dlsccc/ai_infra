from agent_bench.workloads.generator import generate_workloads


def test_generate_mock_workloads() -> None:
    config = {
        "workloads": [
            {"type": "plain_chat", "count": 1, "prompt_tokens": 64},
            {
                "type": "multi_tool_serial",
                "count": 1,
                "system_tokens": 64,
                "tool_count": 2,
                "tool_tokens": 16,
                "turns": 3,
                "observation_tokens": 32,
            },
        ]
    }
    traces = generate_workloads(config)
    assert len(traces) == 2
    assert len(traces[0].requests) == 1
    assert len(traces[1].requests) == 3
    assert traces[0].metadata["trace_id"] == "plain_chat_cfg00_000"
    assert traces[1].metadata["trace_id"] == "multi_tool_serial_cfg01_000"
