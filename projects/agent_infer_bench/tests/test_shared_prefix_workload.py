from agent_bench.workloads.generator import generate_workloads


def test_generate_shared_prefix_replay_workload() -> None:
    config = {
        "workloads": [
            {
                "type": "shared_prefix_replay",
                "count": 1,
                "shared_prefix_tokens": 128,
                "suffix_tokens": 32,
                "repeats": 2,
                "output_tokens": 16,
                "variant": "changing_suffix",
            }
        ]
    }
    traces = generate_workloads(config)
    assert len(traces) == 1
    assert traces[0].workload_type == "shared_prefix_replay"
    assert len(traces[0].requests) == 2
    assert traces[0].requests[1].metadata["prefix_overlap_tokens"] > 0
