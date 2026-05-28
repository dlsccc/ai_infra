from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_bench.backends.base import GenerationRequest
from agent_bench.workloads.schemas import make_tools
from agent_bench.workloads.token_utils import common_prefix_tokens, filler_tokens


@dataclass(frozen=True)
class WorkloadTrace:
    workload_type: str
    requests: list[GenerationRequest]
    metadata: dict[str, Any]


def generate_workloads(config: dict[str, Any]) -> list[WorkloadTrace]:
    traces: list[WorkloadTrace] = []
    for workload_group_idx, workload_config in enumerate(config.get("workloads", [])):
        workload_type = workload_config["type"]
        count = int(workload_config.get("count", 1))
        for idx in range(count):
            trace_id = f"{workload_type}_cfg{workload_group_idx:02d}_{idx:03d}"
            if workload_type == "plain_chat":
                traces.append(_plain_chat(trace_id, workload_config))
            else:
                traces.append(_agent_trace(trace_id, workload_type, workload_config))
    return traces


def _plain_chat(trace_id: str, config: dict[str, Any]) -> WorkloadTrace:
    prompt_tokens = int(config.get("prompt_tokens", 128))
    output_tokens = int(config.get("output_tokens", 64))
    prompt = (
        "You are a helpful assistant.\n"
        f"User request: {filler_tokens('plain_user', prompt_tokens)}"
    )
    request = GenerationRequest(
        request_id=f"{trace_id}_turn_00",
        prompt=prompt,
        metadata={
            "trace_id": trace_id,
            "turn": 0,
            "workload_type": "plain_chat",
            "expected_output_tokens": output_tokens,
            "prefix_overlap_tokens": 0,
            "prefix_overlap_ratio": 0.0,
        },
    )
    return WorkloadTrace("plain_chat", [request], {"trace_id": trace_id})


def _agent_trace(trace_id: str, workload_type: str, config: dict[str, Any]) -> WorkloadTrace:
    system_tokens = int(config.get("system_tokens", 256))
    tool_count = int(config.get("tool_count", 5))
    tool_tokens = int(config.get("tool_tokens", 120))
    turns = int(config.get("turns", 3))
    observation_tokens = int(config.get("observation_tokens", 128))
    output_tokens = int(config.get("output_tokens", 64))

    system_prompt = "System:\n" + filler_tokens("system", system_tokens)
    tools = "\n".join(tool.render() for tool in make_tools(tool_count, tool_tokens))
    stable_prefix = f"{system_prompt}\n\nTools:\n{tools}\n\n"

    requests: list[GenerationRequest] = []
    history: list[str] = []
    previous_prompt = ""
    for turn in range(turns):
        user = f"User turn {turn}: {filler_tokens(f'user_{turn}', 48)}"
        if turn > 0:
            observation = (
                f"Observation turn {turn - 1}: "
                + filler_tokens(f"observation_{turn - 1}", observation_tokens)
            )
            history.append(observation)

        prompt = stable_prefix + "\n".join(history + [user]) + "\nAssistant:"
        prefix_tokens = common_prefix_tokens(previous_prompt, prompt) if previous_prompt else 0
        total_prompt_parts = max(1, len(prompt.split()))
        requests.append(
            GenerationRequest(
                request_id=f"{trace_id}_turn_{turn:02d}",
                prompt=prompt,
                metadata={
                    "trace_id": trace_id,
                    "turn": turn,
                    "workload_type": workload_type,
                    "expected_output_tokens": output_tokens,
                    "prefix_overlap_tokens": prefix_tokens,
                    "prefix_overlap_ratio": prefix_tokens / total_prompt_parts,
                    "tool_count": tool_count,
                    "observation_tokens": observation_tokens,
                },
            )
        )
        history.append(
            'Assistant tool call: {"name": "tool_00", "arguments": {"query": "example"}}'
        )
        previous_prompt = prompt

    return WorkloadTrace(workload_type, requests, {"trace_id": trace_id, "turns": turns})
