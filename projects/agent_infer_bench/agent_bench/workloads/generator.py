from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agent_bench.backends.base import GenerationRequest
from agent_bench.optimizations.context_compiler import (
    ContextSegment,
    compile_context,
    render_original_layout,
)
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
            elif workload_type == "shared_prefix_replay":
                traces.append(_shared_prefix_replay(trace_id, workload_config))
            elif workload_type == "context_compiler_ablation":
                traces.append(_context_compiler_ablation(trace_id, workload_config))
            elif workload_type == "context_compiler_realistic_ablation":
                traces.append(_context_compiler_realistic_ablation(trace_id, workload_config))
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


def _shared_prefix_replay(trace_id: str, config: dict[str, Any]) -> WorkloadTrace:
    shared_prefix_tokens = int(config.get("shared_prefix_tokens", 0))
    suffix_tokens = int(config.get("suffix_tokens", 48))
    output_tokens = int(config.get("output_tokens", 64))
    repeats = int(config.get("repeats", 2))
    variant = str(config.get("variant", "identical"))

    shared_prefix = (
        "System:\n"
        + filler_tokens("shared_prefix", shared_prefix_tokens)
        + "\n\nContext:\n"
    )

    requests: list[GenerationRequest] = []
    previous_prompt = ""
    for turn in range(repeats):
        if variant == "identical":
            suffix = filler_tokens("suffix_fixed", suffix_tokens)
        else:
            suffix = filler_tokens(f"suffix_turn_{turn}", suffix_tokens)
        prompt = (
            shared_prefix
            + f"User turn {turn}: {suffix}\n"
            + "Assistant:"
        )
        prefix_tokens = common_prefix_tokens(previous_prompt, prompt) if previous_prompt else 0
        total_prompt_parts = max(1, len(prompt.split()))
        requests.append(
            GenerationRequest(
                request_id=f"{trace_id}_turn_{turn:02d}",
                prompt=prompt,
                metadata={
                    "trace_id": trace_id,
                    "turn": turn,
                    "workload_type": "shared_prefix_replay",
                    "expected_output_tokens": output_tokens,
                    "prefix_overlap_tokens": prefix_tokens,
                    "prefix_overlap_ratio": prefix_tokens / total_prompt_parts,
                    "shared_prefix_tokens": shared_prefix_tokens,
                    "suffix_tokens": suffix_tokens,
                    "variant": variant,
                },
            )
        )
        previous_prompt = prompt

    return WorkloadTrace(
        "shared_prefix_replay",
        requests,
        {
            "trace_id": trace_id,
            "repeats": repeats,
            "shared_prefix_tokens": shared_prefix_tokens,
            "variant": variant,
        },
    )


def _context_compiler_ablation(trace_id: str, config: dict[str, Any]) -> WorkloadTrace:
    system_tokens = int(config.get("system_tokens", 256))
    tool_count = int(config.get("tool_count", 8))
    tool_tokens = int(config.get("tool_tokens", 80))
    turns = int(config.get("turns", 3))
    observation_tokens = int(config.get("observation_tokens", 256))
    output_tokens = int(config.get("output_tokens", 64))
    variants = list(
        config.get(
            "variants",
            [
                "original_bad_layout",
                "stable_tool_order",
                "dynamic_fields_last",
                "context_compiler_no_compression",
                "context_compiler_with_observation_compression",
                "truncation_baseline",
            ],
        )
    )

    tools = make_tools(tool_count, tool_tokens)
    requests: list[GenerationRequest] = []
    for variant in variants:
        previous_prompt = ""
        history: list[ContextSegment] = []
        for turn in range(turns):
            segments = _make_context_segments(
                trace_id=trace_id,
                variant=variant,
                turn=turn,
                system_tokens=system_tokens,
                tools=tools,
                history=history,
                observation_tokens=observation_tokens,
            )
            prompt, planned_segments, compiler_metadata = _render_variant_prompt(
                variant,
                segments,
                max_observation_words=int(config.get("max_observation_words", 80)),
            )
            prefix_tokens = common_prefix_tokens(previous_prompt, prompt) if previous_prompt else 0
            total_prompt_parts = max(1, len(prompt.split()))
            request_id = f"{trace_id}_{variant}_turn_{turn:02d}"
            requests.append(
                GenerationRequest(
                    request_id=request_id,
                    prompt=prompt,
                    metadata={
                        "trace_id": trace_id,
                        "turn": turn,
                        "workload_type": "context_compiler_ablation",
                        "variant": variant,
                        "expected_output_tokens": output_tokens,
                        "prefix_overlap_tokens": prefix_tokens,
                        "prefix_overlap_ratio": prefix_tokens / total_prompt_parts,
                        "tool_count": tool_count,
                        "observation_tokens": observation_tokens,
                        "context_segments": [_segment_to_metadata(segment) for segment in planned_segments],
                        "compiler_metadata": compiler_metadata,
                    },
                )
            )
            history.append(
                ContextSegment(
                    name=f"tool_call_{turn:02d}",
                    text='Assistant tool call: {"name": "tool_00", "arguments": {"query": "example"}}',
                    stability="session_dynamic",
                    segment_type="tool_call",
                    reuse_scope="session",
                )
            )
            history.append(
                ContextSegment(
                    name=f"observation_{turn:02d}",
                    text="Observation: " + filler_tokens(f"observation_{turn}", observation_tokens),
                    stability="turn_dynamic",
                    segment_type="observation",
                    reuse_scope="turn",
                    must_preserve=("error_code", "tool_name"),
                    source_pointer=f"obs://{trace_id}/{turn:02d}",
                )
            )
            previous_prompt = prompt

    return WorkloadTrace(
        "context_compiler_ablation",
        requests,
        {
            "trace_id": trace_id,
            "turns": turns,
            "variants": variants,
        },
    )


def _context_compiler_realistic_ablation(trace_id: str, config: dict[str, Any]) -> WorkloadTrace:
    scenario = str(config.get("scenario", "function_calling"))
    turns = int(config.get("turns", 4))
    observation_tokens = int(config.get("observation_tokens", 384))
    output_tokens = int(config.get("output_tokens", 96))
    variants = list(
        config.get(
            "variants",
            [
                "original_bad_layout",
                "stable_tool_order",
                "dynamic_fields_last",
                "context_compiler_no_compression",
                "context_compiler_with_observation_compression",
                "truncation_baseline",
            ],
        )
    )

    requests: list[GenerationRequest] = []
    for variant in variants:
        previous_prompt = ""
        history: list[ContextSegment] = []
        for turn in range(turns):
            segments = _make_realistic_context_segments(
                trace_id=trace_id,
                scenario=scenario,
                variant=variant,
                turn=turn,
                history=history,
                observation_tokens=observation_tokens,
            )
            prompt, planned_segments, compiler_metadata = _render_variant_prompt(
                variant,
                segments,
                max_observation_words=int(config.get("max_observation_words", 120)),
            )
            prefix_tokens = common_prefix_tokens(previous_prompt, prompt) if previous_prompt else 0
            total_prompt_parts = max(1, len(prompt.split()))
            requests.append(
                GenerationRequest(
                    request_id=f"{trace_id}_{scenario}_{variant}_turn_{turn:02d}",
                    prompt=prompt,
                    metadata={
                        "trace_id": trace_id,
                        "scenario": scenario,
                        "turn": turn,
                        "workload_type": "context_compiler_realistic_ablation",
                        "variant": variant,
                        "expected_output_tokens": output_tokens,
                        "prefix_overlap_tokens": prefix_tokens,
                        "prefix_overlap_ratio": prefix_tokens / total_prompt_parts,
                        "observation_tokens": observation_tokens,
                        "context_segments": [_segment_to_metadata(segment) for segment in planned_segments],
                        "compiler_metadata": compiler_metadata,
                    },
                )
            )
            history.extend(
                _next_realistic_history_segments(
                    trace_id=trace_id,
                    scenario=scenario,
                    turn=turn,
                    observation_tokens=observation_tokens,
                )
            )
            previous_prompt = prompt

    return WorkloadTrace(
        "context_compiler_realistic_ablation",
        requests,
        {
            "trace_id": trace_id,
            "scenario": scenario,
            "turns": turns,
            "variants": variants,
        },
    )


def _make_context_segments(
    *,
    trace_id: str,
    variant: str,
    turn: int,
    system_tokens: int,
    tools: list[Any],
    history: list[ContextSegment],
    observation_tokens: int,
) -> list[ContextSegment]:
    dynamic_header = ContextSegment(
        name="runtime_header",
        text=f"timestamp=2026-05-30T00:{turn:02d}:00Z session={trace_id} retry={turn}",
        stability="ephemeral",
        segment_type="runtime",
        reuse_scope="request",
    )
    system = ContextSegment(
        name="system",
        text="System:\n" + filler_tokens("system", system_tokens),
        stability="static",
        segment_type="system",
        reuse_scope="global",
    )
    ordered_tools = tools
    if variant == "original_bad_layout":
        shift = turn % max(1, len(tools))
        ordered_tools = tools[shift:] + tools[:shift]
    tool_text = "\n".join(
        tool.render(canonical=variant != "original_bad_layout")
        for tool in ordered_tools
    )
    tool_block = ContextSegment(
        name="tools",
        text="Tools:\n" + tool_text,
        stability="static",
        segment_type="tool_schema",
        reuse_scope="agent_type",
    )
    current_user = ContextSegment(
        name=f"user_{turn:02d}",
        text=f"User turn {turn}: " + filler_tokens(f"user_{turn}", 48),
        stability="turn_dynamic",
        segment_type="current_query",
        reuse_scope="turn",
    )

    if variant == "original_bad_layout":
        return [dynamic_header, current_user, tool_block, *history, system]
    if variant == "stable_tool_order":
        return [dynamic_header, system, tool_block, *history, current_user]
    if variant == "dynamic_fields_last":
        return [system, tool_block, *history, current_user, dynamic_header]
    if variant == "truncation_baseline":
        truncated_history = [
            _truncate_observation(segment, max_words=max(24, observation_tokens // 4))
            for segment in history
        ]
        return [system, tool_block, *truncated_history, current_user, dynamic_header]
    return [dynamic_header, current_user, tool_block, *history, system]


def _make_realistic_context_segments(
    *,
    trace_id: str,
    scenario: str,
    variant: str,
    turn: int,
    history: list[ContextSegment],
    observation_tokens: int,
) -> list[ContextSegment]:
    dynamic_header = ContextSegment(
        name="runtime_header",
        text=(
            f"request_time=2026-06-08T09:{turn:02d}:00Z\n"
            f"session_id={trace_id}\n"
            f"retry_count={turn}\n"
            f"trace_nonce={trace_id}-{turn:02d}"
        ),
        stability="ephemeral",
        segment_type="runtime",
        reuse_scope="request",
    )
    if scenario == "coding_agent":
        system = ContextSegment(
            name="system",
            text=(
                "You are a careful coding repair agent. "
                "Inspect repository evidence, call tools when needed, and propose minimal patches. "
                "Preserve file paths, failing tests, line numbers, error messages, and command exit codes."
            ),
            stability="static",
            segment_type="system",
            reuse_scope="global",
        )
        tool_block = ContextSegment(
            name="tools",
            text=_render_realistic_tools(_coding_tools(), canonical=variant != "original_bad_layout", turn=turn),
            stability="static",
            segment_type="tool_schema",
            reuse_scope="agent_type",
        )
        current_user = ContextSegment(
            name=f"user_{turn:02d}",
            text=(
                "Issue: pytest fails after recent parser changes. "
                "Find the root cause and decide the next tool call. "
                f"Current focus turn={turn}."
            ),
            stability="turn_dynamic",
            segment_type="current_query",
            reuse_scope="turn",
        )
    else:
        system = ContextSegment(
            name="system",
            text=(
                "You are a reliable function-calling assistant. "
                "Choose exactly one tool and return valid JSON with required arguments. "
                "Never invent tool names or omit required fields."
            ),
            stability="static",
            segment_type="system",
            reuse_scope="global",
        )
        tool_block = ContextSegment(
            name="tools",
            text=_render_realistic_tools(_function_calling_tools(), canonical=variant != "original_bad_layout", turn=turn),
            stability="static",
            segment_type="tool_schema",
            reuse_scope="agent_type",
        )
        current_user = ContextSegment(
            name=f"user_{turn:02d}",
            text=(
                "User request: coordinate a business trip from Shanghai to Singapore, "
                "check weather, estimate hotel budget, and store the itinerary. "
                f"Current turn={turn}; choose the next valid tool call."
            ),
            stability="turn_dynamic",
            segment_type="current_query",
            reuse_scope="turn",
        )

    if variant == "original_bad_layout":
        return [dynamic_header, current_user, tool_block, *history, system]
    if variant == "stable_tool_order":
        return [dynamic_header, system, tool_block, *history, current_user]
    if variant == "dynamic_fields_last":
        return [system, tool_block, *history, current_user, dynamic_header]
    if variant == "truncation_baseline":
        truncated_history = [
            _truncate_observation(segment, max_words=max(40, observation_tokens // 5))
            for segment in history
        ]
        return [system, tool_block, *truncated_history, current_user, dynamic_header]
    return [dynamic_header, current_user, tool_block, *history, system]


def _next_realistic_history_segments(
    *,
    trace_id: str,
    scenario: str,
    turn: int,
    observation_tokens: int,
) -> list[ContextSegment]:
    if scenario == "coding_agent":
        tool_call = (
            f'Assistant tool call: {{"name":"run_tests","arguments":{{"command":"pytest '
            f'tests/test_parser.py::test_nested_call -q","cwd":"/workspace/repo"}}}}'
        )
        observation = (
            f"Command: pytest tests/test_parser.py::test_nested_call -q\n"
            f"cwd: /workspace/repo\n"
            f"exit_code: 1\n"
            f"failing_test: tests/test_parser.py::test_nested_call\n"
            f"file_path: src/parser.py\n"
            f"line_number: {128 + turn}\n"
            f"error_type: AssertionError\n"
            f"root_cause_hint: parser drops nested function-call arguments after comma token.\n"
            f"stderr_tail:\n{filler_tokens(f'pytest_stderr_turn_{turn}', observation_tokens)}"
        )
        must_preserve = ("file_path", "line_number", "failing_test", "error_type", "exit_code")
    else:
        tool_call = (
            f'Assistant tool call: {{"name":"search_flights","arguments":{{"origin":"SHA",'
            f'"destination":"SIN","date":"2026-07-{10 + turn:02d}"}}}}'
        )
        observation = (
            f"tool_result: search_flights\n"
            f"status: success\n"
            f"required_fields_seen: origin,destination,date\n"
            f"candidate_flight: MU543 SHA->SIN departs 09:20 arrives 14:55\n"
            f"fare_usd: {280 + turn * 12}\n"
            f"policy_note: business fare must be below 500 USD unless approved.\n"
            f"api_json_tail:\n{filler_tokens(f'flight_api_json_turn_{turn}', observation_tokens)}"
        )
        must_preserve = ("tool_result", "status", "required_fields_seen", "candidate_flight", "fare_usd")

    return [
        ContextSegment(
            name=f"tool_call_{turn:02d}",
            text=tool_call,
            stability="session_dynamic",
            segment_type="tool_call",
            reuse_scope="session",
        ),
        ContextSegment(
            name=f"observation_{turn:02d}",
            text=observation,
            stability="turn_dynamic",
            segment_type="observation",
            reuse_scope="turn",
            must_preserve=must_preserve,
            source_pointer=f"obs://{trace_id}/{scenario}/{turn:02d}",
        ),
    ]


def _function_calling_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "search_flights",
            "description": "Find available flights for a route and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
                "required": ["origin", "destination", "date"],
            },
        },
        {
            "name": "get_weather",
            "description": "Retrieve weather forecast for a city and date.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "date": {"type": "string"}},
                "required": ["city", "date"],
            },
        },
        {
            "name": "estimate_hotel_budget",
            "description": "Estimate hotel budget for a city and number of nights.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}, "nights": {"type": "integer"}},
                "required": ["city", "nights"],
            },
        },
        {
            "name": "save_itinerary",
            "description": "Persist a travel itinerary record.",
            "parameters": {
                "type": "object",
                "properties": {"traveler": {"type": "string"}, "summary": {"type": "string"}},
                "required": ["traveler", "summary"],
            },
        },
    ]


def _coding_tools() -> list[dict[str, Any]]:
    return [
        {
            "name": "read_file",
            "description": "Read a file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "start_line": {"type": "integer"}},
                "required": ["path"],
            },
        },
        {
            "name": "search_repo",
            "description": "Search repository text for a query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "run_tests",
            "description": "Run a test command in the sandbox.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "cwd": {"type": "string"}},
                "required": ["command"],
            },
        },
        {
            "name": "apply_patch",
            "description": "Apply a unified diff patch.",
            "parameters": {
                "type": "object",
                "properties": {"diff": {"type": "string"}},
                "required": ["diff"],
            },
        },
    ]


def _render_realistic_tools(
    tools: list[dict[str, Any]],
    *,
    canonical: bool,
    turn: int,
) -> str:
    ordered_tools = tools
    if not canonical:
        shift = turn % max(1, len(tools))
        ordered_tools = tools[shift:] + tools[:shift]
    rendered = [
        json.dumps(tool, ensure_ascii=False, sort_keys=canonical, separators=(",", ":") if canonical else None)
        for tool in ordered_tools
    ]
    return "Tools:\n" + "\n".join(rendered)


def _render_variant_prompt(
    variant: str,
    segments: list[ContextSegment],
    *,
    max_observation_words: int,
) -> tuple[str, list[ContextSegment], dict[str, Any]]:
    if variant == "context_compiler_no_compression":
        compiled = compile_context(segments, compress_observations=False)
        return compiled.prompt, compiled.segments, compiled.metadata
    if variant == "context_compiler_with_observation_compression":
        compiled = compile_context(
            segments,
            compress_observations=True,
            max_observation_words=max_observation_words,
        )
        return compiled.prompt, compiled.segments, compiled.metadata
    return render_original_layout(segments), segments, {"compiler": "none"}


def _truncate_observation(segment: ContextSegment, *, max_words: int) -> ContextSegment:
    if segment.segment_type != "observation":
        return segment
    words = segment.text.split()
    if len(words) <= max_words:
        return segment
    return ContextSegment(
        name=segment.name,
        text=" ".join(words[:max_words]),
        stability=segment.stability,
        segment_type=segment.segment_type,
        reuse_scope=segment.reuse_scope,
        must_preserve=segment.must_preserve,
        source_pointer=segment.source_pointer,
    )


def _segment_to_metadata(segment: ContextSegment) -> dict[str, Any]:
    return {
        "name": segment.name,
        "text": segment.text,
        "stability": segment.stability,
        "segment_type": segment.segment_type,
        "reuse_scope": segment.reuse_scope,
        "must_preserve": list(segment.must_preserve),
        "source_pointer": segment.source_pointer,
    }
