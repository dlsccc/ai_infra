from __future__ import annotations

from dataclasses import dataclass

from agent_bench.workloads.token_utils import filler_tokens


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str

    def render(self) -> str:
        return (
            '{"name": "'
            + self.name
            + '", "description": "'
            + self.description
            + '", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}'
        )


def make_tools(tool_count: int, tool_tokens: int) -> list[ToolSpec]:
    return [
        ToolSpec(
            name=f"tool_{idx:02d}",
            description=filler_tokens(f"tool_{idx:02d}_desc", tool_tokens),
        )
        for idx in range(tool_count)
    ]

