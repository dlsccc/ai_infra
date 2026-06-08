from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_bench.optimizations.context_compiler import canonical_json
from agent_bench.workloads.token_utils import filler_tokens


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }

    def render(self, *, canonical: bool = False) -> str:
        if canonical:
            return canonical_json(self.to_dict())
        data = self.to_dict()
        return (
            '{"name": "'
            + data["name"]
            + '", "description": "'
            + data["description"]
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
