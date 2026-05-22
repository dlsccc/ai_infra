# 定义benchmark核心数据协议

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GenerationResult:
    request_id: str
    text: str
    input_tokens: int
    output_tokens: int
    ttft_ms: float | None
    total_latency_ms: float
    metadata: dict[str, Any]


class Backend(Protocol):
    name: str

    def generate(
        self,
        requests: list[GenerationRequest],
        sampling_params: dict[str, Any],
    ) -> list[GenerationResult]:
        """Generate responses for a batch of requests."""

