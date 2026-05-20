from __future__ import annotations

from typing import Any

from agent_bench.backends.base import GenerationRequest, GenerationResult


class SGLangBackend:
    name = "sglang"

    def __init__(self, model: str, **engine_kwargs: Any) -> None:
        self.model = model
        self.engine_kwargs = engine_kwargs
        raise NotImplementedError(
            "SGLangBackend is planned for Week 5 after the remote GPU environment is ready."
        )

    def generate(
        self,
        requests: list[GenerationRequest],
        sampling_params: dict[str, Any],
    ) -> list[GenerationResult]:
        raise NotImplementedError

