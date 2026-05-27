from __future__ import annotations

import time
from typing import Any

from agent_bench.backends.base import GenerationRequest, GenerationResult
from agent_bench.tokenization import TokenCounter


class SGLangBackend:
    name = "sglang"

    def __init__(self, model: str, **engine_kwargs: Any) -> None:
        self.model = model
        self.engine_kwargs = engine_kwargs
        import sglang as sgl

        self._token_counter = TokenCounter(model)
        self._engine = sgl.Engine(model_path=model, **engine_kwargs)

    def generate(
        self,
        requests: list[GenerationRequest],
        sampling_params: dict[str, Any],
    ) -> list[GenerationResult]:
        prompts = [request.prompt for request in requests]
        params = {
            "temperature": sampling_params.get("temperature", 0.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "max_new_tokens": sampling_params.get("max_tokens", 64),
        }

        start = time.perf_counter()
        outputs = self._engine.generate(prompts, params)
        total_batch_latency_ms = (time.perf_counter() - start) * 1000.0
        per_request_latency_ms = total_batch_latency_ms / max(1, len(requests))

        results: list[GenerationResult] = []
        for request, output in zip(requests, outputs):
            generated = _extract_text(output)
            input_tokens = self._token_counter.count_prompt_tokens(request.prompt)
            output_tokens = self._token_counter.count_text_tokens(generated)
            results.append(
                GenerationResult(
                    request_id=request.request_id,
                    text=generated,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    ttft_ms=None,
                    total_latency_ms=per_request_latency_ms,
                    metadata={
                        **request.metadata,
                        "backend": self.name,
                        "batch_size": len(requests),
                        "batch_total_latency_ms": total_batch_latency_ms,
                        "input_token_source": "tokenizer_fallback",
                        "output_token_source": "tokenizer_fallback",
                        **self._token_counter.metadata(),
                        "ttft_note": "offline SGLang Engine.generate does not expose real TTFT; use server streaming later",
                    },
                )
            )
        return results


def _extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        for key in ("text", "output_text", "generated_text"):
            if key in output and isinstance(output[key], str):
                return output[key]
    if hasattr(output, "text"):
        value = getattr(output, "text")
        if isinstance(value, str):
            return value
    return str(output)
