from __future__ import annotations

import time
from typing import Any

from agent_bench.backends.base import GenerationRequest, GenerationResult
from agent_bench.workloads.token_utils import estimate_tokens


class VLLMBackend:
    name = "vllm"

    def __init__(self, model: str, **engine_kwargs: Any) -> None:
        self.model = model
        self.engine_kwargs = engine_kwargs
        from vllm import LLM

        self._llm = LLM(model=model, **engine_kwargs)

    def generate(
        self,
        requests: list[GenerationRequest],
        sampling_params: dict[str, Any],
    ) -> list[GenerationResult]:
        from vllm import SamplingParams

        prompts = [request.prompt for request in requests]
        params = SamplingParams(**sampling_params)
        start = time.perf_counter()
        outputs = self._llm.generate(prompts, params)
        total_batch_latency_ms = (time.perf_counter() - start) * 1000.0
        per_request_latency_ms = total_batch_latency_ms / max(1, len(requests))

        results: list[GenerationResult] = []
        for request, output in zip(requests, outputs):
            generated = output.outputs[0].text if output.outputs else ""
            output_token_ids = output.outputs[0].token_ids if output.outputs else []
            prompt_token_ids = getattr(output, "prompt_token_ids", None)
            input_tokens = (
                len(prompt_token_ids)
                if prompt_token_ids is not None
                else estimate_tokens(request.prompt)
            )
            output_tokens = len(output_token_ids) if output_token_ids else estimate_tokens(generated)
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
                        "ttft_note": "offline vLLM LLM.generate does not expose real TTFT; use server streaming later",
                    },
                )
            )
        return results
