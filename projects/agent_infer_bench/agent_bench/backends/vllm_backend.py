from __future__ import annotations

import time
from typing import Any

from agent_bench.backends.base import GenerationRequest, GenerationResult
from agent_bench.tokenization import TokenCounter


class VLLMBackend:
    name = "vllm"

    def __init__(self, model: str, **engine_kwargs: Any) -> None:
        self.model = model
        self.engine_kwargs = engine_kwargs
        from vllm import LLM

        self._token_counter = TokenCounter(model)
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
            if prompt_token_ids is not None:
                input_tokens = len(prompt_token_ids)
                input_token_source = "backend_token_ids"
            else:
                input_tokens = self._token_counter.count_prompt_tokens(request.prompt)
                input_token_source = "tokenizer_fallback"

            if output_token_ids:
                output_tokens = len(output_token_ids)
                output_token_source = "backend_token_ids"
            else:
                output_tokens = self._token_counter.count_text_tokens(generated)
                output_token_source = "tokenizer_fallback"
            results.append(
                GenerationResult(
                    request_id=request.request_id,
                    text=generated,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    ttft_ms=None,
                    total_latency_ms=per_request_latency_ms,
                    decode_latency_ms=None,
                    tpot_ms=None,
                    metadata={
                        **request.metadata,
                        "backend": self.name,
                        "batch_size": len(requests),
                        "batch_total_latency_ms": total_batch_latency_ms,
                        "input_token_source": input_token_source,
                        "output_token_source": output_token_source,
                        **self._token_counter.metadata(),
                        "ttft_note": "offline vLLM LLM.generate does not expose real TTFT; use server streaming later",
                    },
                )
            )
        return results
