from __future__ import annotations

import random
import time
from typing import Any

from agent_bench.backends.base import GenerationRequest, GenerationResult
from agent_bench.workloads.token_utils import estimate_tokens


class MockBackend:
    name = "mock"

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def generate(
        self,
        requests: list[GenerationRequest],
        sampling_params: dict[str, Any],
    ) -> list[GenerationResult]:
        results: list[GenerationResult] = []
        max_tokens = int(sampling_params.get("max_tokens", 64))

        for request in requests:
            input_tokens = estimate_tokens(request.prompt)
            output_tokens = int(request.metadata.get("expected_output_tokens", max_tokens))
            prefill_ms = 4.0 + input_tokens * 0.035
            decode_ms = output_tokens * 1.4
            jitter_ms = self._rng.uniform(0.0, 5.0)
            total_latency_ms = prefill_ms + decode_ms + jitter_ms
            ttft_ms = prefill_ms + self._rng.uniform(0.0, 2.0)
            decode_latency_ms = max(0.0, total_latency_ms - ttft_ms)
            tpot_ms = decode_latency_ms / output_tokens if output_tokens > 0 else None

            # Keep local smoke tests fast while still producing realistic metrics.
            time.sleep(min(total_latency_ms / 1000.0, 0.02))

            results.append(
                GenerationResult(
                    request_id=request.request_id,
                    text='{"name": "mock_tool", "arguments": {"query": "mock"}}',
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    ttft_ms=ttft_ms,
                    total_latency_ms=total_latency_ms,
                    decode_latency_ms=decode_latency_ms,
                    tpot_ms=tpot_ms,
                    metadata={
                        "backend": self.name,
                        "simulated_prefill_ms": prefill_ms,
                        "simulated_decode_ms": decode_ms,
                    },
                )
            )

        return results
