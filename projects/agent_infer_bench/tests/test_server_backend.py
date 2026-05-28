from __future__ import annotations

import json

import httpx

from agent_bench.backends.base import GenerationRequest
from agent_bench.backends.server_backend import OpenAICompatibleServerBackend


class _StubTokenCounter:
    def count_prompt_tokens(self, prompt: str) -> int:
        return len(prompt.split())

    def count_text_tokens(self, text: str) -> int:
        return len(text.split())

    def metadata(self) -> dict[str, str]:
        return {
            "tokenizer_name_or_path": "stub",
            "tokenizer_class": "StubTokenCounter",
        }


def test_openai_compatible_server_backend_streaming_metrics() -> None:
    events = [
        'data: {"choices":[{"delta":{"role":"assistant"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert body["stream"] is True
        assert body["messages"][0]["content"] == "test prompt"
        return httpx.Response(200, text="\n\n".join(events), headers={"content-type": "text/event-stream"})

    transport = httpx.MockTransport(handler)
    backend = OpenAICompatibleServerBackend(
        name="test_server",
        base_url="http://test.local",
        model="Qwen/Qwen2.5-7B-Instruct",
        transport=transport,
        token_counter=_StubTokenCounter(),
    )
    request = GenerationRequest(
        request_id="req_001",
        prompt="test prompt",
        metadata={"expected_output_tokens": 8},
    )

    results = backend.generate([request], {"max_tokens": 8})

    assert len(results) == 1
    result = results[0]
    assert result.text == "Hello world"
    assert result.ttft_ms is not None
    assert result.total_latency_ms >= result.ttft_ms
    assert result.decode_latency_ms is not None
    assert result.tpot_ms is not None
    assert result.metadata["streaming"] is True
