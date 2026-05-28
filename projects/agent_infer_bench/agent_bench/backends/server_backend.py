from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from agent_bench.backends.base import GenerationRequest, GenerationResult
from agent_bench.tokenization import TokenCounter


class OpenAICompatibleServerBackend:
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 300.0,
        chat_path: str = "/v1/chat/completions",
        default_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | httpx.AsyncBaseTransport | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.name = name
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.chat_path = chat_path
        self._api_key = api_key or "EMPTY"
        self._timeout_s = timeout_s
        self._transport = transport
        self._token_counter = token_counter or TokenCounter(model)
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if default_headers:
            self._headers.update(default_headers)

    def generate(
        self,
        requests: list[GenerationRequest],
        sampling_params: dict[str, Any],
    ) -> list[GenerationResult]:
        return asyncio.run(self._generate_many(requests, sampling_params))

    async def _generate_many(
        self,
        requests: list[GenerationRequest],
        sampling_params: dict[str, Any],
    ) -> list[GenerationResult]:
        async with httpx.AsyncClient(timeout=self._timeout_s, transport=self._transport) as client:
            tasks = [
                self._generate_one_async(client, request, sampling_params)
                for request in requests
            ]
            return list(await asyncio.gather(*tasks))

    async def _generate_one_async(
        self,
        client: httpx.AsyncClient,
        request: GenerationRequest,
        sampling_params: dict[str, Any],
    ) -> GenerationResult:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": True,
            "temperature": sampling_params.get("temperature", 0.0),
            "top_p": sampling_params.get("top_p", 1.0),
            "max_tokens": int(
                request.metadata.get(
                    "expected_output_tokens",
                    sampling_params.get("max_tokens", 64),
                )
            ),
        }

        input_tokens = self._token_counter.count_prompt_tokens(request.prompt)
        start = time.perf_counter()
        first_token_at: float | None = None
        text_parts: list[str] = []
        finish_reason: str | None = None

        async with client.stream(
            "POST",
            f"{self.base_url}{self.chat_path}",
            headers=self._headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                delta = _extract_delta_text(event)
                if delta:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(delta)
                finish_reason = _extract_finish_reason(event) or finish_reason

        end = time.perf_counter()
        text = "".join(text_parts)
        output_tokens = self._token_counter.count_text_tokens(text)
        total_latency_ms = (end - start) * 1000.0
        ttft_ms = ((first_token_at - start) * 1000.0) if first_token_at is not None else None
        decode_latency_ms = (
            ((end - first_token_at) * 1000.0) if first_token_at is not None else None
        )
        tpot_ms = (
            decode_latency_ms / output_tokens
            if decode_latency_ms is not None and output_tokens > 0
            else None
        )

        return GenerationResult(
            request_id=request.request_id,
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            total_latency_ms=total_latency_ms,
            decode_latency_ms=decode_latency_ms,
            tpot_ms=tpot_ms,
            metadata={
                **request.metadata,
                "backend": self.name,
                "base_url": self.base_url,
                "streaming": True,
                "finish_reason": finish_reason,
                "input_token_source": "tokenizer_fallback",
                "output_token_source": "tokenizer_fallback",
                **self._token_counter.metadata(),
            },
        )


class VLLMServerBackend(OpenAICompatibleServerBackend):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        super().__init__(
            name="vllm_server",
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
        )


class SGLangServerBackend(OpenAICompatibleServerBackend):
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: float = 300.0,
    ) -> None:
        super().__init__(
            name="sglang_server",
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout_s=timeout_s,
        )


def _extract_delta_text(event: dict[str, Any]) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta = choices[0].get("delta", {})
    if not isinstance(delta, dict):
        return ""
    content = delta.get("content", "")
    return content if isinstance(content, str) else ""


def _extract_finish_reason(event: dict[str, Any]) -> str | None:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    finish_reason = choices[0].get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else None
