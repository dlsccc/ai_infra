from __future__ import annotations

from typing import Any


class TokenCounter:
    def __init__(self, model_name_or_path: str) -> None:
        from transformers import AutoTokenizer

        self.model_name_or_path = model_name_or_path
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )

    def count_text_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def count_prompt_tokens(self, prompt: str) -> int:
        return self.count_text_tokens(prompt)

    def metadata(self) -> dict[str, Any]:
        tokenizer_name = getattr(self._tokenizer, "name_or_path", self.model_name_or_path)
        tokenizer_cls = type(self._tokenizer).__name__
        return {
            "tokenizer_name_or_path": tokenizer_name,
            "tokenizer_class": tokenizer_cls,
        }

