from __future__ import annotations


def estimate_tokens(text: str) -> int:
    """Cheap token estimate for local smoke tests.

    Real experiments should use the target model tokenizer. This estimate keeps
    mock runs dependency-light and deterministic.
    """

    if not text:
        return 0
    return max(1, len(text) // 4)


def filler_tokens(prefix: str, target_tokens: int) -> str:
    if target_tokens <= 0:
        return ""
    words = [f"{prefix}_{idx}" for idx in range(target_tokens)]
    return " ".join(words)


def common_prefix_tokens(left: str, right: str) -> int:
    left_parts = left.split()
    right_parts = right.split()
    count = 0
    for left_token, right_token in zip(left_parts, right_parts):
        if left_token != right_token:
            break
        count += 1
    return count

