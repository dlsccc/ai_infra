from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


STABILITY_ORDER = {
    "static": 0,
    "semi_static": 1,
    "session_dynamic": 2,
    "turn_dynamic": 3,
    "ephemeral": 4,
}


@dataclass(frozen=True)
class ContextSegment:
    name: str
    text: str
    stability: str
    segment_type: str
    reuse_scope: str = "request"
    must_preserve: tuple[str, ...] = ()
    source_pointer: str | None = None


@dataclass(frozen=True)
class CompiledContext:
    prompt: str
    segments: list[ContextSegment]
    metadata: dict[str, Any]


def compile_context(
    segments: list[ContextSegment],
    *,
    compress_observations: bool = False,
    max_observation_words: int = 80,
) -> CompiledContext:
    """Compile structured agent context into a cache-friendly prompt.

    This MVP only performs deterministic layout planning and conservative
    observation compression. It intentionally avoids KV eviction/prefetch logic.
    """

    planned_segments = [
        _maybe_compress_observation(segment, max_observation_words)
        if compress_observations
        else segment
        for segment in segments
    ]
    planned_segments = sorted(
        planned_segments,
        key=lambda item: (
            STABILITY_ORDER.get(item.stability, 99),
            item.segment_type,
            item.name,
        ),
    )

    prompt_parts = [_render_segment(segment) for segment in planned_segments if segment.text.strip()]
    prompt = "\n\n".join(prompt_parts).strip() + "\nAssistant:"
    metadata = {
        "compiler": "context_compiler_mvp",
        "compress_observations": compress_observations,
        "segments": [_segment_metadata(segment) for segment in planned_segments],
    }
    return CompiledContext(prompt=prompt, segments=planned_segments, metadata=metadata)


def canonical_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def segment_cache_key(segment: ContextSegment) -> str:
    payload = canonical_json(
        {
            "name": segment.name,
            "stability": segment.stability,
            "segment_type": segment.segment_type,
            "reuse_scope": segment.reuse_scope,
            "text": segment.text,
        }
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def render_original_layout(segments: list[ContextSegment]) -> str:
    prompt_parts = [_render_segment(segment) for segment in segments if segment.text.strip()]
    return "\n\n".join(prompt_parts).strip() + "\nAssistant:"


def _maybe_compress_observation(
    segment: ContextSegment,
    max_observation_words: int,
) -> ContextSegment:
    if segment.segment_type != "observation":
        return segment
    words = segment.text.split()
    if len(words) <= max_observation_words:
        return segment

    preserved = words[: max(1, max_observation_words // 2)]
    tail = words[-max(1, max_observation_words // 4) :]
    source_pointer = segment.source_pointer or f"obs://{segment.name}"
    compressed_text = (
        " ".join(preserved)
        + "\n[... observation compressed; recoverable at "
        + source_pointer
        + " ...]\n"
        + " ".join(tail)
    )
    return ContextSegment(
        name=segment.name,
        text=compressed_text,
        stability=segment.stability,
        segment_type=segment.segment_type,
        reuse_scope=segment.reuse_scope,
        must_preserve=segment.must_preserve,
        source_pointer=source_pointer,
    )


def _render_segment(segment: ContextSegment) -> str:
    header = f"[{segment.segment_type.upper()}:{segment.name}]"
    return f"{header}\n{segment.text.strip()}"


def _segment_metadata(segment: ContextSegment) -> dict[str, Any]:
    return {
        "name": segment.name,
        "stability": segment.stability,
        "segment_type": segment.segment_type,
        "reuse_scope": segment.reuse_scope,
        "cache_key": segment_cache_key(segment),
        "source_pointer": segment.source_pointer,
        "word_count": len(segment.text.split()),
    }

