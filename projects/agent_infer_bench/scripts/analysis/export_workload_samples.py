from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_bench.workloads.generator import generate_workloads  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", default="function_calling")
    parser.add_argument("--turn", type=int, default=1)
    parser.add_argument("--max-prompt-chars", type=int, default=5000)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    traces = generate_workloads(config)
    rows: list[dict[str, Any]] = []
    for trace in traces:
        for request in trace.requests:
            metadata = request.metadata
            if metadata.get("scenario") != args.scenario:
                continue
            if int(metadata.get("turn", -1)) != args.turn:
                continue
            rows.append({"request": request, "metadata": metadata})

    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row["metadata"].get("variant", "unknown"))].append(row)

    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        _render_markdown(
            config_path=config_path,
            scenario=args.scenario,
            turn=args.turn,
            by_variant=by_variant,
            max_prompt_chars=args.max_prompt_chars,
        ),
        encoding="utf-8",
    )
    print(f"Wrote workload samples to {output}")


def _render_markdown(
    *,
    config_path: Path,
    scenario: str,
    turn: int,
    by_variant: dict[str, list[dict[str, Any]]],
    max_prompt_chars: int,
) -> str:
    lines = [
        "# Context Compiler Workload Samples",
        "",
        f"Config: `{config_path}`",
        f"Scenario: `{scenario}`",
        f"Turn: `{turn}`",
        "",
        "说明：这份文件只展示自动生成的请求样例，不会调用模型。",
        "同一个 scenario/turn 下对比不同 variant，可以直观看到稳定内容、动态字段、observation 的位置变化。",
        "",
    ]

    for variant in sorted(by_variant):
        row = by_variant[variant][0]
        request = row["request"]
        metadata = row["metadata"]
        prompt = request.prompt
        truncated = len(prompt) > max_prompt_chars
        shown_prompt = prompt[:max_prompt_chars]
        if truncated:
            shown_prompt += "\n\n[... prompt truncated in sample export ...]"

        lines.extend(
            [
                f"## {variant}",
                "",
                f"Request ID: `{request.request_id}`",
                f"Prompt chars: `{len(prompt)}`",
                f"Prefix overlap ratio: `{metadata.get('prefix_overlap_ratio', 0.0):.4f}`",
                f"Compiler: `{metadata.get('compiler_metadata', {}).get('compiler', 'none')}`",
                "",
                "### Segment Order",
                "",
            ]
        )
        for segment in metadata.get("context_segments", []):
            lines.append(
                "- "
                f"`{segment.get('segment_type')}/{segment.get('name')}` "
                f"stability=`{segment.get('stability')}` "
                f"scope=`{segment.get('reuse_scope')}` "
                f"words=`{len(str(segment.get('text', '')).split())}`"
            )

        lines.extend(
            [
                "",
                "### Prompt",
                "",
                "```text",
                shown_prompt,
                "```",
                "",
            ]
        )

    return "\n".join(lines)


if __name__ == "__main__":
    main()
