from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_bench.analysis.workload_inspector import inspect_workloads
from agent_bench.tokenization import TokenCounter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to workload config YAML.")
    parser.add_argument(
        "--context-limit",
        type=int,
        default=None,
        help="Optional max context length to flag oversized requests.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    token_counter = TokenCounter(str(_resolve_model_path(config["model"])))
    report = inspect_workloads(config, token_counter)

    print("# Workload Token Inspection")
    print()
    print(f"Config: {config_path}")
    print(f"Model/tokenizer: {_resolve_model_path(config['model'])}")
    print(f"Trace count: {report['trace_count']}")
    print(f"Request count: {report['request_count']}")
    print(f"Max input tokens: {report['max_input_tokens']}")
    print()

    print("## Requests")
    for row in report["request_rows"]:
        status = ""
        if args.context_limit is not None and row["input_tokens"] > args.context_limit:
            status = "  EXCEEDS_LIMIT"
        print(
            f"{row['request_id']}: workload={row['workload_type']}, "
            f"turn={row['turn']}, input_tokens={row['input_tokens']}, "
            f"expected_output_tokens={row['expected_output_tokens']}, "
            f"prefix_overlap_tokens={row['prefix_overlap_tokens']}, "
            f"prefix_overlap_ratio={_fmt_ratio(row['prefix_overlap_ratio'])}"
            f"{status}"
        )

    print()
    print("## Workload Summary")
    for row in report["workload_summary"]:
        print(
            f"{row['workload_type']}: request_count={row['request_count']}, "
            f"min={row['min_input_tokens']}, max={row['max_input_tokens']}, "
            f"mean={row['mean_input_tokens']:.2f}"
        )


def _resolve_model_path(model_id: str) -> Path:
    candidate = Path(model_id)
    if candidate.exists():
        return candidate

    local_dir = Path("/root/autodl-tmp/models") / model_id.split("/")[-1]
    if local_dir.exists():
        return local_dir

    return candidate


def _fmt_ratio(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "n/a"


if __name__ == "__main__":
    main()
