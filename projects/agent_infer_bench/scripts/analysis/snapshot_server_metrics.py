from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_bench.metrics.server_metrics import parse_prometheus_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    response = httpx.get(args.metrics_url, timeout=10.0)
    response.raise_for_status()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(response.text, encoding="utf-8")

    parsed = parse_prometheus_metrics(response.text)
    (output.with_suffix(output.suffix + ".json")).write_text(
        json.dumps(parsed.values, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote metrics snapshot to {output}")


if __name__ == "__main__":
    main()

