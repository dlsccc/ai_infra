from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_bench.backends.vllm_backend import VLLMBackend
from agent_bench.metrics.collector import write_run
from agent_bench.workloads.generator import generate_workloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline_vllm.yaml"))
    parser.add_argument(
        "--model",
        default=None,
        help="Override config model. Use this for a local model path on the remote server.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.model:
        config["model"] = args.model

    backend = VLLMBackend(model=config["model"], **config.get("engine", {}))
    traces = generate_workloads(config)
    requests = [request for trace in traces for request in trace.requests]
    results = backend.generate(requests, config.get("sampling", {}))
    output_dir = ROOT / config["output_dir"]
    write_run(
        output_dir=output_dir,
        config=config,
        results=results,
        extra={
            "trace_count": len(traces),
            "workload_types": sorted({trace.workload_type for trace in traces}),
            "runner": "run_vllm_smoke.py",
        },
    )
    print(f"Wrote vLLM smoke results to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
