from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_bench.backends.mock_backend import MockBackend
from agent_bench.metrics.collector import write_run
from agent_bench.workloads.generator import generate_workloads


def main() -> None:
    config_path = ROOT / "configs" / "mock_smoke.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    backend = MockBackend(seed=int(config.get("seed", 42)))
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
        },
    )
    print(f"Wrote smoke results to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()

