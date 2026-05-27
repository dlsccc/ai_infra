from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_bench.backends.vllm_backend import VLLMBackend
from agent_bench.metrics.collector import write_run
from agent_bench.workloads.generator import generate_workloads


def main() -> None:
    config_path = ROOT / "configs" / "baseline_vllm.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    backend = VLLMBackend(
        model=str(_resolve_model_path(config["model"])),
        dtype="auto",
        max_model_len=int(config["engine"]["max_model_len"]),
        gpu_memory_utilization=float(config["engine"]["gpu_memory_utilization"]),
        tensor_parallel_size=int(config["engine"]["tensor_parallel_size"]),
        trust_remote_code=True,
    )

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
            "token_source_summary": _token_source_summary(results),
        },
    )
    print(f"Wrote vLLM smoke results to {output_dir / 'results.json'}")
    print(f"Token accounting: {_token_source_summary(results)}")


def _resolve_model_path(model_id: str) -> Path:
    candidate = Path(model_id)
    if candidate.exists():
        return candidate

    local_dir = Path("/root/autodl-tmp/models") / model_id.split("/")[-1]
    if local_dir.exists():
        return local_dir

    return candidate


if __name__ == "__main__":
    main()


def _token_source_summary(results: list[object]) -> dict[str, str]:
    input_sources = sorted(
        {
            result.metadata.get("input_token_source", "unknown")
            for result in results
            if hasattr(result, "metadata")
        }
    )
    output_sources = sorted(
        {
            result.metadata.get("output_token_source", "unknown")
            for result in results
            if hasattr(result, "metadata")
        }
    )
    return {
        "input": ", ".join(input_sources),
        "output": ", ".join(output_sources),
    }
