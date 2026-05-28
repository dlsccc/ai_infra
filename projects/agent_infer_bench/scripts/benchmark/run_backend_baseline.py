from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agent_bench.backends.sglang_backend import SGLangBackend
from agent_bench.backends.server_backend import SGLangServerBackend, VLLMServerBackend
from agent_bench.backends.vllm_backend import VLLMBackend
from agent_bench.metrics.collector import write_run
from agent_bench.workloads.generator import generate_workloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    backend = _make_backend(config)

    traces = generate_workloads(config)
    repeat = int(config.get("repeat", 1))
    concurrency = int(config.get("concurrency", 1))

    all_runs: list[list[Any]] = []
    for _ in range(repeat):
        all_runs.append(_run_with_concurrency(backend, traces, config.get("sampling", {}), concurrency))

    selected_results = all_runs[-1]
    output_dir = ROOT / config["output_dir"]
    write_run(
        output_dir=output_dir,
        config=config,
        results=selected_results,
        extra={
            "trace_count": len(traces),
            "workload_types": sorted({trace.workload_type for trace in traces}),
            "repeat": repeat,
            "concurrency": concurrency,
            "repeat_summary": _repeat_summary(all_runs),
            "token_source_summary": _token_source_summary(selected_results),
        },
    )

    repeat_path = output_dir / "repeat_summary.json"
    repeat_path.write_text(
        json.dumps(_repeat_summary(all_runs), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote baseline results to {output_dir / 'results.json'}")
    print(f"Wrote repeat summary to {repeat_path}")
    print(f"Token accounting: {_token_source_summary(selected_results)}")


def _make_backend(config: dict[str, Any]) -> Any:
    model = str(_resolve_model_path(config["model"]))
    backend_name = config["backend"]
    if backend_name == "vllm":
        server = config.get("server")
        if server:
            return VLLMServerBackend(
                base_url=str(server["base_url"]),
                model=str(_resolve_model_path(config["model"])),
                api_key=server.get("api_key"),
                timeout_s=float(server.get("timeout_s", 300.0)),
            )
        engine = config["engine"]
        return VLLMBackend(
            model=model,
            dtype="auto",
            max_model_len=int(engine["max_model_len"]),
            gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
            tensor_parallel_size=int(engine["tensor_parallel_size"]),
            trust_remote_code=True,
        )
    if backend_name == "sglang":
        server = config.get("server")
        if server:
            return SGLangServerBackend(
                base_url=str(server["base_url"]),
                model=str(_resolve_model_path(config["model"])),
                api_key=server.get("api_key"),
                timeout_s=float(server.get("timeout_s", 300.0)),
            )
        engine = config["engine"]
        return SGLangBackend(
            model=model,
            tp_size=int(engine["tensor_parallel_size"]),
            mem_fraction_static=float(engine["mem_fraction_static"]),
            context_length=int(engine["context_length"]),
        )
    raise ValueError(f"Unsupported backend: {backend_name}")


def _resolve_model_path(model_id: str) -> Path:
    candidate = Path(model_id)
    if candidate.exists():
        return candidate

    local_dir = Path("/root/autodl-tmp/models") / model_id.split("/")[-1]
    if local_dir.exists():
        return local_dir

    return candidate


def _repeat_summary(all_runs: list[list[Any]]) -> dict[str, Any]:
    mean_latencies = [
        statistics.fmean(result.total_latency_ms for result in run) for run in all_runs
    ]
    mean_ttfts = [
        statistics.fmean(result.ttft_ms for result in run if result.ttft_ms is not None)
        for run in all_runs
        if any(result.ttft_ms is not None for result in run)
    ]
    mean_tpots = [
        statistics.fmean(result.tpot_ms for result in run if result.tpot_ms is not None)
        for run in all_runs
        if any(result.tpot_ms is not None for result in run)
    ]
    mean_output_tokens = [
        statistics.fmean(result.output_tokens for result in run) for run in all_runs
    ]
    return {
        "run_count": len(all_runs),
        "mean_latency_ms_by_run": mean_latencies,
        "mean_ttft_ms_by_run": mean_ttfts,
        "mean_tpot_ms_by_run": mean_tpots,
        "mean_output_tokens_by_run": mean_output_tokens,
        "mean_latency_ms_mean": statistics.fmean(mean_latencies) if mean_latencies else None,
        "mean_latency_ms_std": statistics.pstdev(mean_latencies) if len(mean_latencies) > 1 else 0.0,
    }


def _token_source_summary(results: list[Any]) -> dict[str, str]:
    input_sources = sorted({result.metadata.get("input_token_source", "unknown") for result in results})
    output_sources = sorted({result.metadata.get("output_token_source", "unknown") for result in results})
    return {
        "input": ", ".join(input_sources),
        "output": ", ".join(output_sources),
    }


def _run_with_concurrency(
    backend: Any,
    traces: list[Any],
    sampling_params: dict[str, Any],
    concurrency: int,
) -> list[Any]:
    requests = [request for trace in traces for request in trace.requests]
    results: list[Any] = []
    batch_index = 0
    for batch_start in range(0, len(requests), max(1, concurrency)):
        batch = requests[batch_start : batch_start + max(1, concurrency)]
        batch_results = backend.generate(batch, sampling_params)
        for result in batch_results:
            result.metadata.setdefault("batch_index", batch_index)
            result.metadata.setdefault("configured_concurrency", concurrency)
        results.extend(batch_results)
        batch_index += 1
    return results


if __name__ == "__main__":
    main()
