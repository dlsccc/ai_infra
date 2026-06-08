# AgentInferBench

AgentInferBench is now focused on **Agent Context Compiler** experiments for long-observation tool-using agents.

The current research question is:

```text
Can we diagnose and repair cacheability breakdowns in tool-using agent contexts before they enter LLM serving backends?
```

## Current Scope

This project supports:

1. Context compiler MVP: segment parsing, stable layout, deterministic tool serialization, recoverable observation compression.
2. Cacheability diagnostics: PSR, DPR, CRO and prefix-overlap analysis.
3. Realistic synthetic workloads: function-calling agents and coding agents with long observations.
4. Server-backed experiments: vLLM and SGLang through OpenAI-compatible streaming APIs.
5. Backend metrics snapshots: Prometheus `/metrics` before/after each variant run.

Out of scope for this repository phase:

```text
KV eviction, KV prefetch, multi-agent KV sharing, tool-result caching, semantic answer caching.
```

## Layout

```text
agent_bench/
  analysis/
    cacheability_metrics.py     # PSR/DPR/CRO computation
    summarize.py                # generic result summaries
    workload_inspector.py        # workload token inspection
  backends/
    base.py
    mock_backend.py
    server_backend.py            # OpenAI-compatible vLLM/SGLang server backend
    sglang_backend.py
    vllm_backend.py
  metrics/
    collector.py                 # result writer
    server_metrics.py            # Prometheus metrics parser
  optimizations/
    context_compiler.py           # Agent Context Compiler MVP
  workloads/
    generator.py                  # synthetic + realistic context compiler workloads
    schemas.py
    token_utils.py

configs/
  context_compiler/
    mvp_mock.yaml
    realistic_mock.yaml
    realistic_vllm.yaml
    realistic_sglang.yaml
  env.yaml

scripts/
  analysis/
    analyze_context_compiler.py
    inspect_workload_tokens.py
    snapshot_server_metrics.py
  benchmark/
    run_backend_baseline.py
    run_context_compiler_variant.py
  setup/
    check_gpu_env.sh
    start_sglang_server.sh
    start_vllm_prefix_cache_off.sh
    start_vllm_prefix_cache_on.sh
    start_vllm_server.sh

experiments/
  runs/
  reports/

tests/
```

## Local Smoke Test

Run from this directory:

```bash
python scripts/benchmark/run_context_compiler_variant.py \
  --config configs/context_compiler/realistic_mock.yaml \
  --variant context_compiler_with_observation_compression \
  --warmup-requests 1
```

Run tests:

```bash
python -m pytest tests
```

## Server Experiment

Start vLLM with prefix caching:

```bash
MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.85 \
bash scripts/setup/start_vllm_prefix_cache_on.sh
```

Run one variant:

```bash
python scripts/benchmark/run_context_compiler_variant.py \
  --config configs/context_compiler/realistic_vllm.yaml \
  --variant original_bad_layout \
  --metrics-url http://127.0.0.1:8000/metrics \
  --warmup-requests 4
```

Recommended variants:

```text
original_bad_layout
stable_tool_order
dynamic_fields_last
context_compiler_no_compression
context_compiler_with_observation_compression
truncation_baseline
```

For final numbers, restart the backend before each variant to clear process-local prefix/KV cache state.

## Key Outputs

Each variant run writes:

```text
results.json
repeat_summary.json
metrics_before.prom
metrics_after.prom
metrics_delta.json
warmup/results.json
```

Metrics of interest:

```text
PSR / DPR / CRO
TTFT / total latency / P95 latency
backend-reported prefix cache hits and queries when available
```

