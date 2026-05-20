# AgentInferBench

AgentInferBench is a benchmark and analysis project for tool-using Agent LLM serving workloads.

The short-term goal is to compare vLLM and SGLang on Agent-style multi-turn workloads, then evaluate lightweight cache-aware optimizations such as prompt canonicalization, stable tool ordering, and session-aware routing.

## Current Status

Week 5 focus:

1. Build the project skeleton.
2. Run a local mock benchmark without GPU.
3. Prepare remote GPU environment checks.
4. Add vLLM and SGLang backends after the remote server is ready.

## Layout

```text
agent_bench/
  backends/       # backend adapters: mock, vLLM, SGLang
  workloads/      # Agent workload generation
  metrics/        # latency and result collection
  analysis/       # result summaries and plots
  optimizations/  # prompt/routing optimizations
configs/          # experiment configs
scripts/          # setup, benchmark, analysis entrypoints
experiments/      # raw runs and generated reports
docs/             # weekly notes, reports, blogs
tests/            # lightweight tests
```

## Quick Start

Run the local mock smoke test:

```bash
cd projects/agent_infer_bench
python scripts/benchmark/run_all_smoke.py
python scripts/analysis/make_tables.py --run-dir experiments/runs/week05/mock_smoke
```

This does not require a GPU. It validates the workload generation, backend interface, metric collection, and result format.

## Remote GPU Plan

After a remote GPU server is available:

1. Clone or pull this repository on the server.
2. Run `bash scripts/setup/check_gpu_env.sh`.
3. Install vLLM and SGLang in a clean environment.
4. Run small hello-world tests for each backend.
5. Replace the mock backend with vLLM/SGLang backend smoke tests.

## Core Metrics

- TTFT: time to first token.
- TPOT: time per output token.
- JCT: job completion time for a complete Agent task.
- Throughput: tokens/s and requests/s.
- Prefix overlap: a proxy for cache reuse opportunity.
- Peak memory: GPU memory high-water mark when available.

