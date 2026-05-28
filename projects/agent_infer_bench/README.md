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

## Week 6 Server Benchmark Path

Week 6 official metrics should use server-backed streaming inference instead of offline `generate()`.

1. Start a vLLM server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint.
2. Or start an SGLang server with the same API shape.
3. Use the Week 6 server configs:
   - `configs/week06_plain_vllm_server.yaml`
   - `configs/week06_plain_sglang_server.yaml`
   - `configs/week06_agent_vllm_server.yaml`
   - `configs/week06_agent_sglang_server.yaml`
4. Run:

```bash
python scripts/benchmark/run_backend_baseline.py --config configs/week06_plain_vllm_server.yaml
python scripts/benchmark/run_backend_baseline.py --config configs/week06_agent_vllm_server.yaml
```

Or use the helper scripts on the remote server:

```bash
bash scripts/setup/start_vllm_server.sh
bash scripts/benchmark/run_week06_vllm_server.sh
```

```bash
bash scripts/setup/start_sglang_server.sh
bash scripts/benchmark/run_week06_sglang_server.sh
```

The default model path in the Week 6 server configs and startup scripts is:

```text
/root/autodl-tmp/models/Qwen2.5-7B-Instruct
```

You can override the startup script defaults with environment variables such as:

```bash
MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct PORT=8000 bash scripts/setup/start_vllm_server.sh
```

The server benchmark path records request-level:

- TTFT
- decode latency
- TPOT
- total latency
- tokens/s
- requests/s

Offline backends remain useful for smoke tests, schema validation, and fallback experiments, but not for true TTFT.

If a server request exceeds the model context limit, inspect the real tokenizer length of every generated workload request first:

```bash
python scripts/analysis/inspect_workload_tokens.py \
  --config configs/week06_agent_vllm_server.yaml \
  --context-limit 4096
```

This prints each `request_id`, its workload type, turn index, real input token count, and flags requests that exceed the provided context limit.

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
