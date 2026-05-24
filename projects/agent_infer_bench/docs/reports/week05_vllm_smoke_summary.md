# Week 05 vLLM Smoke Summary

## Purpose

This smoke run validates the first end-to-end real inference path for `AgentInferBench`:

- synthetic workload generation
- real `VLLMBackend`
- real Qwen model loading
- result serialization to `results.json`
- markdown summary generation

This is an engineering validation step, not a final benchmark result.

## Environment

Fill in or revise after the latest rerun.

| Item | Value |
|---|---|
| Server | |
| GPU | |
| GPU memory | |
| Driver CUDA version (`nvidia-smi`) | 12.8 |
| Python | 3.11.15 |
| Torch | |
| Torch CUDA | |
| vLLM | |
| Transformers | |
| Tokenizers | |
| Model path | |
| Conda env path | |

## Smoke Config

Config file:

`projects/agent_infer_bench/configs/baseline_vllm.yaml`

Current engine settings:

| Setting | Value |
|---|---|
| `tensor_parallel_size` | `1` |
| `gpu_memory_utilization` | `0.85` |
| `max_model_len` | `4096` |

Current smoke workloads:

| Workload | Notes |
|---|---|
| `plain_chat` short | quick minimal request |
| `plain_chat` medium | longer prompt sanity check |
| `single_tool` | first tool-style Agent prompt |

## Output Location

Raw run directory:

`projects/agent_infer_bench/experiments/runs/week05/vllm_smoke/`

Expected files:

- `results.json`
- `summary.md`

## Key Results

Fill this section from the latest generated `summary.md`.

| Metric | Value |
|---|---|
| Request count | 3 |
| Mean latency (ms) | 462.60 |
| P95 latency (ms) | 462.60 |
| Mean TTFT (ms) | null |
| Tokens/s | 138.35 |

## Main Observations

Suggested points to record:

1. vLLM successfully initialized with the current CUDA / torch stack.
2. The local model path was loaded successfully.
3. The synthetic workload pipeline worked with a real backend.
4. Offline `LLM.generate()` currently gives total latency, but not true streaming TTFT.
5. Synthetic token-budget fields are not equal to the tokenizer-measured token count.

## Problems Encountered

Record real issues encountered during setup and reruns.

| Problem | Root Cause | Fix |
|---|---|---|
| Hugging Face download timeout | direct access unavailable | set `HF_ENDPOINT=https://hf-mirror.com` |
| Torch / driver mismatch | incompatible CUDA runtime | use torch build compatible with driver |
| Prompt too long for `max_model_len` | synthetic budget not tokenizer-accurate | reduce smoke config / later switch to tokenizer-aware control |
| Tokenizer compatibility issue | dependency mismatch | align `transformers` / `tokenizers` with working vLLM stack |

## What This Smoke Test Proves

This run proves:

1. the Week 5 project skeleton is functional;
2. the benchmark pipeline can move from `mock` to `vLLM`;
3. the current environment is capable of running a real Qwen inference workload through the project codepath.

This run does **not** yet prove:

1. accurate TTFT measurement;
2. realistic Agent workload scaling behavior;
3. final benchmark-quality latency conclusions;
4. comparison against SGLang.

## Next Step

Immediate next steps after this smoke run:

1. freeze environment versions;
2. record the result in `docs/weekly/week05.md`;
3. set up and validate SGLang;
4. prepare the first controlled Week 6 baseline runs.
