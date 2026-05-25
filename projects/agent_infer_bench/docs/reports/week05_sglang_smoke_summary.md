# Week 05 SGLang Smoke Summary

## Purpose

This smoke run validates the first end-to-end real inference path for `AgentInferBench` with SGLang:

- synthetic workload generation
- real `SGLangBackend`
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
| Driver CUDA version (`nvidia-smi`) | |
| Python | |
| Torch | 2.9.1 |  
| Torch CUDA | cu128 |
| SGLang | 0.5.6 |
| Model path | |
| Conda env path | |

## Smoke Config

Config file:

`projects/agent_infer_bench/configs/baseline_sglang.yaml`

Current engine settings:

| Setting | Value |
|---|---|
| `tensor_parallel_size` | `1` |
| `mem_fraction_static` | `0.85` |
| `context_length` | `4096` |

Current smoke workloads:

| Workload | Notes |
|---|---|
| `plain_chat` short | quick minimal request |
| `plain_chat` medium | longer prompt sanity check |
| `single_tool` | first tool-style Agent prompt |

## Output Location

Raw run directory:

`projects/agent_infer_bench/experiments/runs/week05/sglang_smoke/`

Expected files:

- `results.json`
- `summary.md`

## Key Results

Fill this section from the latest generated `summary.md`.

| Metric | Value |
|---|---|
| Request count | 3 |
| Mean latency (ms) | 3745.32 |
| P95 latency (ms) | 3745.32 |
| Mean TTFT (ms) | null |
| Tokens/s | 14.06 |

## Main Observations

Suggested points to record:

1. SGLang successfully initialized with the current CUDA / torch stack.
2. The local model path was loaded successfully.
3. The synthetic workload pipeline worked with a real backend.
4. Offline SGLang generation currently gives total latency, but not true streaming TTFT.
5. Synthetic token-budget fields are not equal to the tokenizer-measured token count.
6. Note any output-format differences compared with vLLM.

## Problems Encountered

Record real issues encountered during setup and reruns.

| Problem | Root Cause | Fix |
|---|---|---|
| Installation failed due to disk pressure | pip cache and temp files were written to system disk | move pip/temp/conda caches to data disk |
| Prompt too long for context window | synthetic budget not tokenizer-accurate | reduce smoke config / later switch to tokenizer-aware control |
| Output parsing mismatch | backend API return structure differed from assumptions | add compatibility handling in backend |
| Version drift in torch stack | SGLang install changed dependency versions | record and freeze working stack |

## What This Smoke Test Proves

This run proves:

1. the Week 5 project skeleton works with a second real backend;
2. the benchmark pipeline can move from `mock` and `vLLM` to `SGLang`;
3. the current SGLang environment is capable of running a real Qwen inference workload through the project codepath.

This run does **not** yet prove:

1. accurate TTFT measurement;
2. realistic Agent workload scaling behavior;
3. final benchmark-quality latency conclusions;
4. fair head-to-head comparison against vLLM.

## Next Step

Immediate next steps after this smoke run:

1. freeze environment versions;
2. record the result in `docs/weekly/week05.md`;
3. align smoke/baseline configs between vLLM and SGLang;
4. prepare the first controlled Week 6 baseline runs.
