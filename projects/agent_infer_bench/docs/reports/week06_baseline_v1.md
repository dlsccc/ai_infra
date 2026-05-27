# Week 06 Baseline v1

## Purpose

This report records the first controlled baseline comparison between vLLM and SGLang using the same synthetic workload families in `AgentInferBench`.

The goal of this phase is:

1. verify that both backends can run the same workload set;
2. observe workload-level latency and token behavior;
3. identify where the current comparison is still unfair or incomplete.

## Scope

This report should answer:

1. how plain chat differs from Agent-style workloads;
2. how multi-turn history and long observations affect latency;
3. whether the current vLLM and SGLang paths are comparable enough for deeper Week 7 experiments.

This report should **not** claim:

1. definitive framework superiority;
2. production-grade throughput conclusions;
3. real TTFT conclusions;
4. public-benchmark-quality results.

## Backends

| Backend | Status | Notes |
|---|---|---|
| vLLM | | |
| SGLang | | |

## Workloads

| Workload | Description | Why it matters |
|---|---|---|
| `plain_chat` short | minimal chat-style prompt | sanity baseline |
| `plain_chat` medium | longer chat-style prompt | prompt-length sensitivity |
| `single_tool` | one-turn tool-style prompt | system/tool prefix overhead |
| `multi_tool_serial` | multi-turn Agent trace | history growth |
| `long_observation` | multi-turn trace with longer observations | prefill-heavy stress |

## Config

Record the final config paths used:

- `configs/baseline_v1_vllm.yaml`
- `configs/baseline_v1_sglang.yaml`

Record any local edits made before reruns:

| Item | Value |
|---|---|
| vLLM config revision | |
| SGLang config revision | |
| Repeat count | 3 |
| Model path | |
| `max_tokens` | 64 |
| Context limit | 4096 |

## Environment Snapshot

| Item | vLLM | SGLang |
|---|---|---|
| Python | 3.11.15 | |
| Torch | | |
| Torch CUDA | | |
| Backend version | | |
| GPU | | |
| Driver CUDA | | |
| Token source summary | | |

## Raw Result Paths

| Backend | Raw run dir | Summary file | Repeat summary |
|---|---|---|---|
| vLLM | `experiments/runs/week06/vllm_baseline_v1/` | `summary.md` | `repeat_summary.json` |
| SGLang | `experiments/runs/week06/sglang_baseline_v1/` | `summary.md` | `repeat_summary.json` |

## Result Summary

Fill after runs complete.

| Backend | Workload | Actual input tokens | Actual output tokens | Mean latency (ms) | P95 latency (ms) | Tokens/s | Token source | Notes |
|---|---|---:|---:|---:|---:|---:|---|---|
| vLLM | plain_chat short | 319 | | | | | | |
| vLLM | plain_chat medium | | | | | | | |
| vLLM | single_tool | | | | | | | |
| vLLM | multi_tool_serial | | | | | | | |
| vLLM | long_observation | | | | | | | |
| SGLang | plain_chat short | | | | | | | |
| SGLang | plain_chat medium | | | | | | | |
| SGLang | single_tool | | | | | | | |
| SGLang | multi_tool_serial | | | | | | | |
| SGLang | long_observation | | | | | | | |

## Repeat Stability

Copy the key fields from `repeat_summary.json`.

| Backend | Mean latency by run | Mean latency mean | Mean latency std | Comment |
|---|---|---:|---:|---|
| vLLM | | | | |
| SGLang | | | | |

## Observations

Write short observations only. Avoid over-claiming.

### 1. Plain Chat vs Agent-Style Workloads

- Which workloads are clearly more expensive than `plain_chat`?
- How much of that difference appears to come from input growth?

### 2. History Growth

- Does `multi_tool_serial` noticeably increase total latency relative to `single_tool`?
- Does the increase look roughly proportional to observed input-token growth?

### 3. Long Observation Impact

- Does `long_observation` produce a stronger prefill-heavy pattern?
- Is the effect similar on both backends?

### 4. Backend Comparison

- Are there obvious performance gaps?
- Which gaps may still come from backend API or accounting differences rather than real system differences?

## Problems and Anomalies

Record anything suspicious here.

| Issue | Backend | Symptom | Suspected Cause | Action Taken |
|---|---|---|---|---|
| | | | | |

## Current Limitations

Keep these explicit:

1. offline generation does not provide real streaming TTFT;
2. token-budget fields in configs are construction targets, not exact tokenizer guarantees;
3. token accounting is improved, but not fully symmetric if one backend exposes token ids and the other does not;
4. this is still a synthetic controlled baseline, not a public benchmark result.

## Interim Conclusions

Write 3-5 cautious conclusions here.

1. |
2. |
3. |
4. |
5. |

## Next Step

Expected Week 7 direction:

1. deepen prefix/cache-related workload analysis;
2. improve workload realism and token-budget control;
3. start controlled experiments around layout, tool schema stability, and session locality.
