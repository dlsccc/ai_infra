# Week 06 Baseline v1

## Purpose

This report records the first controlled baseline comparison between vLLM and SGLang using the same synthetic workload families in `AgentInferBench`.

The goal of this phase is:

1. verify that both backends can run the same workload set;
2. observe workload-level latency and token behavior;
3. identify where the current comparison is still unfair or incomplete.

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

## Result Summary

Fill after runs complete.

| Backend | Workload | Actual input tokens | Actual output tokens | Mean latency (ms) | P95 latency (ms) | Tokens/s | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| vLLM | plain_chat short | | | | | | |
| vLLM | plain_chat medium | | | | | | |
| vLLM | single_tool | | | | | | |
| vLLM | multi_tool_serial | | | | | | |
| vLLM | long_observation | | | | | | |
| SGLang | plain_chat short | | | | | | |
| SGLang | plain_chat medium | | | | | | |
| SGLang | single_tool | | | | | | |
| SGLang | multi_tool_serial | | | | | | |
| SGLang | long_observation | | | | | | |

## Initial Observations

Write short observations only. Avoid over-claiming.

1. Which workloads are clearly more expensive than plain chat?
2. Does multi-turn history growth noticeably increase total latency?
3. Does long observation produce a stronger prefill-heavy pattern?
4. Which differences may still come from backend API or accounting differences?

## Current Limitations

Keep these explicit:

1. offline generation does not provide real streaming TTFT;
2. token-budget fields in configs are construction targets, not exact tokenizer guarantees;
3. backend token accounting is not yet fully unified;
4. this is still a synthetic controlled baseline, not a public benchmark result.

## Next Step

Expected Week 7 direction:

1. deepen prefix/cache-related workload analysis;
2. improve token accounting and workload realism;
3. start controlled experiments around layout, tool schema stability, and session locality.
