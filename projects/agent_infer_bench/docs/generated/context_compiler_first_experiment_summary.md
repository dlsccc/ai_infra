# Context Compiler Run Summary

Run root: `E:\ai_infra\projects\agent_infer_bench\experiments\runs\context_compiler\realistic`

| condition | variant | req | input tok | out tok | TTFT ms | p95 TTFT | latency ms | p95 latency | TPOT ms | backend cache ratio | source |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mock | context_compiler_with_observation_compression | 4 | 459.0 | 64.0 | 20.57 | 24.43 | 112.84 | 117.42 | 1.44 |  |  |
| mock_metrics_error_smoke | original_bad_layout | 4 | 675.5 | 64.0 | 28.15 | 40.64 | 120.42 | 134.53 | 1.44 |  |  |
| sglang | context_compiler_no_compression | 32 | 8433.8 | 80.9 | 30.54 | 55.48 | 786.98 | 984.01 | 9.33 | 0.9476 | SGLang cached/prompt |
| sglang | context_compiler_with_observation_compression | 32 | 1606.5 | 80.3 | 18.67 | 24.87 | 747.44 | 896.73 | 9.08 | 0.8826 | SGLang cached/prompt |
| sglang | dynamic_fields_last | 32 | 8433.8 | 76.9 | 30.36 | 51.44 | 748.34 | 981.07 | 9.32 | 0.9730 | SGLang cached/prompt |
| sglang | original_bad_layout | 32 | 8497.8 | 88.6 | 31.06 | 50.99 | 856.97 | 979.78 | 9.31 | 0.7901 | SGLang cached/prompt |
| sglang | stable_tool_order | 32 | 8433.8 | 90.6 | 30.00 | 51.60 | 873.75 | 979.78 | 9.30 | 0.7896 | SGLang cached/prompt |
| sglang | truncation_baseline | 32 | 1629.8 | 70.4 | 18.59 | 23.89 | 658.67 | 899.73 | 9.11 | 0.9711 | SGLang cached/prompt |
| vllm | context_compiler_no_compression | 32 | 8433.8 | 82.1 | 31.17 | 52.96 | 848.42 | 1040.65 | 9.94 | 0.9467 | vLLM hits/queries |
| vllm | context_compiler_with_observation_compression | 32 | 1606.5 | 83.9 | 19.57 | 26.92 | 832.07 | 956.62 | 9.70 | 0.8786 | vLLM hits/queries |
| vllm | dynamic_fields_last | 32 | 8433.8 | 76.9 | 30.68 | 54.33 | 791.56 | 1033.70 | 9.88 | 0.9721 | vLLM hits/queries |
| vllm | original_bad_layout | 32 | 8497.8 | 89.1 | 30.92 | 57.17 | 913.80 | 1041.33 | 9.90 | 0.7893 | vLLM hits/queries |
| vllm | stable_tool_order | 32 | 8433.8 | 90.6 | 30.72 | 50.57 | 923.95 | 1035.50 | 9.85 | 0.7888 | vLLM hits/queries |
| vllm | truncation_baseline | 32 | 1629.8 | 70.5 | 19.14 | 26.12 | 701.55 | 950.14 | 9.70 | 0.9665 | vLLM hits/queries |
| vllm_cache_off | context_compiler_no_compression | 32 | 8433.8 | 82.0 | 342.73 | 935.27 | 1161.76 | 1927.94 | 9.97 |  |  |
| vllm_cache_off | context_compiler_with_observation_compression | 32 | 1606.5 | 83.8 | 60.51 | 110.34 | 868.50 | 1035.78 | 9.66 |  |  |
| vllm_cache_off | dynamic_fields_last | 32 | 8433.8 | 76.8 | 343.34 | 943.95 | 1102.09 | 1920.67 | 9.88 |  |  |
| vllm_cache_off | original_bad_layout | 32 | 8497.8 | 89.6 | 346.71 | 948.07 | 1230.58 | 1932.61 | 9.86 |  |  |
| vllm_cache_off | stable_tool_order | 32 | 8433.8 | 90.2 | 343.97 | 943.05 | 1240.69 | 1927.15 | 9.95 |  |  |
| vllm_cache_off | truncation_baseline | 32 | 1629.8 | 70.4 | 61.09 | 135.04 | 743.38 | 1064.18 | 9.71 |  |  |

说明：vLLM 使用 prefix_cache_hits_total / prefix_cache_queries_total 的差值计算 token-level hit ratio。
SGLang 优先显示 cached_tokens_total / prompt_tokens_total 的差值比例；sglang:cache_hit_rate gauge 在本批实验中一直为 0，不作为主指标。
