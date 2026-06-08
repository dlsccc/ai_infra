# Context Compiler MVP Summary

| variant | requests | input tokens | PSR | DPR | CRO | prefix overlap |
|---|---:|---:|---:|---:|---:|---:|
| context_compiler_no_compression | 16 | 5383.0 | 0.692 | 0.300 | 0.575 | 0.467 |
| context_compiler_with_observation_compression | 16 | 4111.5 | 0.858 | 0.131 | 0.661 | 0.615 |
| dynamic_fields_last | 16 | 5383.0 | 0.690 | 0.302 | 0.714 | 0.583 |
| original_bad_layout | 16 | 5361.0 | 0.000 | 0.992 | 0.001 | 0.000 |
| stable_tool_order | 16 | 5383.0 | 0.000 | 0.992 | 0.001 | 0.001 |
| truncation_baseline | 16 | 4094.0 | 0.858 | 0.131 | 0.708 | 0.661 |

PSR = Prefix Stability Ratio; DPR = Dynamic Pollution Ratio; CRO = Cache Reuse Opportunity.

## Latency Summary

| variant | requests | backend input tokens | TTFT ms | latency ms |
|---|---:|---:|---:|---:|
| context_compiler_no_compression | 16 | 5383.0 | 193.30 | 284.94 |
| context_compiler_with_observation_compression | 16 | 4111.5 | 148.98 | 240.48 |
| dynamic_fields_last | 16 | 5383.0 | 193.35 | 284.35 |
| original_bad_layout | 16 | 5361.0 | 192.60 | 284.14 |
| stable_tool_order | 16 | 5383.0 | 193.35 | 283.89 |
| truncation_baseline | 16 | 4094.0 | 148.03 | 238.65 |
