# Context Compiler 服务器实验运行手册

更新时间：2026-06-08

适用版本：

```text
vLLM 0.10.1
SGLang 0.5.6
```

## 1. 本次改造目标

本次实验准备解决三个问题：

```text
1. 原 toy workload 太假，新增 realistic synthetic agent workload。
2. 原实验没有真实 backend metrics，新增 /metrics snapshot 和 delta 解析。
3. 原实验没有 warmup 和 per-variant 隔离，新增单 variant runner。
```

## 2. 新增文件

Realistic workload：

```text
projects/agent_infer_bench/agent_bench/workloads/generator.py
```

Metrics 解析：

```text
projects/agent_infer_bench/agent_bench/metrics/server_metrics.py
projects/agent_infer_bench/scripts/analysis/snapshot_server_metrics.py
```

Per-variant runner：

```text
projects/agent_infer_bench/scripts/benchmark/run_context_compiler_variant.py
```

实验配置：

```text
projects/agent_infer_bench/configs/context_compiler/realistic_vllm.yaml
projects/agent_infer_bench/configs/context_compiler/realistic_sglang.yaml
```

## 3. Realistic Workload 说明

新增 workload 类型：

```text
context_compiler_realistic_ablation
```

支持场景：

```text
function_calling
coding_agent
```

function_calling 场景包含：

```text
真实风格 system prompt
search_flights / get_weather / estimate_hotel_budget / save_itinerary 工具 schema
业务旅行查询
API JSON 风格 observation
```

coding_agent 场景包含：

```text
代码修复 system prompt
read_file / search_repo / run_tests / apply_patch 工具 schema
pytest failure observation
file_path / line_number / failing_test / exit_code / stderr_tail
```

每个场景支持 variants：

```text
original_bad_layout
stable_tool_order
dynamic_fields_last
context_compiler_no_compression
context_compiler_with_observation_compression
truncation_baseline
```

## 4. 启动 vLLM

推荐先用 vLLM 做第一轮。

prefix cache ON：

```bash
cd /path/to/ai_infra/projects/agent_infer_bench
MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.85 \
bash scripts/setup/start_vllm_prefix_cache_on.sh
```

prefix cache OFF：

```bash
cd /path/to/ai_infra/projects/agent_infer_bench
MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.85 \
bash scripts/setup/start_vllm_prefix_cache_off.sh
```

注意：

```text
start_vllm_prefix_cache_on.sh 默认端口是 8000。
start_vllm_prefix_cache_off.sh 默认端口是 8001。
realistic_vllm.yaml 默认访问 8000。
所以做 prefix cache OFF 对照时，建议显式指定 PORT=8000，或者把 config/server.base_url 改成 8001。
最终论文实验更推荐每次只启动一个 backend，并让 ON/OFF 都使用同一个端口，减少误连风险。
```

确认 metrics：

```bash
curl http://127.0.0.1:8000/metrics | head
```

重点搜索：

```bash
curl http://127.0.0.1:8000/metrics | grep -E "prefix_cache|time_to_first|prefill|latency"
```

## 5. 启动 SGLang

如果已有启动脚本：

```bash
cd /path/to/ai_infra/projects/agent_infer_bench
MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct \
bash scripts/setup/start_sglang_server.sh
```

确认 metrics：

```bash
curl http://127.0.0.1:30000/metrics | head
```

重点搜索：

```bash
curl http://127.0.0.1:30000/metrics | grep -E "cache_hit|prefix|latency|ttft"
```

## 6. 为什么需要 Warmup

需要 warmup。

原因：

```text
第一次请求会包含 tokenizer 初始化、CUDA kernel warmup、CUDA graph/engine warmup、内存分配等额外开销。
```

如果不 warmup：

```text
第一个 variant 可能被系统冷启动拖慢。
```

当前 runner 默认：

```text
--warmup-requests 4
```

warmup 不计入正式结果。

## 7. 如何清空 Cache

最稳妥方式：

```text
每个 variant 前重启 backend。
```

原因：

```text
vLLM/SGLang 的 prefix/KV cache 是服务进程内状态；
重启服务可以保证 cache 清空；
也能避免上一个 variant 污染下一个 variant。
```

推荐流程：

```text
1. 停止 backend。
2. 启动 backend。
3. 等待模型加载完成。
4. 跑 warmup。
5. 抓 before metrics。
6. 跑正式 workload。
7. 抓 after metrics。
8. 保存结果。
```

如果不想每次重启：

```text
可以使用 before/after metrics window 隔离统计。
```

但这不能清空 cache。

这种方式适合快速调试，不适合最终论文数字。

如果后端提供 flush endpoint：

```text
可以作为可选优化。
```

但最终论文建议仍以重启 backend 为准。

## 8. 单 Variant 运行命令

vLLM：

```bash
python scripts/benchmark/run_context_compiler_variant.py \
  --config configs/context_compiler/realistic_vllm.yaml \
  --variant original_bad_layout \
  --metrics-url http://127.0.0.1:8000/metrics \
  --warmup-requests 4
```

SGLang：

```bash
python scripts/benchmark/run_context_compiler_variant.py \
  --config configs/context_compiler/realistic_sglang.yaml \
  --variant original_bad_layout \
  --metrics-url http://127.0.0.1:30000/metrics \
  --warmup-requests 4
```

替换 variant：

```text
stable_tool_order
dynamic_fields_last
context_compiler_no_compression
context_compiler_with_observation_compression
truncation_baseline
```

## 9. 推荐最终运行顺序

每个 backend 分别跑：

```text
original_bad_layout
stable_tool_order
dynamic_fields_last
context_compiler_no_compression
context_compiler_with_observation_compression
truncation_baseline
```

每个 variant 前建议重启 backend。

如果时间紧，第一轮只跑：

```text
original_bad_layout
dynamic_fields_last
context_compiler_no_compression
context_compiler_with_observation_compression
truncation_baseline
```

## 10. 输出文件

每个 variant 会输出到：

```text
experiments/runs/context_compiler/realistic/vllm/<variant>/
experiments/runs/context_compiler/realistic/sglang/<variant>/
```

包含：

```text
results.json
repeat_summary.json
metrics_before.prom
metrics_after.prom
metrics_delta.json
warmup/results.json
```

## 11. vLLM Metrics 解释

优先查看：

```text
vllm:prefix_cache_hits
vllm:prefix_cache_queries
```

计算：

```text
hit_rate = Δprefix_cache_hits / Δprefix_cache_queries
```

runner 会写入：

```text
metrics_delta.json
```

字段可能包括：

```text
vllm_prefix_cache_hits_delta
vllm_prefix_cache_queries_delta
vllm_prefix_cache_hit_rate_delta
```

如果字段为 null：

```text
说明当前 metrics 输出中没有匹配到这些指标；
需要查看 metrics_before.prom / metrics_after.prom 的真实字段名。
```

## 12. SGLang Metrics 解释

优先查看：

```text
sglang:cache_hit_rate
```

runner 会尝试写入：

```text
sglang_cache_hit_rate
```

如果字段为 null：

```text
说明当前 metrics 输出中没有匹配到这些指标；
需要查看 metrics_before.prom / metrics_after.prom 的真实字段名。
```

## 13. 第一轮 Go / No-Go

Go 信号：

```text
compiler 提高 PSR/CRO，降低 DPR；
compiler 在 prefix cache on 时 TTFT/JCT 改善更明显；
backend-reported cache metrics 与 cacheability 指标方向一致；
dynamic_fields_last 明显优于 stable_tool_order；
truncation 虽快但缺少 recoverable/quality 保证。
```

No-Go 信号：

```text
compiler 和 dynamic_fields_last 完全一样；
prefix cache on/off 没有任何差异；
backend-reported cache metrics 完全不变；
realistic workload 上没有稳定趋势；
truncation 全面优于 compiler。
```

## 14. 论文中如何表述

如果拿到 backend 指标：

```text
We report backend-reported prefix cache metrics.
```

如果只拿到 latency proxy：

```text
When backend cache counters are unavailable, we report latency-based cache reuse proxies.
```

不要在没有指标时声称：

```text
actual KV cache hit rate
```

应写：

```text
cache reuse proxy
backend-reported prefix cache metric
```

## 15. 服务器跑前 Checklist

进入项目目录：

```bash
cd /path/to/ai_infra/projects/agent_infer_bench
```

确认代码结构：

```bash
ls configs/context_compiler
ls scripts/benchmark
ls scripts/setup
```

确认基础依赖：

```bash
python -c "import yaml, httpx; print('basic deps ok')"
python -c "import vllm; print('vllm ok')"
python -c "import sglang; print('sglang ok')"
```

确认模型路径：

```bash
ls /root/autodl-tmp/models/Qwen2.5-7B-Instruct
```

确认 server 可访问：

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/metrics | head
```

确认 metrics 字段：

```bash
curl http://127.0.0.1:8000/metrics | grep -E "prefix_cache|time_to_first|prefill|latency"
curl http://127.0.0.1:30000/metrics | grep -E "cache_hit|prefix|latency|ttft"
```

如果没有匹配到字段：

```text
先不要直接宣称真实 cache hit rate。
把 metrics_before.prom / metrics_after.prom 保存下来。
之后根据真实字段名修改 agent_bench/metrics/server_metrics.py 的解析规则。
```

建议先跑一个 smoke：

```bash
python scripts/benchmark/run_context_compiler_variant.py \
  --config configs/context_compiler/realistic_vllm.yaml \
  --variant context_compiler_with_observation_compression \
  --metrics-url http://127.0.0.1:8000/metrics \
  --warmup-requests 1
```

smoke 成功后再跑完整 variants：

```text
original_bad_layout
stable_tool_order
dynamic_fields_last
context_compiler_no_compression
context_compiler_with_observation_compression
truncation_baseline
```

每个正式 variant 前：

```text
1. 停止 backend。
2. 重新启动 backend。
3. 等待模型加载完成。
4. 跑对应 variant。
5. 保存该 variant 输出目录。
```

结果不要互相覆盖：

```text
ON 和 OFF 如果都使用 realistic_vllm.yaml，默认会写到同一批目录。
建议 OFF 对照显式加 --output-dir，例如：
experiments/runs/context_compiler/realistic/vllm_prefix_off/<variant>
```

OFF 对照示例：

```bash
PORT=8000 \
MODEL_PATH=/root/autodl-tmp/models/Qwen2.5-7B-Instruct \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.85 \
bash scripts/setup/start_vllm_prefix_cache_off.sh
```

```bash
python scripts/benchmark/run_context_compiler_variant.py \
  --config configs/context_compiler/realistic_vllm.yaml \
  --variant original_bad_layout \
  --metrics-url http://127.0.0.1:8000/metrics \
  --warmup-requests 4 \
  --output-dir experiments/runs/context_compiler/realistic/vllm_prefix_off/original_bad_layout
```
