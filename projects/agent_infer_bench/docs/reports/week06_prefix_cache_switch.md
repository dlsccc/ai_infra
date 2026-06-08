# Week 06 Prefix Cache 开关实验

## 1. 实验目的

本实验对应 Week 6 Day 4，目标是在可控 shared-prefix workload 下，观察：

1. vLLM 在 prefix cache 开启与关闭时的 `TTFT / total latency` 差异。
2. shared prefix length 从 `0 / 256 / 512 / 1024 / 2048` 增长时，缓存收益是否逐步放大。
3. SGLang 默认 radix cache 行为在同一组 workload 下的表现。

注意：本实验不直接声称“真实 cache hit rate”，除非 backend 显式暴露该指标。当前主要记录：

- shared prefix length
- 首轮与后续轮时延
- 同一 shared prefix 下的重复请求时延变化

## 2. 配置文件

- `configs/week06_prefix_cache_vllm_on.yaml`
- `configs/week06_prefix_cache_vllm_off.yaml`
- `configs/week06_prefix_cache_sglang.yaml`

## 3. 服务启动方式

### vLLM prefix cache 开启

```bash
bash scripts/setup/start_vllm_prefix_cache_on.sh
```

### vLLM prefix cache 关闭

```bash
bash scripts/setup/start_vllm_prefix_cache_off.sh
```

### SGLang 默认服务

```bash
bash scripts/setup/start_sglang_server.sh
```

## 4. 运行脚本

### vLLM 对照组

```bash
bash scripts/benchmark/run_week06_prefix_cache_vllm.sh
```

### SGLang 观测组

```bash
bash scripts/benchmark/run_week06_prefix_cache_sglang.sh
```

## 5. Workload 说明

本实验引入专门的 `shared_prefix_replay` workload：

1. 固定一段 shared prefix。
2. 使用两轮请求重复该 shared prefix。
3. 每轮 suffix 保持不同，以避免完全相同请求导致结果退化成“重复答同一句”。

这样可以更集中地观察 shared prefix 对流式时延的影响。

## 6. 预期分析重点

实验完成后重点回答：

1. shared prefix 越长，vLLM prefix cache 开启组是否出现更低的后续轮 TTFT。
2. vLLM prefix cache 关闭组是否失去这种趋势。
3. SGLang 默认 radix cache 行为是否呈现类似收益方向。
4. prefix cache 的收益是否主要体现在 prefill-heavy 部分，而不是 decode-heavy 部分。

## 7. 待补充内容

实验完成后补充：

- 汇总表格
- shared prefix length vs TTFT 图
- 首轮与后续轮对比图
- vLLM on/off 与 SGLang 默认行为的结论
