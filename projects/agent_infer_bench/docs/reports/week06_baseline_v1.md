# Week 06 Baseline v1

## 目的

本报告记录 `AgentInferBench` 第一版可控 baseline 对比实验，目标是在相同 synthetic workload 下比较 vLLM 和 SGLang 的行为。

这一阶段的目标是：

1. 验证两个 backend 都能跑通同一组 workload。
2. 观察不同 workload 下的 token 长度和延迟变化。
3. 找出当前对比中仍然不公平或不完整的地方。

## 范围

本报告重点回答：

1. `plain_chat` 和 Agent 风格 workload 有什么差异。
2. 多轮 history 和长 observation 会如何影响输入长度和整体延迟。
3. 当前 vLLM 和 SGLang 的实验链路是否已经足够支撑 Week 7 更深入的 prefix/cache 分析。

本报告不应直接宣称：

1. 某个框架一定全面优于另一个框架。
2. 当前吞吐结果已经代表生产级性能。
3. 当前结果可以代表真实 TTFT。
4. 当前结果已经达到公开 benchmark 或论文级结论。

## Backends

| Backend | 状态 | 说明 |
|---|---|---|
| vLLM | completed | 离线 `LLM.generate()` 路径可用；当前 token 统计来自 backend token ids |
| SGLang | completed | 离线 `Engine.generate()` 路径可用；当前 token 统计来自 tokenizer fallback |

## Workloads

| Workload | 描述 | 测试目的 |
|---|---|---|
| `plain_chat` short | 最小普通聊天 prompt | 作为最基础的 sanity baseline |
| `plain_chat` medium | 更长的普通聊天 prompt | 观察 prompt 长度增长的影响 |
| `single_tool` | 单轮工具调用风格 prompt | 观察 system/tool prefix 成本 |
| `multi_tool_serial` | 多轮 Agent trace | 观察 history 增长 |
| `long_observation` | 带更长 observation 的多轮 Agent trace | 观察 prefill-heavy 压力 |

## 配置

本次实验使用的配置文件：

- `configs/baseline_v1_vllm.yaml`
- `configs/baseline_v1_sglang.yaml`

重跑前本地做过的配置调整：

| 项目 | 值 |
|---|---|
| vLLM config revision | baseline v1 |
| SGLang config revision | baseline v1 |
| Repeat count | 3 |
| Model path | `/root/autodl-tmp/models/Qwen2.5-7B-Instruct` |
| `max_tokens` | 64 |
| Context limit | 4096 |

## 环境快照

| 项目 | vLLM | SGLang |
|---|---|---|
| Python | 3.11.15 | 3.11.15 |
| Torch |  |  |
| Torch CUDA |  |  |
| Backend version | vLLM | SGLang |
| GPU |  |  |
| Driver CUDA |  |  |
| Token source summary | input/output both `backend_token_ids` | input/output both `tokenizer_fallback` |

## 原始结果路径

| Backend | 原始结果目录 | Summary 文件 | Repeat Summary |
|---|---|---|---|
| vLLM | `experiments/runs/week06/vllm_baseline_v1/` | `summary.md` | `repeat_summary.json` |
| SGLang | `experiments/runs/week06/sglang_baseline_v1/` | `summary.md` | `repeat_summary.json` |

## 结果汇总

| Backend | Workload | 实际输入 tokens | 实际输出 tokens | Mean latency (ms) | P95 latency (ms) | Tokens/s | Token source | 说明 |
|---|---|---:|---:|---:|---:|---:|---|---|
| vLLM | plain_chat short | 319 | 64 | 172.01 | 172.01 | 372.07 | backend token ids | 当前 request 级延迟是 batch 平均值 |
| vLLM | plain_chat medium | 1435 | 64 | 172.01 | 172.01 | 372.07 | backend token ids | 当前 request 级延迟是 batch 平均值 |
| vLLM | single_tool | 1095 | 64 | 172.01 | 172.01 | 372.07 | backend token ids | 当前 request 级延迟是 batch 平均值 |
| vLLM | multi_tool_serial | 1095 / 1354 | 64 / 64 | 172.01 | 172.01 | 372.07 | backend token ids | 共 2 轮；当前 request 级延迟是 batch 平均值 |
| vLLM | long_observation | 1095 / 1854 | 64 / 64 | 172.01 | 172.01 | 372.07 | backend token ids | 共 2 轮；当前 request 级延迟是 batch 平均值 |
| SGLang | plain_chat short | 319 | 64 | 157.49 | 157.49 | 406.38 | tokenizer fallback | 当前 request 级延迟是 batch 平均值 |
| SGLang | plain_chat medium | 1435 | 64 | 157.49 | 157.49 | 406.38 | tokenizer fallback | 当前 request 级延迟是 batch 平均值 |
| SGLang | single_tool | 1095 | 64 | 157.49 | 157.49 | 406.38 | tokenizer fallback | 当前 request 级延迟是 batch 平均值 |
| SGLang | multi_tool_serial | 1095 / 1354 | 64 / 64 | 157.49 | 157.49 | 406.38 | tokenizer fallback | 共 2 轮；当前 request 级延迟是 batch 平均值 |
| SGLang | long_observation | 1095 / 1854 | 64 / 64 | 157.49 | 157.49 | 406.38 | tokenizer fallback | 共 2 轮；当前 request 级延迟是 batch 平均值 |

## Repeat 稳定性

下面的数据来自 `repeat_summary.json`。

| Backend | 每轮 mean latency | Mean latency mean | Mean latency std | 说明 |
|---|---|---:|---:|---|
| vLLM | 223.52, 172.17, 172.01 | 189.23 | 24.24 | 第一轮较慢，后两轮趋于稳定 |
| SGLang | 430.42, 225.21, 157.49 | 271.04 | 116.04 | warmup 开销更明显，波动更大 |

## 初步观察

### 1. Plain Chat vs Agent-Style Workloads

- 实际输入 token 数从 `plain_chat short` 到 `plain_chat medium` 明显增加，进入 `single_tool` 后仍然维持在较高水平。
- `single_tool` 的输入已经明显大于 `plain_chat short`，说明 system prompt 和 tool schema 本身就会显著抬高输入长度。
- `long_observation` turn 1 达到本轮实验里最大的输入长度 `1854`，说明工具返回内容会明显推动上下文膨胀。

### 2. History 增长

- `multi_tool_serial` 从 turn 0 的 `1095` input tokens 增长到 turn 1 的 `1354`。
- 这说明即使只有两轮，Agent 的 history 累积也已经足以显著拉长 prompt。
- 由于当前 request 级 latency 仍然是 batch 平均值，暂时不能直接从 latency 上看到这种增长，但 token 增长本身已经是可信信号。

### 3. Long Observation 的影响

- `long_observation` 从 turn 0 的 `1095` 增长到 turn 1 的 `1854`，明显高于 `multi_tool_serial` turn 1 的 `1354`。
- 这支持一个很重要的判断：在 Agent 场景里，长工具返回是 prompt 膨胀的重要来源。
- 两个 backend 都观察到了相同的 token 增长模式，因为它们共享同一套 synthetic workload。

### 4. Backend 对比

- 按当前 batch 级 summary 看，SGLang 在最终 selected run 上略快于 vLLM：`157.49 ms` vs `172.01 ms`。
- 这个差异还不能直接解释成“框架性能结论”。
- 当前 token 统计并不完全对称：vLLM 使用 backend token ids，SGLang 使用 tokenizer fallback。
- 更关键的是，两个 backend 现在都把整个 batch 的总耗时平均分配给每条 request，因此 request 级延迟比较还不可靠。

## 问题与异常

| 问题 | Backend | 现象 | 怀疑原因 | 已采取动作 |
|---|---|---|---|---|
| 同一轮内每条 request latency 完全相同 | vLLM / SGLang | 每条 request 的 `total_latency_ms` 都一样 | 当前统计口径是 batch 总耗时均摊到每条 request | 在报告中明确标注限制，后续改进统计方式 |
| Token source 不对称 | vLLM / SGLang | vLLM 用 backend token ids，SGLang 用 tokenizer fallback | 两个 backend 暴露 token 细节的方式不同 | 已把 token source 写进 metadata 和 summary |
| Warmup 波动较大 | 尤其 SGLang | 第一轮明显慢于后续轮次 | 初始化和 cache warmup 开销较大 | 保留 repeat summary，避免只看第一轮 |

## 当前限制

1. 当前离线 `generate()` 方式不能给出真实 streaming TTFT。
2. 配置里的 token budget 仍然是 workload 构造目标，不是严格 tokenizer 保证。
3. token 统计已经比 Week5 更可信，但两边仍然没有做到完全对称。
4. 当前 request 级 latency 不是真实单请求 wall-clock latency，而是 batch 总耗时均摊值。
5. 这仍然是一版 synthetic controlled baseline，不是公开 benchmark 结果。

## 当前阶段结论

1. Week6 baseline v1 已经成功让 vLLM 和 SGLang 跑通同一组 controlled workloads。
2. Agent 风格 workload 在当前实验里已经明显比 plain chat 拥有更长的输入上下文。
3. 长 observation 是 prompt 增长的重要来源，后续应作为 context management 的重点问题处理。
4. 当前 token 统计已经能支持 workload-level 的分析，但 backend 统计口径仍需继续统一。
5. 当前 latency 结果更适合做 batch-level 观察，不适合做 request-level 精细比较。

## 下一步

Week 7 更适合进入：

1. prefix/cache 相关 workload 深挖；
2. prompt layout / tool schema 稳定性 / session locality 实验；
3. 更进一步统一 token 和 latency 统计口径。
