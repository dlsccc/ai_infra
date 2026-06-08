# Agent Context Compiler 三组最小验证实验计划

更新时间：2026-06-08

核心判断：
```text
cacheability 指标是否对应真实 serving 收益；
compiler 是否不伤工具任务质量；
recoverable observation compression 是否优于 truncation / generic summary。
```

## 0. 总体实验顺序

建议按以下顺序做：
```text
实验一：Synthetic + vLLM/SGLang cache validation
实验二：BFCL mini quality validation
实验三：SWE-Bench trace contract + recoverable compression validation
```

先用小规模跑通闭环。
每组实验都要产出一份 markdown report。
每组实验都要保存 raw results、config、summary table 和图。

## 1. 实验一：Synthetic + vLLM/SGLang Cache Validation
### 1.1 实验目的

验证：
```text
PSR / DPR / CRO 是否能预测真实 TTFT / JCT 改善。
```

证明：
```text
Context Compiler 不是指标自嗨，而是真的让 serving backend 更容易利用 prefix/prompt/KV cache。
```

### 1.2 核心问题

回答三个问题：

```text
Q1: dynamic pollution 是否会让 prefix cache 失效？
Q2: compiler 是否能恢复稳定 prefix reuse？
Q3: compiler + prefix cache 是否比 compiler alone / prefix cache alone 更好？
```

### 1.3 使用 workload

使用当前已有配置：

```text
projects/agent_infer_bench/configs/context_compiler/mvp_mock.yaml
```

包含 variants：

```text
original_bad_layout
stable_tool_order
dynamic_fields_last
context_compiler_no_compression
context_compiler_with_observation_compression
truncation_baseline
```

### 1.4 需要新增配置

新增两个 vLLM 配置：

```text
configs/context_compiler/realistic_vllm.yaml
vLLM prefix cache on/off 通过不同 server 启动脚本控制。
```

新增一个 SGLang 配置：

```text
configs/context_compiler/realistic_sglang.yaml
```

如果时间紧，先做 vLLM prefix on/off。

### 1.5 模型选择

首选：

```text
Qwen2.5-7B-Instruct
```

备选：

```text
Qwen3-8B
Llama-3.1-8B-Instruct
```

第一轮不要上 32B。

先验证趋势。

### 1.6 运行设置

建议参数：

```text
turns: 4
count: 8 或 16
tool_count: 8
tool_tokens: 80
observation_tokens: 256 / 512 / 1024
concurrency: 1 / 8
repeat: 3
temperature: 0
max_tokens: 64
```

先跑小矩阵：

```text
observation_tokens: [256, 1024]
concurrency: [1, 8]
```

### 1.7 指标

Cacheability 指标：

```text
PSR
DPR
CRO
prefix_overlap_ratio
input_tokens
```

Serving 指标：

```text
TTFT
JCT / total latency
P50 latency
P95 latency
tokens/s
requests/s
```

如果 backend 暴露：

```text
prefix cache hit rate
cached token count
prefill tokens saved
```

如果 backend 不暴露，不要声称真实 cache hit。

使用 proxy：

```text
prefix cache on/off latency delta
repeated stable-prefix latency delta
```

### 1.8 需要对比的组合

最小组合：

```text
Original + vLLM prefix cache off
Original + vLLM prefix cache on
Compiler + vLLM prefix cache off
Compiler + vLLM prefix cache on
```

推荐组合：

```text
All variants + vLLM prefix off
All variants + vLLM prefix on
All variants + SGLang
```

### 1.9 预期图表

图 1：

```text
variant vs PSR / DPR / CRO
```

图 2：

```text
variant vs TTFT under prefix cache on/off
```

图 3：

```text
CRO vs TTFT improvement scatter plot
```

图 4：

```text
Compiler alone / Prefix cache alone / Compiler + Prefix cache
```

### 1.10 Go 条件

继续推进的条件：

```text
compiler 显著提高 PSR/CRO，降低 DPR；
prefix cache on 时 compiler 的 TTFT/JCT 收益更明显；
stable_tool_order alone 明显弱于 dynamic_fields_last 或 compiler；
compiler_with_compression 不能只因为 token 少而胜出。
```

### 1.11 No-Go 信号

需要警惕：

```text
PSR/CRO 提升但 TTFT/JCT 完全不变；
compiler 与 dynamic_fields_last 几乎一样；
truncation_baseline 全面优于 compiler；
结果只在 mock backend 上成立。
```

### 1.12 产出文件

建议产出：

```text
experiments/runs/context_compiler/realistic/vllm/
experiments/runs/context_compiler/realistic/sglang/
docs/research/reports/exp1_cache_validation.md
```

## 2. 实验二：BFCL Mini Quality Validation

### 2.1 实验目的

验证：

```text
Context Compiler 不会破坏 function calling 质量。
```

重点证明：

```text
tool schema canonicalization 和 layout repair 不伤 JSON validity / tool accuracy。
```

### 2.2 为什么先做 BFCL

BFCL 适合作为第一组真实任务，因为：

```text
质量指标清晰；
tool schema 是核心上下文；
JSON/tool accuracy 容易自动评估；
实验成本低于 SWE-Bench。
```

### 2.3 数据选择

先做 mini set：

```text
50-100 条样本
```

覆盖：

```text
simple function call
multiple function call
parallel / nested function call 可选
```

工具数量分层：

```text
tool_count: 5 / 10 / 20
```

第一轮不追求完整 BFCL。

先追求 pipeline 和趋势。

### 2.4 方法对比

必须对比：

```text
original
stable_tool_order
dynamic_fields_last
truncation_baseline
context_compiler_no_compression
context_compiler_with_contract_guard
```

可选对比：

```text
generic_summarization
LLMLingua-style compression
Don't Break-style boundary
```

第一轮可以不做 LLMLingua。

### 2.5 质量指标

必须计算：

```text
JSON validity
tool name accuracy
argument key accuracy
required field recall
argument value exact match
execution success proxy
```

如果 BFCL 官方 evaluator 可用，优先用官方 evaluator。

如果暂时不用官方 evaluator，先实现简化 evaluator：

```text
解析模型输出 JSON；
比较 expected tool name；
比较 required argument keys；
比较 argument values。
```

### 2.6 Serving 指标

同时记录：

```text
input_tokens
TTFT
JCT / total latency
P95 latency
PSR
DPR
CRO
```

### 2.7 关键实验问题

回答：

```text
Q1: canonical tool schema 是否改变模型 tool selection？
Q2: layout repair 是否影响 JSON validity？
Q3: compression/truncation 是否删除 required fields？
Q4: compiler 是否在质量基本不降的情况下提升 cacheability？
```

### 2.8 Go 条件

继续推进的条件：

```text
compiler quality >= original - 1% 到 2%；
compiler quality 明显高于 truncation/generic summary；
compiler PSR/CRO 明显高于 original；
compiler 在 prefix cache on 下 TTFT/JCT 有收益。
```

### 2.9 No-Go 信号

需要警惕：

```text
canonicalization 后 tool accuracy 明显下降；
compiler 和 stable_tool_order 差不多；
truncation 质量不掉且成本更低；
模型输出格式太不稳定，导致无法评估。
```

### 2.10 产出图表

图 1：

```text
method vs JSON validity / tool accuracy
```

图 2：

```text
method vs input tokens / TTFT
```

图 3：

```text
quality-cost frontier
```

图 4：

```text
tool_count 变化下的 quality 和 cacheability
```

### 2.11 产出文件

建议产出：

```text
experiments/runs/week08/bfcl_mini/
docs/research/reports/exp2_bfcl_quality_validation.md
```

## 3. 实验三：SWE-Bench Trace Contract + Recoverable Compression Validation

### 3.1 实验目的

验证：

```text
Observation Contract 不是拍脑袋；
Recoverable compression 比 truncation / generic summary 更适合长 observation 工具任务。
```

### 3.2 先做 trace-level 实验

第一轮不要完整跑 SWE-Bench agent。

先做 trace-level / observation-level 实验。

目标是验证：

```text
哪些 observation 字段后续真的被使用；
哪些字段必须保留；
哪些字段可以外部化为 pointer；
哪些字段可以丢弃。
```

### 3.3 数据规模

第一轮：

```text
50 条 coding-agent traces
```

如果没有真实 agent trace：

```text
用 SWE-Bench Lite issue + repo snippets + pytest logs 构造 pseudo traces
```

每条 trace 至少包含：

```text
user issue
repo search result
file snippets
test failure log
tool call history
assistant next action or final patch proxy
```

### 3.4 Observation Contract 初版

Coding observation must preserve：

```text
file path
line number
symbol name
error type
failing test name
exit code
stack trace root cause
patch-relevant snippet
```

Recoverable：

```text
full stdout
full stderr
full file snippet
full test log
full search result list
```

Discardable：

```text
progress bars
duplicated warnings
install logs
repeated stack frames
irrelevant file lists
```

### 3.5 数据支撑 contract

做一个轻量引用率分析。

步骤：

```text
1. 从 observation 中抽取字段。
2. 在后续 assistant output / tool call / patch proxy 中查找字段是否被引用。
3. 统计每类字段的 future-use rate。
```

字段类别：

```text
file_path
line_number
symbol_name
error_message
failing_test
exit_code
full_stderr_tail
progress_bar
duplicated_warning
```

预期表：

```text
field_type | count | future_use_rate | contract_class
```

例子：

```text
file_path | 50 | 0.84 | must_preserve
error_message | 50 | 0.76 | must_preserve
full_stderr_body | 50 | 0.12 | recoverable
progress_bar | 20 | 0.00 | discardable
```

### 3.6 压缩方法对比

比较：

```text
full_observation
truncation_head
truncation_tail
generic_summary
contract_aware_summary
contract_aware_summary_plus_pointer
```

第一轮 `generic_summary` 可以用规则摘要或调用模型摘要。

如果用模型摘要，固定 temperature = 0。

### 3.7 Recoverable Runtime MVP

实现最小 ObservationStore：

```text
put(raw_observation) -> obs_id
get(obs_id) -> raw_observation
retrieve(obs_id, query) -> relevant span
```

第一版可以用本地 JSON 文件存储。

第一版 retrieval 可以用：

```text
keyword match
field lookup
line range lookup
```

不需要一开始做 embedding retrieval。

### 3.8 质量任务

先做可自动评估的小任务：

```text
从压缩 observation 中恢复正确 file path；
识别 failing test；
识别 error type；
选择最相关 code snippet；
判断下一步应该 inspect 哪个文件。
```

质量指标：

```text
field recall
contract violation rate
next-action accuracy
recovery success rate
```

### 3.9 Serving 指标

记录：

```text
input_tokens
TTFT
JCT
P95 latency
PSR
DPR
CRO
recovery extra tokens
recovery extra latency
```

### 3.10 Go 条件

继续推进的条件：

```text
contract-aware summary 的 contract violation rate 低于 truncation/generic summary；
summary + pointer 的 input tokens 显著低于 full observation；
需要恢复时 retrieve_observation 能恢复关键字段；
总体质量高于 truncation，成本低于 full observation。
```

### 3.11 No-Go 信号

需要警惕：

```text
generic summary 已经足够好；
contract-aware summary 与 truncation 差别不大；
retrieve_observation 很少有用；
recoverable runtime 带来的额外调用成本过高；
field future-use analysis 没有明显模式。
```

### 3.12 产出图表

图 1：

```text
field_type vs future_use_rate
```

图 2：

```text
method vs contract violation rate
```

图 3：

```text
method vs token cost / quality
```

图 4：

```text
recovery trigger / recovery success / failure rescue cases
```

### 3.13 产出文件

建议产出：

```text
experiments/runs/week08/swe_trace_contract/
docs/research/reports/exp3_swe_contract_recoverable_compression.md
```

## 4. 三组实验后的决策表

完成三组实验后，填这个表：

| 判断问题 | Go 条件 | 结果 |
|---|---|---|
| cacheability 指标是否对应真实 TTFT/JCT？ | 是 | TBD |
| compiler 是否优于 stable_tool_order / dynamic_fields_last？ | 是 | TBD |
| compiler 是否质量不掉？ | 是 | TBD |
| recoverable compression 是否优于 truncation？ | 是 | TBD |
| compiler + serving cache 是否有组合增益？ | 是 | TBD |

如果 5 个问题中至少 4 个为 Go：

```text
继续扩展完整论文实验矩阵。
```

如果只有 2-3 个为 Go：

```text
降级为 workshop / arXiv / 技术报告。
```

如果少于 2 个为 Go：

```text
停止主线，转向 benchmark/characterization 项目。
```

## 5. 推荐时间线

Day 1-2：

```text
跑 vLLM prefix cache on/off。
```

Day 3：

```text
整理实验一报告。
```

Day 4-6：

```text
接 BFCL mini evaluator。
```

Day 7：

```text
整理实验二报告。
```

Day 8-10：

```text
做 SWE trace contract analysis。
```

Day 11-12：

```text
实现 ObservationStore + retrieve_observation MVP。
```

Day 13-14：

```text
整理三组实验 Go / No-Go 总结。
```
