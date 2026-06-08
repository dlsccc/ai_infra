# Agent Context Compiler：面向工具调用智能体的上下文编译器设计草案

更新时间：2026-06-04

## 1. 一句话概括

Agent Context Compiler 的目标是：

```text
把原始、混乱、动态增长的 Agent 上下文，编译成稳定、分层、可压缩、可恢复、并且更适合 prompt cache / prefix cache / LLM serving 的上下文布局。
```

它不是 KV cache manager，也不是普通 prompt compression。它位于：

```text
Agent framework 之后，LLM serving backend 之前。
```

普通 Agent 框架通常直接拼接：

```text
system prompt + tool schemas + memory + history + observations + current query
```

Agent Context Compiler 则先将这些内容解析成结构化片段，判断每个片段的稳定性、复用范围、压缩风险和质量约束，再生成一个更 cache-friendly 的 prompt。

## 2. 为什么需要上下文编译器

Tool-using Agent 的上下文有几个典型问题：

1. system prompt 和 tool schemas 很稳定，但经常被动态字段污染，无法形成稳定 prefix。
2. timestamp、session id、retry count、request id 等高熵字段常常被放在 prompt 开头，直接破坏 prefix cache。
3. observation 会随工具调用快速膨胀，导致 prefill cost 和 TTFT 增长。
4. tool schema 顺序、JSON 序列化格式、history 模板不稳定，会让语义相同的上下文在文本层面不一致。
5. 普通 truncation 或 summarization 虽然能减少 token，但可能删除工具调用所需的关键字段，导致任务质量下降。

核心观点：

```text
Cacheability is not only a serving-system property, but also an agent-design property.
```

中文：

```text
上下文能不能被缓存，不只取决于 vLLM/SGLang/OpenAI API 如何管理 cache，也取决于 Agent 一开始如何构造上下文。
```

## 3. 系统定位

Agent Context Compiler 的输入：

```text
agent instruction
developer/system prompt
tool schemas
long-term memory
short-term memory
conversation history
tool call history
tool observations
current user query
runtime metadata
quality constraints
serving hints
```

Agent Context Compiler 的输出：

```text
compiled prompt
segment metadata
cache keys
reuse scopes
compression decisions
recoverable references
quality guard report
```

它可以接在不同 Agent 框架后面：

```text
LangChain / AutoGen / CrewAI / SWE-Agent / 自研 Agent framework
```

也可以接不同 serving backend：

```text
vLLM / SGLang / OpenAI-compatible API / Anthropic / Gemini
```

## 4. 核心抽象：Context Segment

编译器不把 prompt 当成一整段字符串，而是把它拆成多个 segment。

推荐 segment schema：

```json
{
  "name": "tool_schema_block",
  "text": "...",
  "segment_type": "tool_schema",
  "stability": "static",
  "reuse_scope": "agent_type",
  "cache_key": "sha256:...",
  "must_preserve": ["tool_name", "argument_schema", "required_fields"],
  "source_pointer": null
}
```

关键字段解释：

| 字段 | 含义 |
|---|---|
| `name` | segment 名称，例如 system、tools、observation_03 |
| `text` | segment 原始文本或压缩后文本 |
| `segment_type` | system、tool_schema、memory、history、observation、current_query、runtime |
| `stability` | static、semi_static、session_dynamic、turn_dynamic、ephemeral |
| `reuse_scope` | global、agent_type、session、turn、request |
| `cache_key` | 用于追踪稳定片段的确定性 hash |
| `must_preserve` | 压缩或重排时必须保留的信息 |
| `source_pointer` | 可恢复压缩时指向原始内容的位置 |

## 5. Segment 稳定性分类

| 稳定性 | 示例 | 复用范围 | 推荐处理 |
|---|---|---|---|
| static | system prompt、agent role、固定 tool schemas | 跨请求/跨用户 | 放在最前面，固定格式 |
| semi_static | repo summary、task plan、long-term memory summary | 跨 session 或长周期 | 版本化，生成 cache key |
| session_dynamic | conversation history、short-term memory、之前 tool call 摘要 | session 内复用 | 结构化、可分块 |
| turn_dynamic | current query、current observation、当前 tool result | 当前轮 | 放在后部，必要时压缩 |
| ephemeral | timestamp、session id、retry count、random id | 单次请求 | 后置或移出 prompt |

基本原则：

```text
越稳定、越可能复用的内容越靠前；
越动态、越一次性的内容越靠后；
能变成 metadata 的内容不要污染 prompt prefix。
```

## 6. 编译器 Pass 设计

Agent Context Compiler 可以设计成多个 pass，类似传统编译器。

### 6.1 Pass 1: Segment Parsing

目标：

```text
把原始 Agent 状态解析成结构化 context segments。
```

输入示例：

```text
system prompt
tools list
chat history
tool results
current query
runtime metadata
```

输出示例：

```text
system segment
tool_schema segment
history segments
observation segments
current_query segment
runtime segment
```

这一层可以先靠规则实现，后续再适配 LangChain / AutoGen / SWE-Agent 的结构化对象。

### 6.2 Pass 2: Stability Analysis

目标：

```text
判断每个 segment 是否稳定、是否值得放在 prefix、是否会污染 cache。
```

判断问题：

1. 这段是否跨请求相同？
2. 这段是否跨 turn 相同？
3. 这段是否只在当前 turn 有用？
4. 这段是否包含高熵字段？
5. 这段是否应该从 prompt 中移到 metadata？
6. 这段是否能生成稳定 cache key？

输出：

```text
stability label
reuse_scope
cache_key
pollution risk
```

### 6.3 Pass 3: Canonicalization

目标：

```text
让语义相同的上下文在文本层面也尽可能一致。
```

策略：

1. tool schema 使用 deterministic JSON serialization。
2. JSON key 固定排序。
3. tool list 固定排序，例如按 name、namespace、version。
4. 删除无意义随机空格和不稳定模板。
5. 固定 history 模板。
6. 固定 system/tool/memory 的 section header。

示例：

```text
bad:
{"description": "...", "name": "search", "parameters": {...}}

better:
{"description":"...","name":"search","parameters":{...}}
```

### 6.4 Pass 4: Layout Planning

目标：

```text
重新排列 segment，使稳定 prefix 最大化，动态污染最小化。
```

推荐布局：

```text
[SYSTEM]
[TOOL_SCHEMAS]
[STABLE_MEMORY]
[SESSION_SUMMARY]
[RECENT_HISTORY]
[COMPRESSED_OBSERVATIONS]
[CURRENT_QUERY]
[RUNTIME_METADATA]
Assistant:
```

不推荐布局：

```text
[TIMESTAMP]
[SESSION_ID]
[CURRENT_QUERY]
[OBSERVATION]
[TOOL_SCHEMAS]
[SYSTEM]
Assistant:
```

核心原则：

```text
dynamic fields last
stable tools first
observations compressed and recoverable
history summarized or segmented
```

### 6.5 Pass 5: Tool-Aware Observation Compression

目标：

```text
减少 observation token，同时保留任务成功所需的信息。
```

这不是普通摘要，而是根据工具类型做保真压缩。

#### Function calling

必须保留：

```text
tool name
argument names
argument types
required fields
enum values
execution status
```

#### Coding agent

必须保留：

```text
file path
symbol name
line number
error message
stack trace root cause
test failure
patch-relevant snippet
```

#### Retrieval / research agent

必须保留：

```text
evidence sentence
source title
url/doc id
claim relation
conflicting evidence
```

#### Web agent

必须保留：

```text
current goal
page state
visible interactive elements
last action
error feedback
```

### 6.6 Pass 6: Recoverable Reference Generation

目标：

```text
压缩不是不可逆删除，而是 summary + pointer。
```

示例：

```text
[OBSERVATION: pytest_result]
Tests failed.
Key error: AssertionError in tests/test_parser.py::test_nested_call
Last 20 stderr lines retained.
Full output recoverable at obs://run_42/pytest_stderr
```

好处：

1. 降低当前 prompt 成本。
2. 保留恢复完整 observation 的能力。
3. 避免一次压缩错误永久损害任务。
4. 更适合 long-horizon agent。

### 6.7 Pass 7: Quality Guard

目标：

```text
在输出 compiled prompt 前检查关键字段是否被误删。
```

检查规则：

1. tool schema 是否仍包含 required fields。
2. JSON schema 是否仍合法。
3. coding observation 是否仍包含 file path、line number、error message。
4. retrieval observation 是否仍包含 evidence 和 source。
5. runtime metadata 是否没有污染 prefix。
6. 压缩比例是否超过任务允许阈值。

## 7. Cacheability 指标体系

### 7.1 Prefix Stability Ratio

```text
PSR = stable_prefix_tokens / total_input_tokens
```

含义：

```text
输入里有多少 token 位于稳定 prefix 中。
```

### 7.2 Dynamic Pollution Ratio

```text
DPR = dynamic_tokens_before_stable_boundary / total_input_tokens
```

含义：

```text
有多少动态 token 污染了可缓存前缀。
```

### 7.3 Cache Reuse Opportunity

```text
CRO(i, j) = longest_common_prefix(prompt_i, prompt_j) / min(len(prompt_i), len(prompt_j))
```

含义：

```text
两个请求之间理论上有多少 prefix 可以复用。
```

### 7.4 Canonicalization Gain

```text
CG = prefix_overlap_after_compiler - prefix_overlap_before_compiler
```

含义：

```text
编译器让可复用 prefix 增加了多少。
```

### 7.5 Quality-Preserved Cost Reduction

```text
QPCR = cost_reduction under quality_drop <= epsilon
```

示例：

```text
在 tool accuracy 下降不超过 1% 的前提下，TTFT 降低多少？
在 task success rate 下降不超过 2% 的前提下，input token cost 降低多少？
```

## 8. 质量约束指标

不同任务需要不同的质量约束。

### 8.1 BFCL / Function Calling

质量指标：

1. JSON validity。
2. tool name accuracy。
3. argument key accuracy。
4. argument value accuracy。
5. execution success proxy。

### 8.2 SWE-Bench Lite / Coding Agent

质量指标：

1. patch applies。
2. test pass rate。
3. file localization accuracy。
4. edit localization accuracy。
5. tool call correctness。
6. error diagnosis preservation。

### 8.3 Retrieval / HoVer / Deep Research

质量指标：

1. answer correctness。
2. evidence recall。
3. citation accuracy。
4. fact verification accuracy。
5. contradiction preservation。

### 8.4 Web Agent / VisualWebArena

质量指标：

1. task success。
2. action accuracy。
3. loop rate。
4. page state consistency。

由于 PANDO 已经覆盖了 cache-aware prompting + skill distillation 的 Web Agent 场景，Web Agent 更适合作为补充实验，不建议作为主战场。

## 9. Baseline 设计

最小闭环 baseline：

1. Original prompt。
2. Stable tool ordering。
3. Dynamic fields last。
4. Truncation baseline。
5. Generic summarization baseline。
6. Agent Context Compiler without compression。
7. Agent Context Compiler with recoverable observation compression。

Serving baseline：

1. vLLM prefix cache off。
2. vLLM prefix cache on。
3. SGLang radix cache。
4. 如果后续需要，再加 session-sticky routing / prefix-hash routing。

注意：

```text
routing 不应成为主贡献，否则会撞 KVFlow/PBKV/Preble。
```

## 10. 最小可行研究闭环

### 10.1 第一阶段：Synthetic Controlled Workload

目标：

```text
验证 compiler 是否真的改善 cacheability 指标。
```

任务：

1. 构造 original_bad_layout。
2. 构造 stable_tool_order。
3. 构造 dynamic_fields_last。
4. 构造 context_compiler_no_compression。
5. 构造 context_compiler_with_observation_compression。
6. 构造 truncation_baseline。

验收：

1. Compiler 的 PSR 显著高于 original。
2. Compiler 的 DPR 显著低于 original。
3. Compiler 的 CRO 高于 original。
4. Compiler 不只是减少 token，还改善前缀结构。

### 10.2 第二阶段：Serving Latency

目标：

```text
验证 cacheability 指标提升是否能转化为 TTFT/JCT 收益。
```

实验：

1. vLLM prefix cache off。
2. vLLM prefix cache on。
3. SGLang。

关键观察：

```text
如果 prefix cache on 时 compiler 收益更明显，说明它确实改善了 cache 可利用性。
```

### 10.3 第三阶段：真实任务质量闭环

优先任务：

```text
BFCL / function calling mini set
```

原因：

1. 质量指标清楚。
2. 工具 schema 是核心上下文。
3. JSON validity 和 tool accuracy 容易自动评估。
4. 比 SWE-Bench 更适合作为第一真实闭环。

下一步任务：

```text
SWE-Bench Lite / coding agent traces
```

原因：

1. observation 更长。
2. 更能体现 recoverable compression 的价值。
3. 更有论文说服力。

## 11. 与已有工作的区别

### 11.1 与 KVFlow / PBKV

KVFlow/PBKV 做：

```text
根据 workflow 或预测结果管理已经生成的 KV cache。
```

Agent Context Compiler 做：

```text
在 KV cache 生成之前，优化 Agent 应该生成什么样的上下文。
```

### 11.2 与 PANDO

PANDO 做：

```text
面向 multimodal web agent 的 online skill distillation 和 cache-aware prompting。
```

Agent Context Compiler 做：

```text
面向通用 tool-using agent 的结构化上下文编译。
```

避免重合的方法：

1. 不主打 WebArena。
2. 不做 skill distillation。
3. 主打多类工具任务和通用 segment compiler。

### 11.3 与 ACE

ACE 做：

```text
通过执行反馈动态改进 Agent 上下文，提高任务成功率。
```

Agent Context Compiler 做：

```text
根据 cacheability 和质量约束编译上下文，提高 serving efficiency。
```

### 11.4 与 Prompt Compression

Prompt compression 做：

```text
减少 token。
```

Agent Context Compiler 做：

```text
减少不必要 token，同时提高 prefix/cache 可利用性，并保证工具任务质量。
```

### 11.5 与 ToolCaching / TVCache

ToolCaching / TVCache 做：

```text
缓存工具调用结果或工具状态。
```

Agent Context Compiler 做：

```text
不复用工具结果，只压缩、引用和重排 observation。
```

## 12. 论文贡献表述

可以写成：

```text
We propose Agent Context Compiler, a framework that transforms raw tool-using agent contexts into cache-friendly layouts under task-specific quality constraints.
```

中文：

```text
我们提出 Agent Context Compiler，一个在任务质量约束下，将原始工具调用智能体上下文转换为缓存友好布局的框架。
```

核心贡献：

1. 提出 agent context cacheability 的指标体系。
2. 设计通用上下文编译器，包括 segment parsing、stability analysis、canonicalization、layout planning、tool-aware recoverable compression。
3. 在多类工具任务上证明，compiler 能在保持任务质量的同时降低 input token cost、TTFT/JCT，并提高 prompt/prefix cache 可利用性。
4. 系统分析 compiler 有效和失效的边界条件。

## 13. 当前 MVP 状态

当前仓库已经有最小骨架：

```text
projects/agent_infer_bench/agent_bench/optimizations/context_compiler.py
projects/agent_infer_bench/agent_bench/analysis/cacheability_metrics.py
projects/agent_infer_bench/scripts/analysis/analyze_context_compiler.py
projects/agent_infer_bench/configs/week07_context_compiler.yaml
```

当前已支持：

1. ContextSegment 抽象。
2. deterministic JSON serialization。
3. static / dynamic / ephemeral layout planning。
4. observation recoverable compression。
5. PSR / DPR / CRO 指标。
6. synthetic controlled ablation。

下一步：

1. 接 vLLM prefix cache on/off。
2. 接 SGLang radix cache。
3. 加 BFCL mini quality evaluation。
4. 加 SWE-Bench Lite/coding trace observation compression。
5. 把 mock latency 替换为真实 serving latency。

## 14. Go / No-Go 标准

继续推进的 Go 条件：

1. Compiler 稳定提高 PSR/CRO，降低 DPR。
2. 在真实 prefix cache backend 上，cacheability 提升能转化为 TTFT/JCT 改善。
3. Compiler 优于 stable tool ordering 和 truncation baseline。
4. 在 BFCL mini 上，JSON validity 和 tool accuracy 基本不下降。
5. 在 coding trace 上，recoverable compression 比普通 truncation 更稳。

停止或降级的 No-Go 条件：

1. Compiler 收益和简单 truncation 一样。
2. 质量明显下降。
3. 真实 backend 上 latency 没有任何改善。
4. 指标只能在 synthetic workload 上成立。
5. 方法无法和 PANDO / ACE / prompt compression 拉开边界。

## 15. 最重要的研究边界

不要做：

1. KV eviction。
2. KV prefetch。
3. future agent invocation prediction。
4. tool result cache。
5. cross-agent KV tensor sharing。
6. 只服务 Web Agent 的 skill distillation。

要做：

1. 通用 tool-using agent context segmentation。
2. cacheability-aware layout planning。
3. deterministic tool schema canonicalization。
4. tool-aware recoverable observation compression。
5. quality-preserved cost reduction。
6. 多类工具任务上的端到端验证。

## 16. 理论支撑与可证明性质

Agent Context Compiler 不需要发展成非常重的理论论文，但如果目标是顶会，需要把它从“prompt 工程经验”提升为“有形式化定义和可证明性质的上下文变换系统”。

### 16.1 形式化模型

将一个 Agent prompt 表示为 segment 序列：

```text
X = [s_1, s_2, ..., s_n]
```

每个 segment 有属性：

```text
s_i = (text_i, stability_i, type_i, reuse_scope_i, quality_constraints_i)
```

其中：

```text
stability_i ∈ {static, semi_static, session_dynamic, turn_dynamic, ephemeral}
```

Compiler 是一个变换函数：

```text
C(X) -> X'
```

它允许做的安全变换包括：

1. 在不改变 segment 内部 token 的情况下重排 segment。
2. 对 tool schema 做 deterministic serialization。
3. 将 ephemeral / turn_dynamic segment 后置。
4. 对 observation 做满足质量约束的可恢复压缩。
5. 为压缩内容生成 source pointer。

### 16.2 命题一：Prefix Cacheability 单调提升

直觉：

```text
如果多个请求共享相同 static segment，但原始布局中这些 static segment 被 dynamic segment 阻断，
那么把 static segment 前移后，longest common prefix 不会变短，通常会变长。
```

可写成：

```text
E[LCP(C(X_i), C(X_j))] >= E[LCP(X_i, X_j)]
```

适用条件：

1. `X_i` 和 `X_j` 来自同一 agent type 或共享 tool/schema/system block。
2. Compiler 不改变 shared static segment 的 token 内容。
3. Compiler 对 static segment 使用确定性排序。
4. Dynamic segment 不被插入到 static prefix 前面。

证明思路：

1. 原始 prompt 中，LCP 会在第一个不相同的 token 处停止。
2. 如果 dynamic/ephemeral segment 出现在 static segment 之前，则即使后续 static segment 完全相同，也无法被 prefix cache 利用。
3. Compiler 将所有共享 static segment 以确定性顺序放到前缀。
4. 因此，编译后请求之间至少共享这些 static prefix token。
5. 所以 expected LCP 不小于原始布局。

这条命题的作用：

```text
证明 layout planning 不是随意 prompt trick，而是在优化可形式化的 prefix reuse objective。
```

### 16.3 命题二：动态污染上界

定义 Dynamic Pollution Ratio：

```text
DPR(X) = dynamic_tokens_before_stable_boundary / total_input_tokens
```

如果 compiler 能够将所有 `ephemeral` 和 `turn_dynamic` segment 移到 stable prefix 之后，则：

```text
DPR(C(X)) = 0
```

更一般地，如果存在不可移动动态字段，例如安全策略要求必须出现在开头，则有：

```text
DPR(C(X)) <= unavoidable_dynamic_tokens / total_input_tokens
```

这条命题解释了为什么：

```text
stable tool ordering alone is insufficient if volatile metadata still appears before the stable block.
```

也就是：

```text
只稳定工具顺序没用；只要动态字段仍在最前面，prefix cache 仍可能完全失效。
```

### 16.4 命题三：可恢复压缩的信息安全性

设 observation 为：

```text
o
```

Recoverable compressor 输出：

```text
Compress(o) = (summary(o), pointer(o), preserved_fields(o))
```

如果存在确定性 retrieval operator：

```text
R(pointer(o)) = o
```

则 recoverable compression 不是不可逆删除，而是将完整信息从 prompt 内联状态移动到外部可寻址状态：

```text
inline observation -> summary + external reference
```

与 truncation 的区别：

```text
truncation: 信息永久丢失。
recoverable compression: 信息默认不进入 prompt，但可以按需恢复。
```

这条性质可以写成：

```text
Recoverable compression preserves access to the original observation under a valid retrieval operator.
```

### 16.5 理论章节的定位

论文中不需要把理论写得过度复杂。建议定位为：

```text
We provide a formal cacheability model and prove that segment-stable layout planning monotonically improves prefix reuse opportunity under deterministic segment equivalence.
```

中文：

```text
我们给出 agent context cacheability 的形式化模型，并证明在确定性 segment 等价条件下，稳定 segment 布局规划能单调提升 prefix reuse opportunity。
```

## 17. Recoverable Reference 的真实运行机制

Recoverable reference 不能只停留在 prompt 文本中的 `obs://...` 字符串，否则审稿人会质疑：

```text
模型真的会用这个 pointer 吗？
pointer 背后是谁来恢复？
恢复是否会抵消节省的 token 和 latency？
```

因此需要实现真实 runtime。

### 17.1 ObservationStore

设计一个最小 ObservationStore：

```python
class ObservationStore:
    def put(self, raw_observation: str, metadata: dict) -> str:
        ...

    def get(self, obs_id: str) -> str:
        ...

    def search(self, obs_id: str, query: str) -> str:
        ...
```

基本流程：

```text
raw observation -> ObservationStore.put -> obs_id
compiler prompt 中写入 summary + obs_id
agent 如需更多细节 -> 调用 retrieve_observation(obs_id)
```

### 17.2 retrieve_observation 工具

为 Agent 暴露一个真实工具：

```json
{
  "name": "retrieve_observation",
  "description": "Retrieve full or partial raw observation by observation id.",
  "parameters": {
    "type": "object",
    "properties": {
      "obs_id": {"type": "string"},
      "query": {"type": "string"}
    },
    "required": ["obs_id"]
  }
}
```

这样 recoverable reference 不再是“提示词技巧”，而是系统机制：

```text
prompt 内只保留摘要；
完整内容通过工具按需取回。
```

### 17.3 需要评估的恢复指标

必须报告：

1. Recovery trigger rate：Agent 多常调用 `retrieve_observation`。
2. Recovery usefulness：恢复后是否能挽救失败样例。
3. Recovery overhead：恢复带来的额外 latency/token 是否小于直接内联完整 observation。
4. Recovery precision：query-based retrieval 是否取回了正确片段。
5. Failure rescue value：少数恢复调用是否显著提升任务成功率。

理想发现：

```text
Recoverable compression has low average recovery overhead but high failure rescue value.
```

中文：

```text
大多数任务不需要恢复，因此平均开销低；少数关键失败样例通过恢复完整 observation 被挽救。
```

## 18. 顶会实验矩阵

如果目标是 ICLR / MLSys / NeurIPS / ASPLOS / EuroSys 级别，实验不能只停留在 synthetic workload。需要覆盖任务质量、cacheability、serving latency 和组合优化。

### 18.1 任务矩阵

| 类别 | 数据/任务 | 主要质量指标 | 作用 |
|---|---|---|---|
| Function calling | BFCL | JSON validity, tool name accuracy, argument accuracy | 验证 tool schema canonicalization 不伤质量 |
| Coding agent | SWE-Bench Lite 或 coding-agent traces | file localization, patch/test proxy, success rate | 验证长 observation 和 recoverable compression |
| Retrieval agent | HoVer / HotpotQA agent traces | answer EM/F1, evidence recall, citation accuracy | 验证 evidence compression |
| Web agent optional | VisualWebArena 小子集 | task success, action accuracy, loop rate | 只做补充，避免和 PANDO 主战场撞车 |

最低要求：

```text
至少 3 类任务，其中至少 2 类是真实工具任务。
```

### 18.2 方法矩阵

| 方法 | 是否必做 | 目的 |
|---|---|---|
| Original prompt | 必做 | 原始 Agent baseline |
| Stable tool order only | 必做 | 分离 tool ordering 贡献 |
| Dynamic fields last only | 必做 | 分离 layout 贡献 |
| Truncation | 必做 | 对比最简单 token reduction |
| Generic summarization | 必做 | 对比普通摘要 |
| LLMLingua / LongLLMLingua-style compression | 建议 | 对比强 prompt compression |
| Don't Break the Cache-style boundary strategy | 必做 | 对比现有 prompt caching 经验策略 |
| Context Compiler without compression | 必做 | 分离 layout/canonicalization 贡献 |
| Context Compiler with recoverable compression | 必做 | 完整方法 |
| Context Compiler + serving prefix cache | 必做 | 证明与 serving 侧互补 |

### 18.3 Serving 矩阵

| Backend | 设置 |
|---|---|
| vLLM | prefix cache off / on |
| SGLang | radix cache |
| OpenAI / Anthropic prompt caching | 可选，预算允许时做 |
| Mock / simulator | 只用于开发，不作为主结果 |

### 18.4 并发与模型矩阵

建议设置：

```text
models: Qwen2.5-7B, Llama-3.1-8B 或 Qwen3-8B
backends: vLLM, SGLang
concurrency: 1, 8, 32
repeat: 3
```

完整实验规模估算：

```text
3 tasks × 2 models × 2 backends × 3 concurrency × 8 methods × 3 repeats
= 864 runs
```

实际执行时可以分层：

1. 主文只放 4-5 个核心方法。
2. appendix 放完整 8-10 个方法。
3. BFCL 可以规模较大。
4. SWE-Bench Lite 可以规模较小但深入分析。
5. Web agent 只作为补充，不做主线。

### 18.5 核心图表

主文至少需要：

1. Context segment breakdown。
2. PSR / DPR / CRO 对比。
3. TTFT/JCT/P95 latency 对比。
4. Quality-cost frontier。
5. Recoverable compression vs truncation。
6. Compiler + serving cache 组合增益。
7. Failure cases。
8. Case study：原始 prompt 到 compiled prompt 的变化。

## 19. 反直觉发现与论文记忆点

顶会论文需要让审稿人记住几个发现，而不只是“我的方法更快”。

### 19.1 发现一：稳定工具顺序本身可能没用

可能结论：

```text
Stable tool ordering does not improve prefix reuse when volatile runtime metadata appears before tool schemas.
```

中文：

```text
只稳定工具顺序没用；如果 timestamp/session id 仍在最前面，prefix cache 依然完全失效。
```

当前 MVP 中已经出现类似信号：

```text
stable_tool_order 的 PSR 仍接近 0，因为 dynamic header 仍在 prefix 前部。
```

### 19.2 发现二：压缩 token 不等于提高 cacheability

Truncation 可能降低 token 和 latency，但它可能：

1. 删除 file path。
2. 删除 error code。
3. 删除 enum value。
4. 删除 evidence source。
5. 降低 task success。

因此需要证明：

```text
Compiler achieves similar cost reduction as truncation with better task quality.
```

### 19.3 发现三：Generic summarization 在 Agent 任务中可能危险

普通摘要模型常常保留“语义大意”，但删除 Agent 执行所需的离散细节：

```text
file path
line number
argument key
enum value
error code
source id
```

这能支撑 tool-aware compression 的必要性。

### 19.4 发现四：Serving 侧 cache 优化依赖上游 context 结构

可能结论：

```text
vLLM/SGLang prefix caching only helps when the agent context exposes stable prefixes.
```

这能把本工作和 serving 系统工作连接起来：

```text
Compiler does not replace serving cache; it exposes cache opportunities for serving systems to exploit.
```

### 19.5 发现五：Recoverable reference 平均少用，但关键时刻很值

理想观察：

```text
Most tasks do not need recovery, but a small number of recovery calls rescue otherwise failed tasks.
```

这能说明 recoverable compression 比 truncation 更适合 long-horizon agent。

## 20. 与 Serving 侧优化的组合实验

组合实验非常重要。它能证明 Agent Context Compiler 是 serving optimization 的上游补充，而不是又一个孤立 prompt trick。

### 20.1 组合设置

建议至少做：

```text
Original + vLLM prefix cache off
Original + vLLM prefix cache on
Compiler + vLLM prefix cache off
Compiler + vLLM prefix cache on

Original + SGLang radix cache
Compiler + SGLang radix cache

Compiler + session-sticky routing
Compiler + prefix-aware routing
```

### 20.2 组合增益指标

关注：

```text
Serving cache alone: +X%
Compiler alone: +Y%
Compiler + serving cache: +Z%
```

如果：

```text
Z > X and Z > Y
```

则可以表述为：

```text
Context Compiler exposes cache opportunities that serving systems can exploit.
```

中文：

```text
上下文编译器暴露了 serving 系统原本难以利用的缓存机会。
```

### 20.3 不要让组合实验变成主贡献

需要注意：

```text
组合实验是证明互补性，不是把论文拉回 routing / KV eviction / prefetch。
```

否则会重新撞 KVFlow / PBKV / Preble。

## 21. 开源工具化价值

如果目标是顶会，建议把 Agent Context Compiler 做成一个轻量开源工具，而不是只作为实验脚本。

### 21.1 工具定位

包名可以是：

```text
agent-context-compiler
```

核心 API：

```python
segments = compiler.parse(agent_state)
compiled = compiler.compile(segments, policy="cache_friendly")
metrics = compiler.analyze(original_prompt, compiled.prompt)
```

### 21.2 适配器

最小适配器：

1. OpenAI-compatible messages adapter。
2. LangChain adapter。
3. BFCL adapter。
4. vLLM/SGLang benchmark adapter。

后续可选：

1. AutoGen adapter。
2. CrewAI adapter。
3. SWE-Agent adapter。

### 21.3 开源工具的论文价值

开源工具能带来：

1. 复现性。
2. 工程可信度。
3. 更清晰的系统边界。
4. 让其他 Agent 框架能直接测自己的 cacheability。
5. 让 serving 系统研究者可以把它作为 upstream optimizer。

它能帮助论文从：

```text
一个实验想法
```

提升为：

```text
一个可复用系统和 benchmark suite。
```

## 22. 顶会潜力评估

### 22.1 当前潜力判断

当前方案的顶会潜力可以概括为：

```text
方向有潜力，但必须从 MVP 工程原型升级为“理论 + 系统 + 多任务质量评估 + serving 组合实验”的完整论文。
```

如果只做到当前 MVP：

```text
workshop / arXiv / CCF-C 技术报告级别。
```

如果补齐真实任务质量闭环和 serving 组合实验：

```text
有机会冲 ICLR / MLSys / NeurIPS Datasets & Benchmarks / ACL 系主会或 Findings。
```

如果进一步做成开源工具，并在多个 Agent 框架上验证：

```text
MLSys / ICLR / NeurIPS / ICML 交叉方向会更有竞争力。
```

### 22.2 最适合的投稿叙事

更适合 AI 顶会的叙事：

```text
Quality-Constrained Context Transformation for Efficient Tool-Using Agents
```

更适合 MLSys 的叙事：

```text
A Compiler Layer Between Agent Frameworks and LLM Serving Systems
```

更适合 benchmark / datasets track 的叙事：

```text
Measuring and Improving Agent Context Cacheability Across Tool-Using Workloads
```

### 22.3 最大风险

主要风险：

1. PANDO / ACE / Don't Break the Cache 等工作已经覆盖了部分 context engineering 和 cache-aware prompting。
2. 如果方法只表现为 prompt 重排，会被认为 novelty 不够。
3. 如果收益主要来自 token truncation，会和 prompt compression 工作重合。
4. 如果缺少真实质量指标，会被认为只是系统优化，没有证明 agent correctness。
5. 如果真实 backend 上 TTFT/JCT 没改善，cacheability 指标会被质疑。
6. 如果 recoverable reference 不实现真实 retrieval tool，会被认为不可落地。

### 22.4 顶会 Go 条件

建议只有满足以下条件，才以顶会主会为目标：

1. 有形式化 cacheability model 和至少 2-3 个可证明命题。
2. 有真实 ObservationStore + retrieve_observation tool。
3. 至少 3 类任务，其中 2 类是真实工具任务。
4. 至少 2 个 serving backend 或 1 个 backend + 1 个商业 API prompt caching。
5. Compiler 优于 truncation、generic summarization、Don't Break-style boundary。
6. Compiler + serving cache 存在组合增益。
7. 至少有 2 个反直觉发现。
8. 有开源工具或清晰 artifact。

### 22.5 现实概率判断

坦诚判断：

```text
当前 MVP：顶会概率低，适合验证方向。
补 BFCL + vLLM/SGLang：有 workshop / arXiv / CCF-C 潜力。
补 SWE-Bench Lite + recoverable runtime + serving 组合实验：开始具备 ICLR/MLSys workshop 或主会弱投稿潜力。
补完整实验矩阵 + 开源工具 + 强发现：才有认真冲 ICLR/MLSys/NeurIPS 的资格。
```

这条线最有希望的点不是“我们让 prompt 更短”，而是：

```text
我们定义并实现了 Agent context 的编译层，使 Agent 框架和 LLM serving 系统之间第一次有了可测量、可优化、可复用的 cacheability contract。
```

