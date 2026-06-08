# Cacheability-Aware Agent Context Planning 研究方向草案

更新时间：2026-05-29

## 1. 一句话概括

本方向研究一个问题：

```text
Agent 的推理成本不仅由模型和 serving engine 决定，也由 Agent 如何组织 prompt、tools、memory、history 和 observations 决定。
```

现有 LLM serving 系统通常在 prompt 已经生成之后，再考虑 prefix cache、KV cache eviction、routing 或 prefetch。但对于 tool-using agents，很多 cache 机会在进入 serving engine 之前就已经被破坏了，例如：

1. tool schema 每轮顺序不稳定。
2. 动态字段放在 prompt 开头，例如 timestamp、session id、retry count。
3. observation 过长，且每轮直接拼接进 history。
4. history/memory 以非结构化文本方式滚动增长。
5. 多 agent / 多工具工作流之间没有显式共享稳定上下文块。

因此，本方向希望把优化位置前移到 agent framework / context construction 层，提出一种 **cacheability-aware context planner**：在保持任务成功率和工具调用质量的前提下，让 Agent 生成更 cache-friendly、cost-efficient、quality-preserving 的上下文。

## 2. 与现有计划的关系

当前 `AgentInferBench` 计划可以作为这个方向的 Phase 0：

```text
Phase 0: 证明 tool-using agent 的上下文结构会显著影响 serving latency、prefix reuse 和 cache locality。
```

原计划中的以下内容可以保留：

1. vLLM / SGLang 统一 benchmark。
2. plain chat vs agent workload characterization。
3. prefix overlap ratio、TTFT、TPOT、JCT、P95 latency、吞吐和显存指标。
4. prompt layout、tool schema ordering、dynamic fields、session locality 实验。
5. prompt canonicalization、stable tool ordering、routing simulator。

但如果以顶会为目标，主贡献不应停留在“benchmark + heuristic 优化”。更好的升级方式是：

```text
从“我发现 prompt layout 会影响 prefix cache”
升级为
“我定义了 agent context cacheability，并提出一个 context planner，使 agent 在质量约束下自动生成更可缓存的上下文。”
```

## 3. 核心研究问题

### 3.1 主问题

```text
How can tool-using agents construct their context to maximize serving cacheability while preserving task success?
```

中文表述：

```text
Tool-using agents 应该如何组织 prompt/context，才能在不降低任务成功率的前提下，提高 serving 系统中的 prefix/KV cache 复用，并降低推理延迟与成本？
```

### 3.2 子问题

1. 如何定义和量化 agent context 的 cacheability？
2. 哪些上下文片段应该稳定放在 prefix，哪些应该 late-bound 放在 suffix？
3. tool schema、memory、history、observation、current query 应该如何分层组织？
4. observation 如何压缩，才能降低 prefill cost，同时不损害 tool call accuracy 和 task success？
5. agent framework 是否应该向 serving engine 暴露结构化 context metadata，例如 cache key、reuse scope、stability hint？
6. 这种方法和 KV cache eviction / prefetch / routing 是否互补？

## 4. 论文定位

建议论文主线：

```text
CacheWeaver: Cacheability-Aware Context Planning for Efficient Tool-Using Agents
```

可选标题：

```text
Cacheability-Aware Context Planning for Efficient Tool-Using LLM Agents
Serving-Aware Context Construction for Tool-Using LLM Agents
Making Tool-Using Agents Cache-Friendly: A Context Planning Approach
Cacheability is an Agent Design Property
```

论文 claim：

```text
Cacheability is not only a serving-system property, but also an agent-design property.
```

也就是：

```text
cache 能不能复用，不只取决于 vLLM/SGLang 怎么管理 KV cache，也取决于 Agent 框架如何构造上下文。
```

## 5. 方法设计

### 5.1 Context Segment Taxonomy

把 agent prompt 拆成结构化 segment，而不是当作一整段纯文本：

| Segment 类型 | 示例 | 变化频率 | 推荐位置 | 处理策略 |
|---|---|---:|---|---|
| static | system instruction、agent role、稳定 tool schemas | 极低 | prefix | 固定顺序、固定格式、可跨请求复用 |
| semi-static | task plan、长期 memory summary、repository summary | 低 | prefix 或 middle | 版本化、稳定 cache key |
| session-dynamic | conversation state、short-term memory、recent tool results | 中 | middle | 结构化压缩、分块复用 |
| turn-dynamic | current user query、current observation、current tool result | 高 | suffix | late binding，避免污染 prefix |
| ephemeral | timestamp、session id、retry count、随机 trace id | 极高 | suffix 或 metadata | 尽量不进入 prompt prefix |

核心思想：

```text
越稳定、越可能被复用的内容越靠前；
越动态、越一次性的内容越靠后；
能变成 metadata 的内容不要进入 prompt prefix。
```

### 5.2 Cacheability Metrics

可以定义以下指标：

#### Prefix Stability Ratio

```text
PSR = stable_prefix_tokens / total_input_tokens
```

衡量每轮输入里有多少 token 属于稳定 prefix。

#### Cache Reuse Opportunity

```text
CRO(i, j) = longest_common_prefix(prompt_i, prompt_j) / min(len(prompt_i), len(prompt_j))
```

衡量两个请求之间理论上可复用的 prefix 比例。

#### Cacheability Opportunity Gap

```text
COG = theoretical_reuse_opportunity - observed_cache_benefit_proxy
```

其中 observed cache benefit proxy 可以用：

1. prefix caching on/off 的 TTFT 差异。
2. identical prefix repeated requests 的 latency delta。
3. SGLang/vLLM 暴露的 cache hit 指标，如果可用。
4. prefill tokens avoided 的估计值，如果 backend 可观测。

这个指标的意义是：

```text
Agent prompt 理论上有复用机会，但实际 serving 没有吃到多少收益。
```

#### Quality-Preserved Cost Reduction

```text
QPCR = cost_reduction under success_rate_drop <= epsilon
```

例如：

```text
在 task success rate 下降不超过 1%-2% 的约束下，最大化 TTFT/JCT/token cost 降低。
```

### 5.3 Context Planner

Context planner 输入：

```text
agent instruction
tool schemas
memory
conversation history
tool observations
current query
serving metadata
quality constraints
```

输出：

```text
planned prompt
segment metadata
cache key
reuse scope
compression policy
layout policy
```

Planner 可以包含四类策略：

#### Stable Layout Planning

1. 固定 system prompt 模板。
2. 固定 tool schema 顺序。
3. 去除或移动 prefix 中的动态字段。
4. 对 JSON schema 做 deterministic serialization。
5. 对相同 tool schema 生成稳定 hash 和版本号。

#### Late Binding

把高动态字段推迟到 prompt 后部：

```text
bad:
timestamp + session_id + system + tools + history + query

better:
system + tools + stable memory + history summary + query + timestamp/session metadata
```

#### Observation Compression

对 observation 做任务相关压缩，而不是简单截断：

1. 对搜索结果：保留 title、url、snippet、score、query relation。
2. 对代码搜索：保留 file path、symbol、line range、error message、patch-relevant snippets。
3. 对 API JSON：保留 status、error code、关键字段，删除重复字段。
4. 对长网页：抽取和当前 goal 相关的 facts。
5. 对执行日志：保留异常栈、失败命令、最后 N 行、warning/error。

#### Recoverable Compression

压缩不是不可逆删除，而是保留 pointer：

```text
compressed observation + source pointer / retrieval key
```

当 agent 需要更多细节时，可以重新读取原始 observation。

这比普通 summarization 更适合 agent，因为 agent 经常需要回看细节。

### 5.4 Agent Context Contract

为了和 serving engine 对接，可以设计一个结构化 contract：

```json
{
  "segments": [
    {
      "name": "system_instruction",
      "text": "...",
      "stability": "static",
      "reuse_scope": "global",
      "cache_key": "sha256:..."
    },
    {
      "name": "tool_schema_block",
      "text": "...",
      "stability": "static",
      "reuse_scope": "agent_type",
      "cache_key": "sha256:..."
    },
    {
      "name": "observation_summary",
      "text": "...",
      "stability": "turn_dynamic",
      "reuse_scope": "session",
      "source_pointer": "obs://..."
    }
  ],
  "quality_constraint": {
    "max_success_drop": 0.02,
    "must_preserve": ["tool_name", "file_path", "error_code"]
  }
}
```

短期内可以只在 benchmark 和 router 层使用这些 metadata。长期如果要冲更强系统会议，可以接入 serving engine，用这些信息指导 request grouping、routing、prefix cache key、cache admission 或 prefetch。

## 6. 实验设计

### 6.1 Workloads

至少包含三类 workload：

1. Function calling / tool schema workload
   - 数据：BFCL、ToolBench 或自建 function-call traces。
   - 关注：tool schema 顺序、JSON validity、tool call accuracy。

2. Coding agent workload
   - 数据：SWE-Bench / SWE-Bench Lite / 自建 repo repair traces。
   - 关注：长 observation、文件片段、错误日志、patch success。

3. Web / retrieval agent workload
   - 数据：WebArena、HotpotQA/HoVer + LangChain-style agent，或自建 browse/search traces。
   - 关注：网页 observation 压缩、事实保真、answer correctness。

如果 GPU 资源有限，第一阶段可以先做：

```text
BFCL + SWE-Bench Lite + synthetic multi-turn tool traces
```

### 6.2 Baselines

Prompt/context baselines：

1. Original agent prompt。
2. Random tool order。
3. Stable tool order。
4. Manual prompt canonicalization。
5. Truncation-based observation compression。
6. Generic summarization-based compression。
7. Proposed cacheability-aware context planner。

Serving baselines：

1. vLLM prefix caching on/off。
2. SGLang RadixAttention / prefix cache。
3. Round-robin routing。
4. Session-sticky routing。
5. Prefix-hash routing。
6. 如果可复现，再和 KVFlow/PBKV 的公开结果或简化实现做对照，但不要把它们作为必须复现的主线。

### 6.3 Metrics

效率指标：

1. TTFT。
2. TPOT。
3. end-to-end latency。
4. job completion time。
5. P50/P95/P99 latency。
6. tokens/s。
7. requests/s。
8. peak GPU memory。
9. input tokens / output tokens。
10. estimated prefill cost。

Cacheability 指标：

1. prefix overlap ratio。
2. Prefix Stability Ratio。
3. Cache Reuse Opportunity。
4. Cacheability Opportunity Gap。
5. cache hit rate，如果 backend 暴露。
6. repeated-prefix latency benefit。

质量指标：

1. task success rate。
2. tool call accuracy。
3. JSON validity。
4. answer exact match / F1 / judge score。
5. patch success rate，针对 SWE-Bench。
6. compression faithfulness。
7. recovery rate，针对 recoverable compression。

### 6.4 关键实验

#### Experiment 1: Agent Context Characterization

问题：

```text
真实 agent traces 中，哪些 token 是稳定的、动态的、可压缩的、污染 prefix 的？
```

产出：

1. token breakdown。
2. prefix overlap by turn。
3. dynamic field impact。
4. observation growth curve。
5. cacheability opportunity gap。

#### Experiment 2: Layout Optimization

比较：

1. original layout。
2. stable tools first。
3. dynamic fields last。
4. static/semi-static/dynamic segmented layout。

目标：

```text
证明 context layout 本身能改变 prefix cache 收益。
```

#### Experiment 3: Observation Compression

比较：

1. no compression。
2. truncation。
3. generic summarization。
4. tool-aware compression。
5. recoverable compression。

目标：

```text
在 success rate drop <= epsilon 的约束下最大化 latency/token cost reduction。
```

#### Experiment 4: End-to-End Agent Performance

在真实 agent task 上比较：

1. original agent。
2. original + serving prefix cache。
3. original + sticky routing。
4. context planner only。
5. context planner + serving prefix cache。
6. context planner + prefix-aware routing。

目标：

```text
证明 context planner 和 serving cache/routing 是互补的。
```

#### Experiment 5: Failure Cases

主动分析：

1. compression 丢失关键细节导致 task failure。
2. context 重排改变模型行为。
3. tool schema canonicalization 影响 tool selection。
4. 超长 observation 低复用，cacheability 收益有限。
5. 高动态任务中 prefix reuse opportunity 本来就低。

## 7. 可能的顶会潜力判断

### 7.1 为什么有潜力

这个方向有潜力的原因：

1. Agent 使用越来越多，但多数 agent framework 仍然把 prompt/context 当成纯文本拼接问题。
2. LLM serving 的 prefix/KV cache 优化已经很热，但大多发生在 serving engine 内部。
3. KVFlow/PBKV 说明“Agent + cache management”是有顶会兴趣的方向。
4. 但 agent framework 如何生成 cache-friendly context 仍然没有被系统化解决。
5. 该方向天然连接 AI 质量指标和系统效率指标，适合 MLSys / NeurIPS / ICLR / ICML / ASPLOS / EuroSys 交叉叙事。

### 7.2 什么情况下够顶会

需要满足：

1. 不只是 heuristic，需要有清晰的问题定义和指标，例如 cacheability、opportunity gap、quality-preserved cost reduction。
2. 不只 synthetic workload，需要真实 agent traces 和真实任务成功率。
3. 不只降低 token 数，需要证明不损害 tool-use quality。
4. 不只外部分析，最好接入至少一个真实 agent framework 和一个真实 serving backend。
5. 不只和 naive baseline 比，要和 prompt compression、stable layout、serving prefix cache、routing baseline 比。
6. 必须清楚说明与 KVFlow/PBKV 的互补关系。

### 7.3 什么情况下不够顶会

以下版本大概率不够：

1. 只做 prompt canonicalization，缺少质量评估。
2. 只做 synthetic traces，缺少真实任务。
3. 只测 latency，不测 task success。
4. 只做 benchmark，没有方法。
5. 只做 observation summarization，和已有 prompt compression 工作区分不清。
6. 只做 routing/prefetch，会直接撞 KVFlow/PBKV/Preble。

## 8. 相关工作与重合度判断

### 8.1 KVFlow

KVFlow 的方向是：

```text
已知 multi-agent workflow 结构下，利用 Agent Step Graph 指导 KV cache eviction 和 overlapped prefetch。
```

它主要解决：

1. 未来哪个 agent 会执行。
2. 哪些 agent prefix KV 应该保留在 GPU。
3. 哪些 CPU-offloaded KV 应该提前 prefetch。

与本方向的关系：

```text
KVFlow 管理已经生成的 KV cache；
本方向让 agent 在生成 prompt/context 时就更 cache-friendly。
```

重合风险：

1. 如果本方向也主打“预测未来 agent 调用来做 KV eviction/prefetch”，会严重重合。
2. 如果只做 session-aware routing，也会显得像 KVFlow/PBKV 的简化版。

差异化空间：

1. 聚焦 context construction，而不是 cache eviction。
2. 聚焦 tool-using single/multi-turn agents，而不仅是固定 multi-agent workflow。
3. 加入 quality-preserving observation compression。
4. 暴露 agent context contract，连接 agent framework 和 serving engine。

### 8.2 PBKV

PBKV 的方向是：

```text
动态 agent workflows 下，训练 predictor 预测未来 K 步 agent 调用，并据此做 KV cache management。
```

它比 KVFlow 更进一步，针对 workflow 不完全静态的场景。

与本方向的关系：

```text
PBKV 预测未来调用，用于 KV cache eviction/prefetch；
本方向规划当前和未来 prompt/context 的结构，用于提高 cacheability 和降低 prefill cost。
```

重合风险：

1. 不应把“future agent invocation prediction”作为主贡献。
2. 不应把“prediction-based KV eviction”作为主贡献。

差异化空间：

1. 研究 cache opportunity 如何在 prompt 构造阶段被创造或破坏。
2. 研究 observation compression 和 task quality 的关系。
3. 研究 agent framework 的结构化 context contract。
4. 将 PBKV/KVFlow 作为 downstream cache manager，本方向作为 upstream context planner。

### 8.3 Preble

Preble 关注 multi-tenant LLM serving 中的 prefix cache 复用和 request scheduling。它的核心在 serving scheduler 层，希望把共享 prefix 的请求放到合适的实例上，提高 prefix cache 命中。

与本方向的关系：

```text
Preble 优化请求调度以利用已有 prefix overlap；
本方向优化 agent context，使 prefix overlap 更稳定、更显式、更容易被调度器利用。
```

重合风险：

1. 如果只做 prefix-aware routing，会和 Preble 很接近。
2. 如果没有 agent-specific context planning，创新性不足。

差异化空间：

1. Agent-specific context segmentation。
2. Tool schema / observation / memory 的上下文规划。
3. task success 与 latency/cost 的联合评估。

### 8.4 SGLang / RadixAttention

SGLang 的 RadixAttention 通过 radix tree 管理 prefix cache，自动发现共享 prefix 并复用 KV cache。

与本方向的关系：

```text
SGLang 能复用共享 prefix；
本方向让 agent prompt 更容易形成共享 prefix。
```

重合风险较低，因为 SGLang 是 serving mechanism，本方向是 agent context construction。

### 8.5 vLLM / PagedAttention / Prefix Caching

vLLM 通过 PagedAttention 管理 KV cache memory，并支持 prefix caching。它解决的是 serving engine 内部的高效 KV 管理。

与本方向的关系：

```text
vLLM 提供 cache 机制；
本方向提高 agent context 对这些机制的可利用性。
```

### 8.6 LMCache / CacheGen / CacheBlend / 其他 KV Cache 系统

这类工作主要关注 KV cache 的存储、传输、压缩、复用、跨请求共享或跨层融合。

与本方向的关系：

```text
它们优化 KV cache 的生命周期；
本方向优化生成 KV cache 之前的 context 结构。
```

重合风险中等，尤其当本方向开始做 KV compression 或 cache admission 时，需要小心不要偏离到已有系统工作。

### 8.7 Prompt Compression: LLMLingua / LongLLMLingua / Selective Context

Prompt compression 工作关注如何删除或压缩不重要 token，使 LLM 输入更短。

与本方向的关系：

```text
Prompt compression 压短输入；
本方向在 agent 场景下做 serving-aware、tool-aware、quality-constrained context planning。
```

重合风险：

1. 如果只做 summarization/truncation，会和 prompt compression 很像。
2. 如果只证明 token 变少、latency 变低，不够新。

差异化空间：

1. Agent observation 具有工具类型、状态转移和可恢复引用。
2. 质量指标不是普通 QA accuracy，而是 tool call accuracy、task success、patch success。
3. 优化目标不是单纯 token reduction，而是 cacheability + cost + quality。
4. 压缩策略和 context layout / prefix cache 共同优化。

### 8.8 Agent Memory / Context Management

Agent memory 工作关注长期记忆、短期记忆、reflection、episodic memory、retrieval memory 等。

与本方向的关系：

```text
Agent memory 主要关心让 agent 记住什么；
本方向关心以什么结构记住和呈现，才能兼顾 serving efficiency。
```

差异化空间：

1. memory 的稳定性和 reuse scope。
2. memory summary 的 cache key 和版本化。
3. memory placement 对 prefix cache 的影响。
4. memory compression 对 task quality 和 latency 的联合影响。

### 8.9 Prompt Caching API / 工业实践

OpenAI、Anthropic、Gemini 等 API 都支持或暴露某种 prompt/context caching 能力，工业实践通常建议把稳定内容放在 prompt 前部，例如 system instruction、tool definitions、examples。

与本方向的关系：

```text
工业 API 已经证明 cache-friendly prompt design 很重要；
但这些通常是经验性建议，不是系统化 agent context planner。
```

差异化空间：

1. 形式化 cacheability 指标。
2. 面向 tool-using agents 的自动 context planning。
3. 真实 agent workloads 上的质量-成本评估。
4. 与开源 serving backend 的端到端验证。

## 9. 是否可做

### 9.1 可做性结论

结论：

```text
这个方向可做，但必须避开“KV cache 管理层”的正面战场，把主贡献放在 agent context construction / observation compression / cacheability contract。
```

如果目标是顶会，建议不要把论文写成：

```text
又一个 KV cache eviction / prefetch / routing 方法。
```

而应写成：

```text
第一个系统研究 tool-using agent context cacheability，并提出 agent-framework 层 context planner，在质量约束下提升 serving efficiency。
```

### 9.2 风险

主要风险：

1. 相关工作发展很快，KVFlow/PBKV/Preble 已经覆盖了很多 cache-aware serving 机制。
2. Prompt compression 方向已有大量工作，observation compression 必须体现 agent-specific 和 serving-aware。
3. 如果真实任务 success rate 下降，方法很难说服审稿人。
4. 如果收益只来自 token 变少，而不是 cacheability 改善，创新性会变弱。
5. 如果只用 synthetic traces，顶会说服力不足。

### 9.3 降低风险的做法

1. 从第一天就记录质量指标，不只记录 latency。
2. 做真实 traces，不只做合成 workloads。
3. 把 KVFlow/PBKV 放在 related work 中明确定位为 downstream cache managers。
4. 避免把 scheduler / eviction / prefetch 作为主贡献。
5. 把 context planner 做成可插拔 agent framework 组件。
6. 强调 cacheability metric 和 context contract，而不是零散 prompt trick。

## 10. 推荐路线图

### Month 1: Measurement and Metrics

目标：

```text
证明 agent context cacheability 是真实问题。
```

任务：

1. 收集 BFCL / SWE-Bench Lite / synthetic tool traces。
2. 标注 context segments：system、tools、memory、history、observation、query。
3. 计算 prefix overlap、PSR、CRO、COG。
4. 跑 vLLM/SGLang prefix cache on/off 实验。
5. 产出 characterization 报告。

### Month 2: Context Planner MVP

目标：

```text
实现自动 context layout planning 和 stable tool schema canonicalization。
```

任务：

1. 实现 deterministic tool serialization。
2. 实现 static/dynamic segment planner。
3. 实现 dynamic field late binding。
4. 实现基本 observation compressor。
5. 在 BFCL 和 synthetic agent traces 上跑 ablation。

### Month 3: Quality-Constrained Compression

目标：

```text
把 observation compression 从工程 trick 升级成质量约束优化。
```

任务：

1. 为不同 tool output 实现 tool-aware compressors。
2. 加入 recoverable compression。
3. 评估 task success、tool accuracy、JSON validity。
4. 和 truncation、generic summarization、LLMLingua-style baseline 比。

### Month 4-6: Real Agent Integration

目标：

```text
接入真实 agent framework 和 serving backend。
```

任务：

1. 接 LangChain / AutoGen / SWE-Agent 之一。
2. 接 vLLM 或 SGLang。
3. 跑 SWE-Bench Lite / HoVer / WebArena 子集。
4. 评估端到端 JCT、success rate、cacheability、P95 latency。
5. 写 paper draft。

### Month 6-9: Paper Strengthening

目标：

```text
补强顶会审稿会追问的部分。
```

任务：

1. 更强 baselines。
2. 更多模型和 backend。
3. 消融实验。
4. failure cases。
5. generalization across tasks。
6. artifact reproducibility。

## 11. 与 KVFlow/PBKV 的推荐表述

论文里可以这样写：

```text
KVFlow and PBKV demonstrate that agent workflows expose valuable signals for KV-cache management inside serving systems.
Our work asks a complementary question: before the serving system manages KV cache, can the agent framework construct contexts that are more cacheable in the first place?
```

中文：

```text
KVFlow 和 PBKV 说明 agent workflow 信息可以帮助 serving 系统更好地管理 KV cache。
我们的工作关注一个互补问题：在 serving 系统管理 KV cache 之前，agent framework 能否生成更可缓存的上下文？
```

推荐不要这样写：

```text
我们比 KVFlow/PBKV 更好地预测未来 cache reuse。
```

因为这会正面撞它们的核心贡献。

## 12. 初步结论

这个方向有继续发展的价值，但要守住三条边界：

1. 主贡献是 agent context planning，不是 KV cache eviction。
2. 核心指标是 cacheability + quality-preserved cost reduction，不是单纯 latency。
3. 实验必须有真实 agent task，不然容易退化成 prompt engineering。

如果能做到以上三点，它有机会形成一篇比较有辨识度的 AI Infra / Agent Efficiency 论文。短期可以产出 arXiv / workshop / CCF-C 级别成果；中长期如果真实系统实现、真实任务和方法抽象都足够强，可以尝试 MLSys / NeurIPS / ICLR / ICML / ASPLOS / EuroSys 等更高目标。

