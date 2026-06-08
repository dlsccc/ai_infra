# Cacheability-Aware Agent Context Planning 顶会冲刺计划

更新时间：2026-05-29  
当前阶段：AgentInferBench 已推进到 Week 6 左右  
目标：从现有 Agent inference benchmark 项目，升级为有顶会潜力的 AI Infra / Agent Efficiency 研究论文。

## 1. 战略判断

### 1.1 现在的计划为什么不够顶会

当前 `AgentInferBench` 的价值很明确：

1. 能构建 agent serving benchmark。
2. 能分析 plain chat 与 tool-using agent 的 inference 差异。
3. 能验证 prompt layout、tool schema ordering、prefix caching、session locality 对 latency 的影响。
4. 能形成技术报告、博客、求职项目和 workshop/arXiv 雏形。

但如果目标是 ICLR / NeurIPS / MLSys / ASPLOS / EuroSys / NSDI 级别主会，当前形态还不够。主要问题是：

1. 主贡献偏 measurement + engineering heuristic，缺少清晰的新问题定义。
2. prompt canonicalization 和 stable tool ordering 容易被认为是 prompt engineering。
3. session/cache-aware routing 容易撞 KVFlow、PBKV、Preble 等 serving/cache 管理工作。
4. 如果只测 latency，不测 task success / tool accuracy，会被 AI 方向审稿人认为质量约束不足。
5. 如果只用 synthetic traces，会被系统方向审稿人认为 workload 不真实。

因此，顶会版需要把问题升级为：

```text
Agent context cacheability 是 agent framework 层的设计问题，而不仅是 serving engine 层的 cache 管理问题。
```

### 1.2 建议主线

推荐论文主线：

```text
Cacheability-Aware Context Planning for Efficient Tool-Using LLM Agents
```

可用项目名：

```text
CacheWeaver
```

一句话主张：

```text
Cacheability is not only a serving-system property, but also an agent-design property.
```

中文表述：

```text
Agent 的 serving 成本不只由 vLLM/SGLang 如何管理 KV cache 决定，也由 Agent 如何组织 tools、memory、history、observations 和 current query 决定。
```

### 1.3 与 KVFlow / PBKV 的边界

这条线必须避开“预测未来 agent 调用来做 KV cache eviction/prefetch”的正面战场。

推荐定位：

```text
KVFlow/PBKV: serving engine 层管理已经生成的 KV cache。
CacheWeaver: agent framework 层生成更可缓存、更低成本、质量可控的 context。
```

也就是：

1. 不主打 KV eviction。
2. 不主打 CPU/GPU KV prefetch。
3. 不主打 future invocation predictor。
4. 主打 context construction、context layout、observation compression、cacheability metrics、quality-cost tradeoff。

## 2. 顶会论文需要达到的标准

### 2.1 最低可投顶会标准

如果要认真冲顶会，至少需要满足：

1. 有明确问题定义：什么是 agent context cacheability，为什么现有 agent framework 会破坏它。
2. 有新指标：Prefix Stability Ratio、Cache Reuse Opportunity、Cacheability Opportunity Gap、Quality-Preserved Cost Reduction。
3. 有方法：不是手工 prompt trick，而是一个自动 context planner。
4. 有真实 workload：至少 BFCL / SWE-Bench Lite / HoVer / WebArena / LangChain traces 中的两类。
5. 有质量指标：tool call accuracy、JSON validity、task success、patch success、answer correctness。
6. 有 serving 指标：TTFT、JCT、P95/P99、prefill cost、tokens/s、peak memory、cache hit proxy。
7. 有强 baseline：original agent、truncation、generic summarization、stable layout、vLLM/SGLang prefix cache、routing baseline。
8. 有消融：layout planning、stable tool serialization、late binding、observation compression、recoverable compression 分开评估。
9. 有 failure cases：压缩导致失败、重排改变行为、高动态任务低收益、长 observation 低复用。
10. 有 artifact：可复现实验脚本、公开 traces 或 trace generator、可运行 demo。

### 2.2 顶会审稿人会问的问题

必须提前准备回答：

1. 这和 KVFlow/PBKV 有什么区别？
2. 这和 prompt compression 有什么区别？
3. 这是不是只是 prompt engineering？
4. 为什么不是把所有东西交给 SGLang RadixAttention / vLLM prefix caching？
5. 降低 latency 是否牺牲了 task success？
6. observation compression 是否丢失关键事实？
7. 真实 agent workload 上收益是否稳定？
8. 方法是否依赖某个具体模型、backend 或 agent framework？
9. 如果 prompt cache API 本来就建议把稳定内容放前面，你的研究新增了什么？
10. 这个方法能否和 KVFlow/PBKV 互补？

## 3. 论文贡献设计

### 3.1 Contribution 1: Agent Context Cacheability Characterization

目标：

```text
证明真实 tool-using agents 中存在大量潜在 prefix/cache reuse 机会，但 agent context construction 经常破坏这些机会。
```

需要产出：

1. context segment taxonomy：static、semi-static、session-dynamic、turn-dynamic、ephemeral。
2. token breakdown by segment。
3. prefix overlap by turn。
4. dynamic-field pollution analysis。
5. observation growth curve。
6. theoretical cache opportunity vs observed latency benefit gap。

### 3.2 Contribution 2: Cacheability Metrics

建议定义：

```text
PSR = stable_prefix_tokens / total_input_tokens
CRO(i,j) = longest_common_prefix(prompt_i,prompt_j) / min(len(prompt_i),len(prompt_j))
COG = theoretical_reuse_opportunity - observed_cache_benefit_proxy
QPCR = cost_reduction under success_rate_drop <= epsilon
```

其中 `epsilon` 可以先设为 1%-2%，后续按任务难度调整。

### 3.3 Contribution 3: Cacheability-Aware Context Planner

Planner 输入：

```text
instruction, tool schemas, memory, history, observations, current query, quality constraints, serving metadata
```

Planner 输出：

```text
planned prompt, segment metadata, cache keys, reuse scopes, compression decisions
```

核心策略：

1. Stable layout planning：稳定内容前置，固定 schema serialization。
2. Late binding：timestamp、session id、retry count、request id 后置或转为 metadata。
3. Tool-aware observation compression：不同工具输出采用不同压缩策略。
4. Recoverable compression：压缩后保留 source pointer，需要时再 retrieve 原文。
5. Quality guard：保留 tool name、file path、error code、API status、关键 evidence。

### 3.4 Contribution 4: End-to-End Quality-Cost Evaluation

目标不是简单降低 token，而是证明：

```text
在任务质量基本不下降的前提下，降低 prefill cost、TTFT、JCT 和 P95/P99 latency。
```

关键图表：

1. Context segment breakdown。
2. Cacheability opportunity gap。
3. Layout planning ablation。
4. Observation compression quality-cost frontier。
5. End-to-end latency vs success rate。
6. vLLM/SGLang backend comparison。
7. Failure cases。
8. Case study：一次真实 agent trace 被 planner 改写前后的变化。

## 4. 研究路线图

### Phase A: 问题验证与指标成型

时间：2026-05-29 至 2026-06-16  
目标：把当前 Week 6 工作升级成顶会论文的 measurement foundation。

任务：

1. 整理现有 AgentInferBench 代码，确保 plain chat、single_tool、multi_tool_serial、long_observation 可稳定跑通。
2. 为每条 prompt trace 标注 segment：system、tools、history、memory、observation、current query、ephemeral。
3. 实现 PSR、CRO、COG 的离线计算。
4. 跑 vLLM prefix caching on/off、SGLang radix cache baseline。
5. 产出第一版 characterization report。

验收标准：

1. 至少 3 类 workload。
2. 至少 100 条 deterministic traces。
3. 至少 5 张 characterization 图。
4. 能说清楚“哪些 context 破坏 cacheability”。

### Phase B: 真实数据与任务质量闭环

时间：2026-06-17 至 2026-07-07  
目标：避免 synthetic-only，被顶会审稿人一票打回。

优先数据：

1. BFCL：function calling / tool schema / JSON validity。
2. SWE-Bench Lite：coding agent / long observation / patch success。
3. HoVer 或 HotpotQA + LangChain-style agent：retrieval / fact verification / iterative reasoning。

任务：

1. 接入 BFCL traces 或生成可复现 function-calling traces。
2. 接入 SWE-Bench Lite 子集，先不要求完整 agent 全自动修复，至少采集工具调用和 observation trace。
3. 为每类任务定义质量指标。
4. 建立 latency-quality 联合结果 schema。

验收标准：

1. 至少 2 类真实任务。
2. 每类至少 100-300 条可复现 trace，资源不够可先 50 条做 pilot。
3. 每条 trace 有 latency metrics 和 quality metrics。
4. 不能只记录 token 数，必须记录 task/tool quality。

### Phase C: Context Planner MVP

时间：2026-07-08 至 2026-07-31  
目标：把 heuristic 升级成方法原型。

任务：

1. 实现 deterministic tool schema serialization。
2. 实现 static/semi-static/dynamic/ephemeral segment planner。
3. 实现 dynamic field late binding。
4. 实现 tool-aware observation compressor v1。
5. 实现 recoverable compression metadata。
6. 提供 LangChain-style wrapper 或独立 prompt builder。

方法输出格式：

```json
{
  "planned_prompt": "...",
  "segments": [
    {
      "name": "tool_schema_block",
      "stability": "static",
      "reuse_scope": "agent_type",
      "cache_key": "sha256:..."
    }
  ],
  "quality_guard": {
    "must_preserve": ["tool_name", "file_path", "error_code"]
  }
}
```

验收标准：

1. Planner 可自动处理至少 3 类 context segment。
2. Planner 不依赖手写每条 prompt。
3. 对 BFCL 不降低 JSON validity / tool call accuracy。
4. 对 synthetic/multi-turn traces 提高 prefix overlap 或降低 TTFT。

### Phase D: Quality-Cost Frontier

时间：2026-08-01 至 2026-08-21  
目标：把 observation compression 从“压短”升级成“质量约束优化”。

任务：

1. 对不同 compressor 做对比：
   - no compression
   - truncation
   - generic LLM summarization
   - tool-aware compression
   - recoverable tool-aware compression
2. 画 quality-cost frontier。
3. 设定 `success_drop <= 1%-2%` 的约束。
4. 分析失败样例。

验收标准：

1. 至少 2 个真实 workload 上有质量-成本曲线。
2. 至少一个 workload 在质量基本不降的情况下有明确 latency/token cost 收益。
3. 至少 5 个失败 case，有解释。

### Phase E: End-to-End System Evaluation

时间：2026-08-22 至 2026-09-10  
目标：形成可投主会的端到端实验。

任务：

1. 跑 vLLM + SGLang 两个 backend，至少一个完整，另一个可以作为 cross-backend validation。
2. 跑 baseline：
   - original agent
   - original + prefix cache
   - stable tool ordering
   - truncation
   - generic summarization
   - CacheWeaver planner
   - CacheWeaver + prefix cache
3. 跑 concurrency：
   - concurrency 1 / 4 / 8 / 16
4. 记录 P95/P99，不只记录 mean。
5. 固定随机种子和环境版本。

验收标准：

1. 至少 6 张核心图可放论文。
2. 至少 2 个真实 workload + 1 个 synthetic controlled workload。
3. 至少 2 个模型或 2 个 backend。
4. 代码能一键重跑核心表格。

### Phase F: Submission Sprint

时间：2026-09-11 至对应会议 deadline  
目标：根据成熟度选择一个主路线提交，不要同时投多个 archival venue。

任务：

1. 8 月 15 日前完成 venue decision。
2. 8 月 25 日前完成 full paper v0。
3. 9 月 1 日前完成 full paper v1 + all figures。
4. 9 月 7 日前完成 related work / limitations / ethics / reproducibility。
5. deadline 前 72 小时冻结实验数字。
6. deadline 前 48 小时冻结正文。
7. deadline 前 24 小时只改格式和错别字。

## 5. 每周执行计划

### Week 0: 2026-05-29 至 2026-06-01

1. 冻结研究主线为 Cacheability-Aware Agent Context Planning。
2. 把 AgentInferBench 当前 Week6 实验补齐。
3. 建立 `docs/research/` 下的论文计划、related work、实验索引。
4. 明确不要继续扩散到 KV eviction / prefetch。

### Week 1: 2026-06-02 至 2026-06-08

1. 实现 context segment 标注。
2. 实现 PSR/CRO 初版。
3. 跑 prompt layout、tool ordering、dynamic field 三类 controlled experiments。
4. 写 `agent_context_cacheability_measurement.md` 初稿。

### Week 2: 2026-06-09 至 2026-06-15

1. 接入 vLLM prefix caching on/off。
2. 接入 SGLang baseline。
3. 实现 COG proxy。
4. 输出第一版 5 张图：token breakdown、prefix overlap、layout impact、cache on/off、dynamic field impact。

### Week 3: 2026-06-16 至 2026-06-22

1. 接 BFCL 或 function-calling traces。
2. 记录 tool call accuracy、JSON validity。
3. 做 tool schema canonicalization ablation。
4. 写 BFCL pilot report。

### Week 4: 2026-06-23 至 2026-06-29

1. 接 SWE-Bench Lite 或 coding-agent trace 子集。
2. 标注 file path、error message、patch-relevant snippets。
3. 实现 coding observation compressor v0。
4. 记录 patch/test proxy，如果完整 patch success 暂时做不到，至少记录 tool-step correctness proxy。

### Week 5: 2026-06-30 至 2026-07-06

1. 接 HoVer/HotpotQA + retrieval agent trace。
2. 实现 retrieval observation compressor v0。
3. 完成真实数据质量指标定义。
4. 写 Phase A/B 总结。

### Week 6: 2026-07-07 至 2026-07-13

1. 实现 Context Planner MVP。
2. 支持 static / semi-static / dynamic / ephemeral segment 分类。
3. 支持 deterministic serialization。
4. 产出 planner 前后 prompt diff case study。

### Week 7: 2026-07-14 至 2026-07-20

1. 在 BFCL 上跑 planner ablation。
2. 在 synthetic multi-turn 上跑 latency ablation。
3. 修复 planner 改变模型行为的问题。
4. 写方法 Section 初稿。

### Week 8: 2026-07-21 至 2026-07-27

1. observation compression v1。
2. tool-aware vs truncation vs generic summarization。
3. 加入 recoverable compression pointer。
4. 开始画 quality-cost frontier。

### Week 9: 2026-07-28 至 2026-08-03

1. 补 SWE-Bench Lite / coding trace 实验。
2. 补 retrieval trace 实验。
3. 形成第一版完整 result table。
4. 如果准备走 ACL/EMNLP/EACL 方向，注意 ARR August 2026 submission 是 2026-08-03。

### Week 10: 2026-08-04 至 2026-08-10

1. 完成 paper v0：Introduction、Motivation、Method、Evaluation skeleton。
2. 完成 6 张核心图的初版。
3. 做 related work gap table。
4. 决定主投 AI venue 还是 systems venue。

### Week 11: 2026-08-11 至 2026-08-17

1. 扩大实验规模。
2. 补 P95/P99。
3. 补 failure cases。
4. 内部做一次“审稿人攻击”：逐条回答 novelty、baseline、quality、real workload。

### Week 12: 2026-08-18 至 2026-08-24

1. paper v1 完整成稿。
2. 所有图表统一风格。
3. 确认实验数字可追溯。
4. 如果冲 NeurIPS workshop，准备 workshop version。

### Week 13: 2026-08-25 至 2026-08-31

1. NeurIPS workshop version 截止准备。
2. 主会版继续补强。
3. 冻结 abstract 和 contribution bullets。
4. 准备 artifact README。

### Week 14: 2026-09-01 至 2026-09-07

1. 如果冲 ASPLOS 2027 September Cycle，进入最后一周。
2. 如果冲 NSDI / EuroSys / ICLR，完成 v2。
3. 补 reviewer checklist 和 limitations。
4. deadline 前不再新增大实验。

### Week 15: 2026-09-08 至 2026-09-14

1. ASPLOS 2027 September Cycle deadline：2026-09-09。
2. NSDI 2027 Fall abstract：2026-09-10。
3. 如果不投 ASPLOS，专注 NSDI/EuroSys/ICLR。
4. 检查双投政策，不能同一篇同时投多个 archival venue。

### Week 16: 2026-09-15 至 2026-09-24

1. NSDI full deadline：2026-09-17。
2. EuroSys Fall abstract：2026-09-17。
3. EuroSys full deadline：2026-09-24。
4. ICLR 2027 预计也在 9 月下旬，官方未发布时按 2026 模式预留。
5. 只能选择一个 archival venue 投主版本。

### Week 17-22: 2026-09-25 至 2026-10-31

1. 根据是否已投稿，进入 rebuttal prep 或下一轮强化。
2. 如果未投，准备 AAMAS 2027 / MLSys 2027 / AISTATS 2027 / ARR October 路线。
3. 扩大真实任务规模。
4. 把系统 artifact 做扎实。

### Week 23-31: 2026-11-01 至 2026-12-31

1. 准备 rebuttal。
2. 根据 review 补实验。
3. 准备 ICML 2027 / MLSys 2027 / ARR next cycle。
4. 打磨公开 GitHub、技术报告、arXiv。

## 6. 2026 下半年会议列表与匹配度

说明：

1. 以下日期以 2026-05-29 检索到的官方页面为准；未发布官方 2027 CFP 的会议会标注为“预计/待确认”。
2. 同一篇论文不能同时投多个 archival venue。需要在 deadline 前选择一条主路线。
3. 匹配度按当前方向 `Cacheability-Aware Agent Context Planning` 评估。

### 6.1 第一梯队：最值得认真冲刺

| Venue | 方向 | Deadline | 匹配度 | 建议 |
|---|---|---:|---:|---|
| ICLR 2027 | AI / LLM / Agent efficiency | 官方 CFP 未发布；参考 ICLR 2026 为 9 月下旬 | 8.5/10 | 如果方法更偏 agent/context/quality-cost，首选 |
| MLSys 2027 | ML systems / LLM serving / agents | 官方 CFP 未发布；MLSys 2026 deadline 为 2025-10-30 | 9/10 | 最自然匹配，但需等 2027 CFP |
| EuroSys 2027 Fall | systems / AI systems | abstract 2026-09-17，full 2026-09-24 | 7.5/10 | 如果实现扎实、系统评估强，可冲 |
| ASPLOS 2027 September Cycle | architecture/OS/PL + ML systems | full 2026-09-09 | 7/10 | 需要更强系统机制，不适合纯 agent prompt 方法 |
| NSDI 2027 Fall | networked systems / distributed serving | abstract 2026-09-10，full 2026-09-17 | 6.5/10 | 除非 routing/serving 系统很强，否则偏难 |

来源：

1. ICLR 2026 CFP 显示 abstract 2025-09-19、full 2025-09-24，可作为 2027 预估参考；2027 官方 CFP 截至本文未检索到。
2. MLSys 2026 CFP 显示 full paper deadline 为 2025-10-30，2027 官方 CFP 截至本文未检索到。
3. EuroSys 2027 官方 CFP：Fall abstract 2026-09-17，full 2026-09-24。
4. ASPLOS 2027 官方 CFP：September Cycle full paper 2026-09-09。
5. NSDI 2027 官方 CFP：Fall abstract 2026-09-10，full 2026-09-17。

### 6.2 第二梯队：可作为备份或不同叙事路线

| Venue | 方向 | Deadline | 匹配度 | 建议 |
|---|---|---:|---:|---|
| ARR August 2026 -> EACL 2027 | NLP / agent / efficiency evaluation | ARR submission 2026-08-03 | 6.5/10 | 时间太紧，除非写短分析/benchmark paper |
| ARR October 2026 | NLP / agent / prompt/context | ARR submission 2026-10-12 | 7/10 | 如果 AI 叙事强，可作为 ICLR 失败/错过后的备份 |
| EMNLP 2026 Industry Track | real-world NLP systems | 2026-06-16 | 5.5/10 | 太近，除非已有完整系统，不建议硬冲 |
| NeurIPS 2026 Workshop | workshop / feedback | suggested 2026-08-29 | 7/10 | 非主会，但适合拿反馈和建立 visibility |
| AAMAS 2027 | autonomous agents / multi-agent systems | 官方未发布；AAMAS 2026 为 2025-10-01 abstract、2025-10-08 full | 7/10 | 如果强调 agent framework/agent design，可考虑 |
| AAAI 2027 | broad AI | 官方未发布；AAAI-26 paper deadline 为 2025-08-01 | 6/10 | 需要更 AI 方法化，纯 infra 不太合适 |
| AISTATS 2027 | ML/statistics | 官方未发布；AISTATS 2026 abstract 2025-09-25、full 2025-10-02 | 4.5/10 | 除非有理论/统计优化，不推荐 |

来源：

1. ARR 官方 dates：August 2026 submission 为 2026-08-03，October 2026 submission 为 2026-10-12；EACL 2027 final ARR submission 为 2026-08-03。
2. EMNLP 2026 Industry Track 官方 CFP：submission deadline 2026-06-16。
3. NeurIPS 2026 官方 dates：workshop contributions suggested submission date 2026-08-29；main conference abstract/full 已在 2026-05 上旬截止。
4. AAMAS 2026 官方 CFP：abstract 2025-10-01，paper 2025-10-08；AAMAS 2027 截至本文未检索到官方 CFP。
5. AAAI-26 官方 CFP：paper deadline 2025-08-01；AAAI 2027 截至本文未检索到官方 CFP。
6. AISTATS 2026 官方 CFP：abstract 2025-09-25，full 2025-10-02；AISTATS 2027 截至本文未检索到官方 CFP。

### 6.3 不建议作为主目标的路线

| Venue | 原因 |
|---|---|
| NeurIPS 2026 Main | 2026-05-04 abstract、2026-05-11 full 已错过 |
| ICML 2026 Main | 2026-01-28 full 已错过 |
| COLM 2026 Main | 2026-03-31 full 已错过 |
| ATC 2026 | 2026-06-10 太近，且系统实现要求高 |
| SoCC 2026 | 2026-02-13 已错过 |
| SIGCOMM 2026 | 方向不匹配且 deadline 已过 |

## 7. 推荐投递路线

### Route A: AI 顶会主线

目标：

```text
ICLR 2027 -> ARR October / AAMAS 2027 -> ICML 2027
```

适用条件：

1. Context planner 有清楚算法/框架抽象。
2. 质量-成本曲线扎实。
3. 至少 2 类真实 agent task。
4. 论文叙事偏 agent design / efficient tool-use。

内部 DDL：

1. 2026-07-31：Planner MVP 完成。
2. 2026-08-15：真实 workload + quality metrics 完成。
3. 2026-08-25：ICLR paper v0。
4. 2026-09-05：ICLR paper v1 + all figures。
5. 2026-09-15：冻结实验。
6. 2026-09 下旬：按 ICLR 2027 官方 deadline 提交。

如果 ICLR 赶不上：

1. 2026-10-12：ARR October 2026。
2. 2026-10 上旬：AAMAS 2027，如果官方 CFP 发布且匹配。
3. 2027-01：ICML 2027，如果 paper 已显著增强。

### Route B: MLSys / Systems 主线

目标：

```text
MLSys 2027 -> EuroSys 2027 Fall / ASPLOS 2027 / NSDI 2027
```

适用条件：

1. 有真实系统 prototype。
2. 接入 vLLM 或 SGLang，不只是离线 prompt builder。
3. 有端到端 serving throughput/latency/P95/P99。
4. 有 artifact 和复现实验。
5. 方法对 serving backend 有实际可利用 metadata 或 runtime integration。

内部 DDL：

1. 2026-07-31：Planner + backend runner 跑通。
2. 2026-08-20：end-to-end system result 完成。
3. 2026-09-01：paper v1。
4. 2026-09-09：ASPLOS 2027 September Cycle，可选。
5. 2026-09-17：NSDI 2027 Fall full，可选。
6. 2026-09-24：EuroSys 2027 Fall full，可选。
7. MLSys 2027 官方 CFP 发布后，按其 deadline 决定是否主投。

风险：

```text
如果只是外部 prompt planning，systems venue 可能认为系统贡献不够。
```

### Route C: 快速反馈路线

目标：

```text
NeurIPS 2026 Workshop / EMNLP Workshop / COLM Workshop
```

适用条件：

1. 主会还不成熟。
2. 想先拿同行反馈。
3. 不希望 workshop 版本影响后续主会投稿。

建议：

1. 2026-08-20 前准备 4-8 页 workshop version。
2. 只放 measurement + early planner，不要把完整主会贡献提前 archival 化。
3. 明确 workshop 是否 archival，避免影响后续主会。

## 8. 关键 Go / No-Go 判断点

### 2026-06-30 判断

Go 条件：

1. 至少 2 个 workload 跑通。
2. PSR/CRO/COG 指标能产生有意义差异。
3. prefix/cacheability 问题不是人工制造的。

No-Go 处理：

```text
如果真实任务上 cacheability 差异不明显，改成 Agent inference characterization / benchmark paper，不继续硬冲顶会。
```

### 2026-07-31 判断

Go 条件：

1. Context Planner MVP 能自动产生 planned prompt。
2. 至少一个真实 workload 上质量不降。
3. 至少一个 serving 指标有稳定收益。

No-Go 处理：

```text
如果收益只来自 token truncation，必须回到 observation compression 的质量约束问题，避免和 prompt compression 撞车。
```

### 2026-08-20 判断

Go 条件：

1. 至少 2 类真实 workload。
2. 至少 6 张核心图。
3. 至少一个端到端 result table。
4. 相关工作边界清晰。

Venue 决策：

1. AI 叙事更强：ICLR。
2. 系统实现更强：MLSys / EuroSys / ASPLOS / NSDI。
3. 都不够强：workshop + arXiv + 下一轮主会。

### 2026-09-01 判断

Go 条件：

1. full paper v1 完整。
2. 结果可以复现。
3. failure cases 已写。
4. limitations 不致命。

No-Go 处理：

```text
不硬投顶会。转向 ARR October / AAMAS / MLSys 2027 / ICML 2027。
```

## 9. 实验最低资源需求

### 9.1 最小可行资源

1. 单张 24GB-48GB GPU。
2. 7B/8B 模型，例如 Qwen2.5-7B、Qwen3-8B、Llama-3.1-8B。
3. vLLM 或 SGLang 至少一个完整 backend。
4. mock backend 用于 trace 和 context planning 测试。

可完成：

1. measurement。
2. context planner。
3. BFCL / synthetic / small SWE-Bench Lite。
4. arXiv / workshop / early main submission。

### 9.2 更强资源

1. 单张 80GB GPU 或多张 48GB GPU。
2. 14B/32B 模型。
3. vLLM + SGLang 双 backend。
4. 并发 16-64。

可完成：

1. 更强系统实验。
2. P95/P99。
3. cross-backend validation。
4. 更适合 MLSys / EuroSys / ASPLOS。

## 10. 写作结构

推荐 paper structure：

1. Introduction
2. Background and Motivation
3. Agent Context Cacheability Characterization
4. Cacheability Metrics
5. CacheWeaver Design
6. Implementation
7. Evaluation
8. Failure Cases and Discussion
9. Related Work
10. Conclusion

Introduction 必须讲清楚三件事：

1. Tool-using agents are expensive because they repeatedly prefill growing and unstable contexts.
2. Existing serving cache systems can only exploit reuse after prompts are constructed.
3. Agent frameworks should construct cache-friendly contexts under quality constraints.

## 11. 立即行动清单

从今天开始，优先做这些：

1. 不再把 routing simulator 作为主贡献。
2. 在现有 workload generator 中加入 segment-level trace schema。
3. 实现 PSR/CRO/COG。
4. 补 BFCL 或 function-calling trace。
5. 给每个实验都加 quality metric。
6. 实现 Context Planner MVP，而不是只手工改 prompt。
7. 每周固定产出一份 paper-facing report，不只写代码日志。
8. 8 月 20 日做 venue decision，不拖到 deadline 前。

## 12. 最终建议

最推荐路线：

```text
主线：ICLR 2027 / MLSys 2027
备份：ARR October 2026 / AAMAS 2027 / NeurIPS Workshop
系统强则：EuroSys 2027 Fall 或 ASPLOS 2027 September
```

如果只能选一个最优主目标：

```text
优先按 ICLR 2027 准备，但实验标准按 MLSys 准备。
```

原因：

1. ICLR 更接受 agent/context/quality-cost 的 AI 叙事。
2. MLSys 的实验标准能强迫工作变扎实。
3. 如果 paper 最后系统味更强，可以切 MLSys/EuroSys；如果 AI 方法味更强，可以切 ICLR/ARR/AAMAS。

最重要的执行原则：

```text
不要为了赶 deadline 提交一个“prompt trick paper”。
要提交一篇清楚定义 agent context cacheability、提出自动 planner、并在真实任务上证明质量-成本收益的论文。
```

