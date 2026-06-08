# Agent Context Compiler：面向工具调用智能体的上下文编译器设计草案

更新时间：2026-06-04

## 1. 定位

Agent Context Compiler: Diagnosing and Repairing Cacheability Breakdowns in Long-Observation Tool-Using Agents

agent场景定位：
Dynamic Tool-Using Agents with Long Observations。面向代码修复/函数调用/检索验证任务的单agent或轻量多轮agent。tool schema相对稳定，observation长且动态，history持续增长，任务质量对细节高度敏感。
这里的agent不聚焦大型网络化多智能体或者多agent通信缓存

代码修复Agent: SWE-Bench Lite / repo debugging
函数调用Agent: BFCL / ToolBench-style tasks
检索验证Agent：HoVer / HotpotQA / DeepResearch-style traces


工具调用型 Agent 经常把稳定的工具定义/系统指令和动态增长的长 observation 一起反复发送。
现有 serving 侧缓存只能在 prompt 构造之后被动复用；
简单截断或摘要又会删掉任务关键细节。


## 2. 实验顺序

1. BFCL/function calling
最容易先做质量闭环。
证明 tool schema canonicalization 和 layout planning 不伤 JSON validity/tool accuracy。

2. SWE-Bench Lite/coding traces
最有论文说服力。
证明 recoverable observation compression 比 truncation 更安全，因为日志/路径/错误细节很关键。

3. HoVer/HotpotQA retrieval-agent traces
补第三类工具任务。
证明 evidence compression 需要保留 source/evidence，而不是普通摘要。

4. WebArena/VisualWebArena
只做 optional。
因为 PANDO 太接近，不要作为主战场。

## 3. 主要的创新点

context compiler不只是进行提示词工程，把提示词进行稳定、半稳定和易改变等分层。而是以下三个方面：
1. 可量化的cacheability diagnostics
segment-level cacheability metrics + fragmentation diagnosis
```
PSR: stable prefix ratio
DPR: dynamic pollution ratio
CRO: cache reuse opportunity
COG: cache opportunity gap
Fragmentation Ratio: 同一 workflow 产生多少不同 stable keys
```
之前没有人系统回答，agent上下文到底有多cacheable；哪里破坏了cache，tool schema drift / dynamic header / observation？ ；理论reuse opportunity和真实TTFT收益之间的差别
2. 自动compiler repair，而非手动prompt engineering
```
自动解析 agent context
自动发现 cache fragmentation
自动重排/规范化/压缩
自动检查质量约束
输出 compiled prompt + metadata
```
diagnose and repair，而非 organize prompts
3. 质量约束下的recoverable observation compression
agent任务里不能随便压缩history/observation，因为关键细节可能是file path / line number / error code / tool argument / JSON required field / evidence source / API status
重点是在质量约束下把长的observation变成summary + pointer，必要时可以retrieve原始的observation

---
总结来说，我发现agent虽然常有稳定上下文，但实际 cacheability 被 dynamic pollution、schema drift 和 observation growth 系统性破坏。
我们提出 Agent Context Compiler，用 segment-level diagnostics 找出 cacheability 破坏源，并在任务质量约束下自动 repair context layout 和 observation representation。

诊断 + 修复 + 质量约束 + serving 验证

Agent 上下文在结构上看似有大量可复用内容，但在实际运行中经常因为动态污染、schema drift 和长 observation 增长而无法被 serving cache 利用。
---
我们研究质量约束下的可恢复 observation 表示，
并将其纳入 Agent Context Compiler，使其同时优化 task quality、context cacheability 和 serving latency。

我与一般observation contract不同之处：
可恢复的observation runtime + 质量契约 + cacheability/serving 组合收益

## 4. 可能的深化
profile-guided、quality-verified、serving-aware的上下文优化系统
1. profile-Guided Context Compilation
先运行Agent traces，收集哪些segment被频繁复用，哪些observation后续真的被用到，哪些字段影响成功率；再根据profile自动决定布局、压缩、保留和恢复策略

收集指标如：
```
segment reuse frequency
segment drift frequency
observation future-use rate
recovery trigger rate
tool failure correlation
cache benefit per segment
quality sensitivity per segment
```
决策：
```
高复用 stable segment -> 强制前置
低复用长 observation -> 压缩
历史上经常被恢复的字段 -> 默认保留
导致失败的字段 -> 加入 must_preserve
```
这样从静态规则变成了数据驱动的上下文编译

2. observation contract
Observation Contract = 每类工具 observation 必须满足的质量保真契约。
**coding tool**
must preserve: file path, line number, symbol, error type, failing test, exit code
recoverable: full stdout/stderr, full file snippet
discardable: duplicated logs, progress bars, repeated warnings
**检索工具**
must preserve: evidence sentence, source id, URL, timestamp, claim relation
recoverable: full document
discardable: boilerplate, navigation text, ads
**function calling**
must preserve: tool name, required args, enum values, type constraints

这样是 contract-aware observation representation

3. counterfactual cacheability analysis
可能有反直觉发现，可以对同一条agent trace做counterfactual rewrites
```
如果只移动 timestamp，会怎样？
如果只稳定 tool order，会怎样？
如果只压缩 observation，会怎样？
如果把 tools 放到 history 后面，会怎样？
如果把 observation 放到 prefix，会怎样？
```
量化 which context defect causes the largest cacheability loss?
这样可以定义 cacheability attribution

4. cost model / decision policy
可以建立一个轻量的cost model
```
ExpectedCost =
prefill_cost
+ decode_cost
+ recovery_cost
+ quality_risk_penalty
- cache_reuse_benefit
```
然后compiler对每个segment决定：
```
inline full
compress
summarize
externalize as pointer
drop
move earlier
move later
```
优化目标是
```
minimize expected serving cost
subject to quality constraints
```
形式：
```
min C(layout, compression)
s.t. QualityDrop <= epsilon
```

5. active recovery policy
将recoverable reference升级为 compiler/runtime根据不确定性自主决定是否恢复
触发信号：
```
模型生成 tool call 低置信
JSON validation 失败
tool argument 缺字段
patch localization 不确定
retrieval evidence conflict
模型请求了不存在的字段
```
通过实验证明
```
大多数请求用 compressed context；
少数失败请求触发 recovery；
总体成本低于 full context；
质量高于 irreversible compression。
```

6. Context IR / Cacheability Contract
提出一个中间表示 agent context IR
```
{
  "segments": [
    {
      "id": "tools",
      "type": "tool_schema",
      "stability": "static",
      "reuse_scope": "agent_type",
      "cache_key": "...",
      "quality_contract": {...}
    }
  ],
  "layout_policy": "...",
  "serving_hints": {
    "cache_scope": "global",
    "prefix_candidate": true
  }
}
```
这个IR可以让：
```
Agent framework 知道怎么构造 context
Compiler 知道怎么优化
Serving benchmark 知道怎么分析 cacheability
未来 serving engine 知道哪些 segment 可共享
```

5. 可能的新主线与贡献
Profile-Guided and Contract-Verified Context Compilation for Long-Observation Tool-Using Agents

核心组件：
```
1. Context IR
2. Cacheability diagnostics
3. Profile-guided compiler
4. Observation contract verifier
5. Recoverable reference + active recovery
6. Serving cache combination evaluation
```

可能的论文贡献：
1. We characterize cacheability breakdowns in long-observation tool-using agents and show that stable context is often structurally present but operationally uncacheable.
2. We introduce Agent Context IR and cacheability diagnostics to attribute cache loss to volatile-prefix pollution, schema drift, and observation growth.
3. We design a profile-guided, contract-verified context compiler that transforms raw agent contexts into cache-friendly layouts with recoverable observations.
4. We evaluate it across function-calling, coding, and retrieval agents, showing quality-preserved cost/latency reductions and complementary gains with vLLM/SGLang prefix caching.


## 5. 综合评估
上述研究计划包含内容过多，我需要决定保留什么，去掉什么，把一些留作未来工作
保留：
```
1.  Cacheability diagnostics（PSR/DPR/CRO + fragmentation）
2. 自动repair（canonicalization + layout planning）
3. Observation contract + recoverable reference
4. 质量验证
5. 在真实backend上serving验证
```
砍掉/放到future work:
```
1. Profile-guided compilation
  这部分单独就可做一篇论文，需要大量traces + 统计分析
2. Active recovery policy
  需要runtime的uncertainty detection，工程量太大
3. cost model / decision policy
  形式化的优化模型需要大量实验验证
4. Context IR作为标准提案
  生态级别的贡献，一篇论文不够，在method里简单描述segment schema就足够了
5. Counterfactual cacheability analysis
  可以作为实验中的一种分析方法保留，但是可能不是核心贡献？
```

现在论文叙事聚焦在三件事：
1. DIAGNOSE
   我们提出segment-level cacheability metrics，
   定量回答"agent上下文到底有多cacheable？
   哪些因素破坏了cache？"
2. REPAIR
   我们提出自动的context repair pipeline，
   包含canonicalization、layout reordering、
   和contract-aware observation compression。
3. RECOVER
   我们提出recoverable observation reference，
   让压缩不再是不可逆的。
   在质量约束下实现cache-friendly的observation表示。
然后再机上理论分析，端到端serving实验和质量验证


现在论文taste的考虑：
"structurally present but operationally uncacheable" 
diagnose + repair 的叙事
observation contract
这些都是好的

但是有一些需要修改的：
1. 指标太多了。PSR, DPR, CRO, COG, Fragmentation Ratio, QPCR...审稿人会迷失，
   建议：保留3个核心指标
   → PSR（稳定前缀占比，越高越好）
   → DPR（动态污染比，越低越好）  
   → CRO（复用机会，越高越好）
   其他的放到appendix或作为辅助分析
2. "深化"部分太发散
   profile-guided、active recovery、cost model、context IR...
   每一个都是好想法，但放在一起给人"还没想清楚要做什么"的感觉
   建议：在论文里只提及核心方法，
   其他的写一句"we leave X to future work"
3. 缺少一个killer example
   需要在Introduction里放一个具体的例子：
   "这是一个真实的coding agent trace，
    它的observation有3000 tokens，
    但其中只有200 tokens在后续轮次被使用。
    这2800个tokens不仅浪费了prefill成本，
    还破坏了前面稳定prefix的cache hit。
    经过我们的compiler处理后，
    observation被压缩到300 tokens + 1个recovery pointer，
    TTFT从Xms降到Yms，任务质量不变。"
4. observation contract需要数据支撑
    建议做一个轻量的数据分析支撑contract
    ```
    方法：
    1. 收集50条SWE-Bench agent traces
    2. 对每个observation，标注哪些token在后续轮次被LLM引用
      （简单方法：检查后续LLM输出中是否包含observation的子串）
    3. 统计每类token的"后续引用率"

    预期发现：
      file_path: 被引用率 85%     → must_preserve
      error_message: 被引用率 78% → must_preserve  
      line_number: 被引用率 72%   → must_preserve
      full_stderr: 被引用率 11%   → recoverable
      progress_bar: 被引用率 0%   → discardable

    这样你的contract就不是拍脑袋的，
    而是"data-driven observation contract"。
    ```
    这个分析本身也可以作为论文的一个小贡献，放在motivation section里



理论部分的加强：
理想的理论层次：
```
Level 0（你现在有的）: 指标定义
  → "PSR = stable_prefix_tokens / total_input_tokens"
  → 这只是notation，不算理论贡献

Level 1（最低要求）: 基于指标的性质定理
  → "在prefix-matching cache下，排列X最大化PSR"
  → 这是一个有证明的结论

Level 2（理想目标）: 将repair建模为优化问题
  → "compiler的repair目标可以形式化为..."
  → 给出最优解或近似保证

Level 3（stretch goal）: 与serving系统的联合分析
  → "compiler的PSR提升如何翻译成TTFT降低"
  → 这需要serving系统的cost model
```
优先建议以下俩个定理：
**定理1：排列最优性（你能证的最重要的定理）**
给定 n 个segments，每个长度 $l_i$，变化概率 $p_i$，最大化期望LCP的排列是按 $l_i(1−p_i)/p_i$降序。

为什么这个定理有价值：

- 它不是trivial的——当segment长度不同时，最优排列不是简单的"稳定优先"
- 它给出了一个可计算的公式，compiler的Layout Planning pass可以直接用
- 证明用交换论证（exchange argument），优雅且简短

**定理2：Dynamic Pollution的传播性（直觉性强的定理）**

在strict prefix-matching下，一个长度为 d 的ephemeral token插入在位置 k（从prompt开头算），导致的cache hit损失为：

$$ΔHit=L_{total}−k$$

即插入位置越靠前，damage越大。特别地，一个1-token的timestamp放在prompt最开头，导致的cache loss等于整个prompt的长度。

为什么这个定理有价值：
极其直观但很少有人写出来
它justify了你的repair策略"move ephemeral fields to the end"
它可以配合一个图：pollution位置 vs cache loss的曲线



反直觉发现 counterfactual cacheability analysis
```
实验设计：Cacheability Attribution

对同一条agent trace，做以下counterfactual rewrites：
  CF1: 只移除timestamp/session_id → 重新计算PSR/CRO
  CF2: 只稳定tool ordering → 重新计算PSR/CRO
  CF3: 只压缩observation → 重新计算PSR/CRO
  CF4: 只做layout reordering → 重新计算PSR/CRO
  CF5: Full compiler → 重新计算PSR/CRO

然后计算每个因素的"cacheability attribution"：
  Attribution(factor_i) = PSR(with_fix_i) - PSR(original)

这会告诉你：
  "在coding agent中，cache的主要破坏者是XXX"
  "在function calling agent中，cache的主要破坏者是YYY"
```
**如果发现不同类型的agent有完全不同的cacheability bottleneck——这就是反直觉发现。**
例如：
- "在coding agent中，92%的cache loss来自long observation，只有3%来自schema drift"
- "在function calling agent中，61%的cache loss来自schema ordering instability"
- "在所有agent中，timestamp/session_id这种看似微小的pollution，平均导致了X%的额外prefill成本"


初步考虑的论文结构:
```
标题：
Agent Context Compiler: Diagnosing and Repairing 
Cacheability Breakdowns in Tool-Using Agents

Abstract (0.25页)

1. Introduction (1页)
   - 问题：agent上下文在结构上有大量可复用内容，
     但实际cacheability被系统性破坏
   - Killer example：一个coding agent trace的具体数据
   - 三个贡献：diagnose, repair, recover
   - 关键结果数字

2. Background & Motivation (1页)
   - 2.1 KV Cache + Prefix Matching（简短）
   - 2.2 Agent Context结构分析
   - 2.3 Motivating Study: 
     "我们分析了X条agent traces，发现..."
     用数据展示cacheability breakdown的严重性

3. Cacheability Diagnostics (1页)  ← Contribution 1
   - 3.1 Segment-level指标：PSR, DPR, CRO
   - 3.2 Counterfactual Attribution Analysis
     → 回答"哪个因素导致了最大的cache loss"
   - 3.3 理论分析：排列最优性定理 + pollution传播定理

4. Context Repair Pipeline (1.5页)  ← Contribution 2
   - 4.1 Canonicalization（快速过，0.25页）
   - 4.2 Layout Reordering（快速过，0.25页）
   - 4.3 Contract-Aware Observation Compression（重点，0.5页）
     → observation contract的定义
     → must_preserve / recoverable / discardable
   - 4.4 Recoverable Reference（重点，0.5页）
     → summary + pointer机制
     → recovery触发条件

5. Evaluation (2.5页)  ← Contribution 3 & 4
   - 5.1 实验设置
   - 5.2 Cacheability Diagnostics结果
     → 不同agent类型的PSR/DPR/CRO
     → Counterfactual attribution发现
   - 5.3 Serving延迟实验
     → TTFT on vLLM/SGLang
   - 5.4 质量验证
     → BFCL: tool accuracy
     → SWE-Bench: patch success
   - 5.5 Ablation
   - 5.6 Recoverable Reference的价值
     → recovery触发频率
     → recovery对质量的挽救效果

6. Related Work (0.5页)

7. Conclusion + Future Work (0.25页)
   Future work: profile-guided compilation,
   active recovery, cost model optimization
```