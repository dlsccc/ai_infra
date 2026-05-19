# Agent 推理系统优化 8 周冲刺计划

更新时间：2026-05-19  
定位：两个月内产出一个可投递、可展示、可继续扩展成论文的 Agent Inference Infra 项目。

## 0. 总目标

主项目：

```text
AgentInferBench:
面向工具调用 Agent 的 LLM Serving Benchmark 与 Cache-Aware 优化实验平台
```

两个月必须交付：

1. 一个可运行 GitHub 项目：支持 vLLM / SGLang 的 Agent workload benchmark。
2. 一份技术报告：Agent 场景与普通 chat 场景的推理特征差异。
3. 一个轻量优化：prompt canonicalization + stable tool ordering + session/cache-aware routing simulator。
4. 一篇高质量技术博客：可用于简历和面试预热。
5. 一版 paper-style report：如果实验结果足够好，再投 workshop / CCF-C / arXiv。

不作为 8 周硬目标：

1. 不承诺复现 ToolSpec / Continuum / CONCUR。
2. 不承诺改 vLLM / SGLang 内部 KV cache manager。
3. 不承诺实现真正的 schema-guided speculative decoding。
4. 不承诺做多 GPU PD disaggregation 实验。
5. 不承诺 AAAI / ASPLOS / OSDI 级别主会投稿。

## 1. 成功标准

### 1.1 求职成功标准

到 Week 12 结束，简历中可以写出如下项目：

```text
AgentInferBench: Agent 场景 LLM Serving Benchmark 与 Cache-Aware 优化
- 构建 4 类 Agent workload，覆盖单工具调用、多工具串行、长 observation、多 Agent 并发。
- 对比 vLLM / SGLang 在 Agent 场景下的 TTFT、TPOT、JCT、P95 latency、吞吐和显存。
- 发现 Agent 多轮交互中 prefix overlap 高，但实际 cache reuse 受 prompt layout、tool schema 顺序、session 路由影响明显。
- 实现 prompt canonicalization、stable tool ordering、session-aware routing 和 cache-aware routing simulator。
- 在典型场景下降低 JCT / P95 TTFT X%，并给出 ablation、失败 case 和 tradeoff 分析。
```

### 1.2 论文雏形标准

到 Week 12，如果满足以下条件，就整理成 workshop / CCF-C / arXiv 版本：

1. 至少支持 vLLM 和 SGLang 两个 backend。
2. 至少有 3 类稳定 Agent workload。
3. 至少有 5 张稳定图表。
4. 至少一个轻量优化在多个实验点上稳定有效。
5. 能解释什么时候有效、什么时候无效、为什么。

如果不满足，则先发布技术博客和 GitHub 项目，不强行投稿。

## 2. 总时间线

```text
Week 5  (5/19-5/25): 环境搭建 + benchmark MVP + SGLang/vLLM 基础认知
Week 6  (5/26-6/01): Baseline 实验 + Agent workload characterization
Week 7  (6/02-6/08): Prefix cache 深度实验 + 相关论文阅读
Week 8  (6/09-6/15): 轻量优化实现：canonicalization + routing simulator
Week 9  (6/16-6/22): 并发实验 + ablation + 可选小型 spec/quant 补充
Week 10 (6/23-6/29): 项目工程化 + README + 图表自动化 + 技术报告初稿
Week 11 (6/30-7/06): 技术博客 + 简历 + 开始投递 + 面试准备
Week 12 (7/07-7/12): Paper-style report + 求职材料定稿 + 后续投稿判断
```

## 3. 项目目录结构

```text
projects/agent_infer_bench/
  README.md
  pyproject.toml 或 requirements.txt
  configs/
    env.yaml
    baseline_vllm.yaml
    baseline_sglang.yaml
    agent_workloads.yaml
    routing_experiments.yaml
  agent_bench/
    __init__.py
    backends/
      base.py
      vllm_backend.py
      sglang_backend.py
    workloads/
      generator.py
      schemas.py
      traces.py
    metrics/
      collector.py
      latency.py
      cache.py
    optimizations/
      prompt_canonicalization.py
      stable_tool_ordering.py
      session_routing.py
      cache_routing_simulator.py
    analysis/
      summarize.py
      plot.py
      workload_characterization.py
  scripts/
    setup/
      check_gpu_env.sh
    benchmark/
      run_baseline.py
      run_agent_workloads.py
      run_prefix_cache.py
      run_routing_ablation.py
    analysis/
      make_tables.py
      make_figures.py
  experiments/
    runs/
    reports/
  docs/
    weekly/
    notes/
    reports/
```

## 4. 固定实验指标

每次实验必须记录：

1. 环境：GPU 型号、显存、驱动、CUDA、Python、PyTorch、vLLM、SGLang、模型版本。
2. 输入：backend、model、batch/concurrency、prompt length、output length、workload type、tool 数量、turn 数。
3. 延迟：TTFT、TPOT、end-to-end latency、job completion time、P50/P95。
4. 吞吐：tokens/s、requests/s。
5. 显存：peak memory、实验开始/结束显存。
6. cache proxy：shared prefix length、prefix overlap ratio、是否开启 prefix caching、命中估计值。
7. 稳定性：每组至少 3 次，记录 mean/std；资源紧张时至少 2 次。

注意：不要预设“必须提升 30% / 50%”。所有提升都是实验结果，不是计划承诺。

## 5. Week 5：环境搭建 + Benchmark MVP

目标：

1. 远程 GPU 环境可稳定跑 vLLM / SGLang。
2. 完成最小统一 benchmark 框架。
3. 读懂 vLLM PagedAttention 与 SGLang RadixAttention 的核心差异。

### Day 1：服务器环境与模型准备

任务：

1. 登录远程 GPU 服务器，记录硬件与软件环境。
2. 配置 Python 3.11、PyTorch、vLLM、SGLang。
3. 下载一个主模型：优先 Qwen2.5-7B-Instruct；如果已有 Qwen3-8B，也可用它。
4. 分别用 vLLM 和 SGLang 跑一次 hello world。
5. 建立项目目录 `projects/agent_infer_bench/`。

产出：

1. `projects/agent_infer_bench/docs/reports/server_setup.md`
2. `projects/agent_infer_bench/scripts/setup/check_gpu_env.sh`
3. `projects/agent_infer_bench/configs/env.yaml`

验收：

1. `nvidia-smi`、CUDA、PyTorch CUDA 均可用。
2. vLLM / SGLang 都能生成一段文本。

### Day 2：统一 Benchmark 数据格式

任务：

1. 定义统一输入格式：
   - `plain_chat`
   - `single_tool`
   - `multi_tool_serial`
   - `long_observation`
   - `multi_agent_concurrent`
2. 定义统一输出 JSON schema：
   - config
   - environment
   - raw_metrics
   - summary_metrics
   - errors
3. 编写 backend 抽象接口：
   - `generate(prompts, sampling_params)`
   - `stream_generate(prompts, sampling_params)`，如果暂时做不到可先留 TODO。
4. 先实现一个 mock backend，用于本地 CPU 环境验证数据流。

产出：

1. `agent_bench/backends/base.py`
2. `agent_bench/backends/mock_backend.py`
3. `agent_bench/metrics/collector.py`
4. `docs/notes/benchmark_schema.md`

验收：

1. mock backend 能跑完一组 workload 并落盘 JSON。
2. 输出格式后续不需要大改。

### Day 3：vLLM Backend MVP

任务：

1. 实现 vLLM backend。
2. 支持基础 sampling 参数：temperature、top_p、max_tokens。
3. 实现 TTFT / total latency 采集；如果 TTFT 暂时拿不到，先记录 total latency，并标注限制。
4. 跑 3 组小实验：
   - prompt 128 / output 64
   - prompt 512 / output 128
   - prompt 1024 / output 128

产出：

1. `agent_bench/backends/vllm_backend.py`
2. `scripts/benchmark/run_baseline.py`
3. `experiments/runs/week05/vllm_smoke/`

验收：

1. vLLM backend 可通过统一入口运行。
2. 结果 JSON 可被 `analysis/summarize.py` 读取。

### Day 4：SGLang Backend MVP

任务：

1. 实现 SGLang backend。
2. 使用与 vLLM 相同的 workload 和 sampling 参数。
3. 跑同样 3 组小实验。
4. 记录 SGLang server 启动参数与已知限制。

产出：

1. `agent_bench/backends/sglang_backend.py`
2. `experiments/runs/week05/sglang_smoke/`
3. `docs/notes/sglang_smoke_test.md`

验收：

1. vLLM / SGLang 可用同一配置文件运行。
2. 实验结果可横向比较。

### Day 5：Agent Workload Generator MVP

任务：

1. 实现 3 类 workload 生成器：
   - `single_tool`
   - `multi_tool_serial`
   - `long_observation`
2. 每类 workload 支持配置：
   - system prompt 长度
   - tool 数量
   - tool description 长度
   - turn 数
   - observation 长度
3. 计算每轮输入中的 prefix overlap ratio。
4. 保存 workload trace，保证可复现。

产出：

1. `agent_bench/workloads/generator.py`
2. `agent_bench/workloads/schemas.py`
3. `agent_bench/workloads/traces.py`
4. `experiments/runs/week05/workload_samples/`

验收：

1. 能生成至少 10 条 deterministic agent trace。
2. 每条 trace 能计算 token 统计和 prefix overlap。

### Day 6：基础图表与周报

任务：

1. 实现结果汇总脚本。
2. 生成最小图表：
   - latency vs prompt length
   - input tokens by turn
   - prefix overlap by turn
3. 写 Week 5 周报。

产出：

1. `agent_bench/analysis/summarize.py`
2. `agent_bench/analysis/plot.py`
3. `docs/weekly/week05.md`

验收：

1. 从实验 JSON 到图表的链路跑通。
2. README 初版包含如何跑 smoke test。

### Day 7：缓冲与风险清理

任务：

1. 修复环境和脚本问题。
2. 补齐 README 的安装说明。
3. 梳理 Week 6 实验矩阵，控制规模。

产出：

1. `README.md` 初版
2. `configs/baseline_vllm.yaml`
3. `configs/baseline_sglang.yaml`

## 6. Week 6：Baseline + Agent Workload Characterization

目标：

1. 产出 vLLM / SGLang 小规模但可信的 baseline。
2. 回答：Agent 场景和普通 chat 场景的推理负载有什么不同。
3. 形成论文/博客的 motivation 数据。

### Day 1：标准 Workload Baseline

任务：

实验矩阵控制在 24 组以内：

```text
backend: [vLLM, SGLang]
prompt_length: [128, 512, 2048]
output_length: [64, 256]
concurrency: [1, 8]
repeat: 3
```

记录：

1. TTFT 或首 token 近似指标。
2. TPOT 或 decode latency 近似指标。
3. total latency。
4. throughput。
5. peak memory。

产出：

1. `experiments/runs/week06/plain_baseline/`
2. `docs/reports/week06_plain_baseline.md`

验收：

1. 能画出 vLLM vs SGLang 的基础对比表。
2. 能说明每个指标的采集方式和误差来源。

### Day 2：Agent Workload Baseline

任务：

跑 3 类 Agent workload：

1. `single_tool`：1-2 轮，固定 tool schema。
2. `multi_tool_serial`：3-5 轮，context 逐轮增长。
3. `long_observation`：3-5 轮，每轮 observation 较长。

每类 workload：

```text
backend: [vLLM, SGLang]
concurrency: [1, 4]
repeat: 3
```

产出：

1. `experiments/runs/week06/agent_baseline/`
2. `docs/reports/week06_agent_baseline.md`

验收：

1. 每轮 JCT、输入 token、输出 token 都能落盘。
2. 至少生成 3 张图：JCT by turn、input tokens by turn、TTFT/latency by turn。

### Day 3：Agent 负载特征分析

任务：

分析以下问题：

1. Agent 每轮输入增长曲线。
2. system prompt / tool descriptions / history / current turn 的 token 占比。
3. prefix overlap ratio 随轮次变化。
4. long observation 对延迟的影响。
5. Agent workload 和 plain chat 在 TTFT/JCT 上的差异。

产出：

1. `agent_bench/analysis/workload_characterization.py`
2. `docs/reports/agent_workload_characterization.md`
3. 5 张核心图：
   - token breakdown
   - prefix overlap by turn
   - JCT by turn
   - latency vs observation length
   - plain chat vs agent comparison

验收：

1. 能用数据回答：“Agent 推理为什么不是普通 chat 的简单重复？”
2. 得到 2-3 个可验证的优化假设。

### Day 4：Prefix Cache 开关实验

任务：

1. vLLM：prefix caching 开启 vs 关闭。
2. SGLang：默认 radix cache 行为记录。
3. 对比 shared prefix length：
   - 0
   - 256
   - 512
   - 1024
   - 2048
4. 观察不同 shared prefix 下的 TTFT/JCT 变化。

注意：

1. 如果无法直接拿到 cache hit rate，记录 proxy：
   - shared prefix length
   - 首轮 vs 后续轮 latency
   - 相同 prefix 重复请求的 latency 改善
2. 不要声称真实 cache hit，除非 backend 明确暴露该指标。

产出：

1. `experiments/runs/week06/prefix_cache_switch/`
2. `docs/reports/week06_prefix_cache_switch.md`

验收：

1. 能说明 prefix caching 对 prefill-heavy 场景的收益。
2. 能说明它对 decode-heavy 场景的局限。

### Day 5：初步 Profiling

任务：

1. 选 2 个代表性 case 做 profiling：
   - plain chat long prompt
   - multi_tool_serial agent
2. 使用 torch profiler 或框架日志记录阶段耗时。
3. 如果 nsys 成本太高，本周不强行做深度 timeline。
4. 记录 profiling 方法的限制。

产出：

1. `docs/notes/profiling_quickstart.md`
2. `docs/reports/week06_bottleneck_notes.md`

验收：

1. 能区分 prefill-heavy 与 decode-heavy。
2. 能解释为什么 Agent 多轮常常造成重复 prefill 压力。

### Day 6：Week 6 报告初稿

任务：

1. 整理所有 Week 6 数据。
2. 写技术报告初稿：
   - 实验环境
   - workload 设计
   - baseline 结果
   - Agent 特征发现
   - 初步优化假设

产出：

1. `docs/reports/week06_agent_inference_characterization.md`
2. `docs/weekly/week06.md`

验收：

1. 这份报告可以作为博客和论文 Section 3 的基础。

### Day 7：缓冲与复盘

任务：

1. 补跑异常数据点。
2. 砍掉不稳定实验。
3. 明确 Week 7 只围绕 prefix/cache 做，不扩散。

产出：

1. `docs/reports/week06_experiment_review.md`

## 7. Week 7：Prefix Cache 深度实验 + 论文阅读

目标：

1. 找出影响 Agent prefix reuse 的工程因素。
2. 为 Week 8 的轻量优化提供依据。
3. 读相关论文，但不陷入复现。

### Day 1：论文阅读与问题定义

任务：

阅读并各写 0.5-1 页笔记：

1. vLLM / PagedAttention。
2. SGLang / RadixAttention。
3. LMCache。
4. KVFlow 或 Continuum，二选一深入。

每篇笔记固定回答：

1. 它解决什么问题。
2. 它的关键机制是什么。
3. 它和 AgentInferBench 有什么关系。
4. 哪些部分本项目不做。

产出：

1. `docs/notes/paper_reading_week07.md`

验收：

1. 能清楚讲出 PagedAttention、RadixAttention、prefix caching、KV cache offload 的区别。

### Day 2：Prompt Layout 实验

任务：

测试不同 prompt layout 对 prefix reuse 的影响：

1. layout A：固定 system + 固定 tools + history + current turn。
2. layout B：system + history + tools + current turn。
3. layout C：每轮 tool schema 顺序随机。
4. layout D：tool schema 中加入动态字段，例如时间戳、session id。

记录：

1. JCT。
2. 首轮与后续轮 latency。
3. prefix overlap proxy。
4. 输出正确性是否明显受影响。

产出：

1. `experiments/runs/week07/prompt_layout/`
2. `docs/reports/week07_prompt_layout.md`

验收：

1. 能证明 prompt layout 会影响 cache reuse 或 prefill cost。
2. 找到 canonicalization 的设计动机。

### Day 3：Tool Schema 顺序与稳定性实验

任务：

实验变量：

1. tool 数量：[5, 10, 20]
2. tool schema 顺序：[stable, random]
3. tool description 是否包含动态信息：[no, yes]

记录：

1. input token。
2. prefix overlap ratio。
3. JCT / TTFT proxy。
4. 结果波动。

产出：

1. `experiments/runs/week07/tool_schema_stability/`
2. `docs/reports/week07_tool_schema_stability.md`

验收：

1. 能给出 stable tool ordering 的必要性或无效结论。
2. 如果实验无显著差异，也要记录原因。

### Day 4：Session 并发与 Cache Locality 实验

任务：

构造多 Agent 并发：

```text
num_agents: [1, 4, 8, 16]
turns_per_agent: 5
scheduling: [round_robin, session_contiguous]
backend: [vLLM, SGLang]
```

解释：

1. `round_robin`：多个 agent 轮流提交请求。
2. `session_contiguous`：同一 agent 尽量连续提交。

记录：

1. avg JCT。
2. P95 JCT。
3. latency by turn。
4. 显存峰值。

产出：

1. `experiments/runs/week07/session_locality/`
2. `docs/reports/week07_session_locality.md`

验收：

1. 能判断 session locality 是否影响性能。
2. 为 Week 8 session-aware routing simulator 提供数据。

### Day 5：Speculative Decoding 小实验（降级版）

任务：

只做验证，不做 ToolSpec 复现：

1. 阅读 speculative decoding 基础原理。
2. 如果 vLLM 环境支持，跑 2-3 个 case：
   - plain QA
   - JSON/function-call style output
   - free-form output
3. 记录 acceptance rate、throughput 或 latency。
4. 如果环境不支持，写清楚原因，作为 future work。

产出：

1. `docs/notes/spec_decode_quick_note.md`
2. `experiments/runs/week07/spec_decode_optional/`，如果跑通。

验收：

1. 能解释 acceptance rate 为什么影响加速。
2. 不把 spec decode 放进主线优化。

### Day 6：Week 7 综合分析

任务：

1. 汇总 prompt layout、tool schema、session locality 的结果。
2. 提炼 Week 8 要实现的轻量优化：
   - prompt canonicalization
   - stable tool ordering
   - session-aware routing simulator
   - cache-aware routing simulator

产出：

1. `docs/reports/week07_prefix_cache_deep_dive.md`
2. `docs/weekly/week07.md`

验收：

1. Week 8 的优化方案有数据依据。

### Day 7：缓冲

任务：

1. 补跑异常实验。
2. 删除或标记不可信数据。
3. 更新项目 README 的实验说明。

## 8. Week 8：轻量优化实现

目标：

1. 实现不改 serving engine 内核的优化策略。
2. 让优化可以被复现、ablation、解释。

### Day 1：Prompt Canonicalization

任务：

实现 canonicalization：

1. 移除 tool schema 中非必要动态字段。
2. 统一 JSON key 顺序。
3. 统一空格、换行、分隔符。
4. 将 system prompt、tool schema、history、current turn 严格分区。
5. 给每个分区计算 hash，便于分析重复性。

产出：

1. `agent_bench/optimizations/prompt_canonicalization.py`
2. `tests/test_prompt_canonicalization.py`
3. `docs/notes/prompt_canonicalization_design.md`

验收：

1. 同语义不同格式的 tool schema 能规范成稳定文本。
2. 不改变模型可读性。

### Day 2：Stable Tool Ordering

任务：

1. 实现按 tool name / schema hash 排序。
2. 对比 stable ordering 和 random ordering。
3. 记录 token 级 prefix overlap 的变化。
4. 确认输出的 tool call 格式不被破坏。

产出：

1. `agent_bench/optimizations/stable_tool_ordering.py`
2. `tests/test_stable_tool_ordering.py`
3. `experiments/runs/week08/stable_tool_ordering/`

验收：

1. 相同 tool 集合在不同输入顺序下生成相同规范 prompt。
2. 有一张图展示 overlap 或 latency 的变化。

### Day 3：Session-Aware Routing Simulator

任务：

实现模拟器，不直接改生产 engine：

1. 输入：agent 请求序列、session id、prefix hash、backend instance 数量。
2. 策略：
   - random routing
   - round robin
   - session sticky
   - prefix hash routing
3. 输出：
   - same-session locality
   - prefix locality
   - estimated cache reuse opportunity
   - load balance 指标

产出：

1. `agent_bench/optimizations/session_routing.py`
2. `agent_bench/optimizations/cache_routing_simulator.py`
3. `tests/test_session_routing.py`

验收：

1. 可以在离线 trace 上比较不同路由策略。
2. 能看到 locality 与 load balance 的 tradeoff。

### Day 4：将优化接入 Benchmark

任务：

1. 在 benchmark 配置中加入：
   - `canonicalization: true/false`
   - `stable_tool_ordering: true/false`
   - `routing_strategy`
2. 跑小规模端到端实验：
   - baseline
   - canonicalization only
   - stable ordering only
   - canonicalization + stable ordering
3. 对 session routing 先做离线 simulator；如果多实例环境可用，再做真实路由。

产出：

1. `configs/routing_experiments.yaml`
2. `scripts/benchmark/run_routing_ablation.py`
3. `experiments/runs/week08/optimization_smoke/`

验收：

1. 每个优化开关可独立打开/关闭。
2. ablation 数据格式统一。

### Day 5：优化机制文档

任务：

写清楚：

1. 这个优化解决什么问题。
2. 它不解决什么问题。
3. 它和真实 KV cache manager 的关系。
4. 它为什么对 Agent 有意义。
5. 可能的负面影响：
   - canonicalization 改变 prompt 表达。
   - sticky routing 导致负载不均。
   - prefix routing 在 cache pressure 高时收益下降。

产出：

1. `docs/reports/week08_optimization_design.md`
2. 方法图草稿：Agent prompt 分层 + routing simulator。

验收：

1. 面试时不会被误解为“你改了 vLLM 内核”。
2. 能诚实说明这是 engine-external 优化。

### Day 6：Week 8 实验汇总

任务：

1. 汇总优化前后结果。
2. 生成 ablation 初版图。
3. 写 Week 8 周报。

产出：

1. `docs/reports/week08_ablation_initial.md`
2. `docs/weekly/week08.md`

验收：

1. 至少一个优化在至少一个 Agent workload 上有可解释收益。
2. 如果无收益，也能解释为什么，并调整 Week 9 实验。

### Day 7：缓冲

任务：

1. 修复代码质量问题。
2. 补测试。
3. 准备 Week 9 并发实验。

## 9. Week 9：并发实验 + Ablation + 可选补充

目标：

1. 完成主项目核心实验。
2. 得到可写进简历和报告的 ablation。
3. 可选做小型量化或 spec decode，不影响主线。

### Day 1：Ablation 主实验

任务：

实验组合：

```text
Base
+ Canonicalization
+ Stable Tool Ordering
+ Canonicalization + Stable Tool Ordering
+ Canonicalization + Stable Tool Ordering + Session Routing Simulator
```

workload：

1. `single_tool`
2. `multi_tool_serial`
3. `long_observation`

backend：

1. vLLM
2. SGLang

产出：

1. `experiments/runs/week09/ablation_main/`
2. `docs/reports/week09_ablation_main.md`

验收：

1. 能给出每个优化模块的独立贡献。
2. 不稳定数据必须标注，不隐藏。

### Day 2：多 Agent 并发实验

任务：

并发设置：

```text
num_agents: [1, 4, 8, 16]
turns_per_agent: 5
workload: multi_tool_serial
routing: [round_robin, session_sticky, prefix_hash]
```

记录：

1. avg JCT。
2. P95 JCT。
3. throughput。
4. estimated cache locality。
5. load balance。

产出：

1. `experiments/runs/week09/concurrency/`
2. `docs/reports/week09_concurrency.md`

验收：

1. 能说明 session/prefix routing 的收益和代价。

### Day 3：失败 Case 与边界条件

任务：

主动找失败 case：

1. tool schema 每轮变化。
2. observation 超长且低复用。
3. agent 数量高导致 cache pressure。
4. prompt canonicalization 改变模型输出风格。
5. sticky routing 造成单实例热点。

产出：

1. `docs/reports/week09_failure_cases.md`

验收：

1. 至少写出 3 个方法不 work 的场景。
2. 每个失败 case 有数据或明确原因。

### Day 4：可选量化小实验

任务：

如果资源允许，跑小表：

```text
model: Qwen2.5-7B
precision: [BF16/FP16, AWQ INT4]
workload: single_tool, multi_tool_serial
metrics: latency, JSON validity, tool name accuracy
```

如果环境不支持，写调研笔记，不占主线。

产出：

1. `docs/notes/quantization_agent_optional.md`
2. `experiments/runs/week09/quant_optional/`，如果跑通。

验收：

1. 量化只作为附录或 future work，不作为核心贡献。

### Day 5：可选 Spec Decode 小实验补充

任务：

如果 Week 7 spec decode 跑通，则补 2 个 case：

1. JSON/function-call style。
2. free-form reasoning style。

目标：

1. 观察结构化输出是否 acceptance rate 更高。
2. 不实现 ToolSpec。

产出：

1. `docs/notes/spec_decode_agent_optional.md`
2. `experiments/runs/week09/spec_optional/`，如果跑通。

验收：

1. 只作为 discussion，不影响主项目。

### Day 6：核心结果定稿

任务：

整理 6 张核心图：

1. Agent token breakdown。
2. prefix overlap by turn。
3. vLLM vs SGLang baseline。
4. prompt layout / tool ordering impact。
5. ablation study。
6. concurrency / routing tradeoff。

产出：

1. `docs/reports/week09_core_results.md`
2. `docs/weekly/week09.md`

验收：

1. 这 6 张图可以支撑技术博客和简历。

### Day 7：缓冲

任务：

1. 补跑关键缺失数据。
2. 冻结主要实验范围。
3. Week 10 开始只整理，不再扩散新方向。

## 10. Week 10：工程化 + 技术报告初稿

目标：

1. 项目达到可展示状态。
2. README 能让别人复现实验。
3. 写出完整技术报告初稿。

### Day 1：代码整理

任务：

1. 清理脚本入口。
2. 删除临时 debug 文件。
3. 补充类型标注和必要注释。
4. 固定依赖版本。
5. 增加 `--dry-run` 或 mock backend 示例。

产出：

1. `requirements.txt` 或 `pyproject.toml`
2. `README.md` 安装章节
3. `tests/` 基础测试

验收：

1. 新环境能按 README 跑 mock demo。

### Day 2：复现实验脚本

任务：

1. 写统一命令：
   - run smoke
   - run baseline
   - run agent workload
   - run ablation
   - generate figures
2. 所有命令在 README 中列出。
3. 标注 GPU 资源要求和预估耗时。

产出：

1. `scripts/benchmark/run_all_smoke.py`
2. `scripts/analysis/make_tables.py`
3. `scripts/analysis/make_figures.py`

验收：

1. 用户可以从 raw results 生成报告图表。

### Day 3：README 展示版

任务：

README 必须包含：

1. 项目一句话介绍。
2. 为什么 Agent inference 不同。
3. 支持的 backend。
4. workload 类型。
5. 快速开始。
6. 核心结果图。
7. 方法说明。
8. Limitations。
9. Roadmap。

产出：

1. `README.md` 完整版

验收：

1. 面试官 5 分钟能看懂项目价值。

### Day 4：技术报告 Section 1-3

任务：

写报告：

1. Introduction。
2. Background：
   - LLM serving
   - KV cache
   - prefix caching
   - Agent workload
3. Workload Characterization。

产出：

1. `docs/reports/agent_infer_bench_technical_report.md`

验收：

1. Section 3 有 Week 6/7 的核心图。

### Day 5：技术报告 Section 4-5

任务：

写报告：

1. Method：
   - prompt canonicalization
   - stable tool ordering
   - session/cache-aware routing simulator
2. Evaluation：
   - baseline
   - ablation
   - concurrency
   - failure cases

产出：

1. `docs/reports/agent_infer_bench_technical_report.md` 完整初稿

验收：

1. 报告可以作为 paper-style draft 的基础。

### Day 6：图表质量与数字核对

任务：

1. 检查所有图表标题、坐标轴、单位。
2. 检查所有数字是否来自同一实验版本。
3. 记录异常值和解释。
4. 生成一份结果索引。

产出：

1. `experiments/reports/result_index.md`
2. `docs/weekly/week10.md`

验收：

1. 没有“图里一个数、文里一个数”的问题。

### Day 7：缓冲

任务：

1. 找一个朋友或自己隔天 review README。
2. 修复最影响展示的问题。

## 11. Week 11：博客 + 简历 + 开始投递

目标：

1. 项目开始对外展示。
2. 简历可投递。
3. 同步准备面试。

### Day 1：技术博客大纲与素材

任务：

博客题目：

```text
Agent 推理为什么更贵：vLLM / SGLang 场景下的负载分析与优化实验
```

大纲：

1. Agent 推理和普通 chat 的差异。
2. 实验设置。
3. workload characterization。
4. vLLM vs SGLang baseline。
5. prefix/cache 相关发现。
6. 轻量优化与 ablation。
7. 工程经验和局限。

产出：

1. `docs/blogs/agent_inference_optimization_blog_draft.md`

验收：

1. 所有图表和表格引用路径确定。

### Day 2：博客初稿

任务：

1. 写完整博客初稿。
2. 控制在 3000-6000 中文字。
3. 每个结论必须有数据支撑。
4. 不夸大，不写“业界首个”等无法证明的话。

产出：

1. `docs/blogs/agent_inference_optimization_blog_draft.md`

验收：

1. 读者能复现实验或理解方法边界。

### Day 3：简历项目描述

任务：

写 3 个版本：

1. 详细版：用于简历项目经历。
2. 精简版：用于招聘平台。
3. 口述版：用于面试自我介绍。

模板：

```text
AgentInferBench: Agent 场景 LLM Serving Benchmark 与 Cache-Aware 优化
- 构建 vLLM/SGLang 统一 benchmark，覆盖 4 类工具调用 Agent workload。
- 系统分析多轮 Agent 的 token 增长、prefix overlap、JCT/P95 latency 和显存变化。
- 发现 prompt layout、tool schema 顺序和 session locality 会影响 cache reuse 机会。
- 实现 prompt canonicalization、stable tool ordering、session/cache-aware routing simulator。
- 在 XXX 场景下将 XXX 指标改善 X%，并分析高并发/低复用场景下的失效原因。
```

产出：

1. `docs/interview/resume_project_agent_infer_bench.md`

验收：

1. 所有数字都有实验来源。

### Day 4：面试问题准备

任务：

准备 15 个高频问题：

1. KV cache 是什么，为什么重要。
2. PagedAttention 解决了什么。
3. RadixAttention 和 prefix caching 的关系。
4. prefill 和 decode 的瓶颈不同在哪里。
5. Agent workload 与 chat workload 有什么不同。
6. TTFT、TPOT、JCT 如何定义。
7. 你的 benchmark 如何保证公平。
8. 你的优化为什么有效。
9. 什么时候无效。
10. 为什么不直接改 vLLM 内核。
11. sticky routing 的负载均衡风险。
12. prompt canonicalization 是否会影响模型行为。
13. spec decode 为什么没有作为主线。
14. PD disaggregation 对 Agent 有什么意义。
15. 如果要生产化，下一步怎么做。

产出：

1. `docs/interview/agent_infra_qa.md`

验收：

1. 每题都有 1 分钟和 3 分钟两个版本答案。

### Day 5：开始投递

任务：

1. 更新简历。
2. 更新 GitHub README。
3. 选择 10-20 个岗位投递。
4. 岗位关键词：
   - AI Infra Engineer
   - LLM Inference Engineer
   - ML Systems Engineer
   - Agent Infrastructure Engineer
   - 大模型推理部署
   - 大模型基础设施

产出：

1. `docs/job_search/target_jobs.md`
2. `docs/job_search/application_log.md`

验收：

1. 至少投递第一批岗位。

### Day 6：博客发布与反馈

任务：

1. 发布博客到一个平台：知乎、掘金、公众号、Medium 任选。
2. 发给 2-3 个同行或朋友要反馈。
3. 根据反馈修正 README 和简历说法。

产出：

1. 博客链接记录在 `README.md`
2. `docs/weekly/week11.md`

验收：

1. 项目有公开展示入口。

### Day 7：缓冲

任务：

1. 根据投递反馈补知识。
2. 准备系统设计题：
   - 设计一个 1000 QPS 的 Agent 推理服务。

## 12. Week 12：Paper-Style Report + 求职材料定稿

目标：

1. 把技术报告整理成论文雏形。
2. 判断是否投稿。
3. 求职材料定稿。

### Day 1：Paper-Style Report 框架

任务：

论文标题候选：

```text
AgentInferBench: Characterizing and Optimizing LLM Serving for Tool-Using Agents
```

结构：

1. Introduction
2. Background
3. Benchmark Design
4. Workload Characterization
5. Lightweight Cache-Aware Optimization
6. Evaluation
7. Discussion
8. Related Work
9. Conclusion

产出：

1. `docs/paper/agent_infer_bench_draft.md`

验收：

1. 技术报告被压缩成论文结构。

### Day 2：Related Work 与差异化

任务：

整理相关工作：

1. vLLM / PagedAttention。
2. SGLang / RadixAttention。
3. LMCache。
4. KVFlow / Continuum。
5. ToolSpec / speculative decoding。
6. DistServe / PD disaggregation。

每个相关工作写：

1. 它做了什么。
2. 本项目与它的区别。
3. 本项目是否互补。

产出：

1. `docs/paper/related_work.md`

验收：

1. 不夸大 novelty。
2. 能清楚说明本项目是 benchmark + characterization + lightweight optimization。

### Day 3：论文图表整理

任务：

最终图表：

1. System overview。
2. Workload design。
3. Token breakdown。
4. Prefix overlap by turn。
5. vLLM vs SGLang baseline。
6. Ablation study。
7. Routing tradeoff。
8. Failure cases。

产出：

1. `docs/paper/figures/`
2. `docs/paper/tables/`

验收：

1. 每张图都有 caption。
2. 每张图都能追溯到实验目录。

### Day 4：投稿判断

任务：

按以下标准打分：

1. 数据是否稳定。
2. 优化是否有清晰收益。
3. 是否有足够 baseline。
4. 是否有失败 case。
5. 是否有明确 novelty。
6. 是否有合适 venue。

决策：

1. 如果分数高：准备 workshop / CCF-C / arXiv。
2. 如果分数一般：先发 arXiv 技术报告或博客，不急投正式会议。
3. 如果分数低：继续作为求职项目，下一阶段补实验。

产出：

1. `docs/paper/submission_decision.md`

验收：

1. 不为了投稿牺牲求职节奏。

### Day 5：简历与投递材料定稿

任务：

1. 修订简历。
2. 修订项目 README。
3. 准备 2 分钟项目 pitch。
4. 准备 10 分钟项目深挖讲稿。
5. 准备系统设计回答：
   - Gateway
   - Scheduler
   - vLLM/SGLang serving pool
   - cache-aware routing
   - observability
   - autoscaling
   - failure handling

产出：

1. `docs/interview/project_pitch.md`
2. `docs/interview/system_design_agent_serving.md`

验收：

1. 面试时能从业务场景讲到系统指标，再讲到工程实现。

### Day 6：补投与复盘

任务：

1. 第二批投递。
2. 复盘第一批岗位反馈。
3. 标记知识短板：
   - CUDA/Triton
   - 分布式 serving
   - vLLM internals
   - Kubernetes / deployment
   - observability

产出：

1. `docs/job_search/week12_review.md`
2. `docs/weekly/week12.md`

### Day 7：后续路线规划

任务：

制定 4 周后续计划：

1. 如果面试多：优先补面试知识。
2. 如果项目反馈好：补 vLLM/SGLang PR。
3. 如果论文可投：完善 paper。
4. 如果数据不够强：补更真实的 Agent traces。

产出：

1. `docs/reports/post_week12_plan.md`

## 13. 每周固定复盘模板

每周在 `docs/weekly/weekXX.md` 写：

```markdown
# Week XX

## 本周目标

## 完成情况

## 核心实验

| 实验 | 状态 | 关键结论 | 数据路径 |
|---|---|---|---|

## 关键图表

## 当前最可信的结论

## 当前不可信或需要重跑的数据

## 遇到的问题

## 下周计划

## 简历/论文可用素材
```

## 14. 面试准备每日 20 分钟

Week 5：

1. KV cache。
2. PagedAttention。
3. continuous batching。

Week 6：

1. prefill vs decode。
2. TTFT / TPOT / JCT。
3. vLLM vs SGLang 基础差异。

Week 7：

1. RadixAttention。
2. prefix caching。
3. cache eviction 与 cache locality。

Week 8：

1. Agent workload 特征。
2. benchmark 公平性。
3. prompt canonicalization 的风险。

Week 9：

1. routing tradeoff。
2. sticky session。
3. load balance。

Week 10：

1. 系统设计：1000 QPS Agent serving。
2. observability。
3. autoscaling。

Week 11：

1. 项目 pitch。
2. 失败 case。
3. 与相关工作的区别。

Week 12：

1. 模拟面试。
2. 简历深挖。
3. 下一阶段路线。

## 15. 关键原则

1. 两个月内，主线只有 AgentInferBench。
2. 新论文只读到能定位相关工作，不追复现。
3. 所有数字必须来自可追溯实验。
4. 没有稳定收益也可以写，但必须解释原因。
5. 项目价值优先级：可复现 > 数据可信 > 方法新颖 > 图表好看。
6. 求职优先级高于论文；论文是副产物，不是主线。
