# 推理加速学习计划 V3（零基础 / 本地无 GPU / 远程服务器实验）
更新时间：2026-03-12

## 0. 计划定位与约束
### 现实条件
1. 本地无可用 GPU。
2. 可以在本地写代码，后续组服务器跑实验。
3. 对推理加速领域是从 0 开始。

### 计划目标
1. 24 周完成从基础认知到工程实战的闭环。
2. 在"本地开发 + 远程 GPU 实验"模式下持续产出。
3. 最终沉淀 **3 个可放简历和 GitHub 的项目**：
   - 🥇 **项目 A**（主力）：Agent 场景 LLM 推理优化系统
   - 🥈 **项目 B**（辅助）：高性能 Fused Attention Kernel（Triton）
   - 🥉 **项目 C**（辅助）：LLM 推理回归测试与自动分析平台

## 1. 先定规则：高效工作流（必须先建立）

### 1.1 分工原则（避免浪费时间）
1. 本地机做：读源码、写代码、CPU 单测、文档整理。
2. 远程 GPU 机做：性能测试、profiling、量化、长时实验。
3. 不在本地做“重跑大模型”尝试，防止卡在环境和性能上。

### 1.2 统一目录结构（从第 0 周开始）
建议在仓库中统一使用如下结构：
```text
ai_infra/
  docs/
    weekly/          # 周报
    reports/         # 实验报告 & 技术博客
  notes/             # 学习笔记
  scripts/
    setup/           # 环境搭建脚本
    benchmark/       # 性能测试脚本
    profile/         # profiling 脚本
  experiments/
    configs/         # 实验配置 YAML
    runs/            # 实验结果（按日期）
  projects/
    agent_optim/     # 项目 A：Agent 推理优化
    fused_attention/ # 项目 B：Fused Attention Kernel
    regression_tool/ # 项目 C：回归测试平台
  src/
  tests/
```

### 1.3 统一实验流程（每次实验都这样做）
1. 新建配置：`experiments/configs/<exp_name>.yaml`
2. 运行实验：`scripts/benchmark/run_<exp_name>.sh`（远程）
3. 产出结果：`experiments/runs/<date>/<exp_name>/`
4. 自动汇总：`docs/reports/<week>_<topic>.md`
5. 结论复盘：记录“结论 / 失败点 / 下一步”

### 1.4 统一指标定义（避免数据不可比）
每次 benchmark 必须固定：
1. warmup 次数（例如 20）
2. 正式重复次数（例如 100）
3. 统计指标：P50/P95 latency、吞吐（tokens/s）、显存峰值（GB）
4. 输入范围：batch size、prompt length、generate length
5. 软件版本：CUDA、驱动、PyTorch、vLLM、TensorRT

### 1.5 Git 协作节奏（个人学习也要工程化）
1. 每周一个分支：`weekXX/<topic>`
2. 每天结束前至少一次 commit（附实验结论）
3. 每周末合并到 `main` 并打 tag：`wXX-done`

### 1.6 源码阅读优先级（避免漫无目的）
**P0（必读，面试会直接问）：**
- vllm/core/scheduler.py
- vllm/attention/backends/flash_attn.py
- vllm/spec_decode/
**P1（重要，帮助理解系统设计）：**
- vllm/core/block_manager_v2.py
- vllm/engine/async_llm_engine.py
- sglang/srt/managers/schedule_batch.py
**P2（有余力再看）：**
- vllm/worker/
- vllm/executor/

## 2. 环境和硬件策略

### 2.1 本地环境（Windows）
目标：能开发、能测小样例、能和远程无缝切换。
1. Python 3.11
2. Git + SSH Key
3. VS Code + Remote SSH
4. WSL2（建议，用于接近 Linux 开发体验）

### 2.2 远程环境（Linux GPU）
推荐优先级：
1. 单卡 24GB 起步（4090/3090 均可）
2. 128GB 内存优先于多卡数量（初期更实用）
3. 2TB NVMe（模型和日志会很占空间）

### 2.3 容器化要求
无论本地还是远程，都尽量使用同一套 Docker 环境（至少远程要用）。
1. 固定基础镜像和 CUDA 版本
2. 固定依赖版本（requirements lock）
3. 任何可复现实验都以脚本 + 配置形式保存

## 3. 学习节奏（24 周）
**总览时间线**
阶段 A（Week 0-6）  ：补基础 + 建立可复现实验习惯
阶段 B（Week 7-12） ：远程服务器接入 + 第一轮真实实验 + 项目素材积累
阶段 C（Week 13-18）：系统化优化 + 项目 A/B 启动
阶段 D（Week 19-24）：项目冲刺 + 开源化 + 面试准备
**项目时间线**
──────────────────────────────────────────────────────
Week:  0  2  4  6  8  10  12  14  16  18  20  22  24
       │           │   │              │    │         │
       ▼           ▼   ▼              ▼    ▼         ▼
    基础构建    服务器  素材积累     项目启动  冲刺    交付
                       ├ B baseline  ├ A Phase1      ├ 开源化
                       ├ 量化数据    ├ B Phase2      ├ 博客
                       └ profiling   └ A Phase2      └ 简历
──────────────────────────────────────────────────────

## 阶段 A（Week 0-6）：补基础 + 建立可复现实验习惯

### Week 0：初始化与benchmark
本周目标：
1. 建立可复现的开发与实验脚手架。
2. 写出第一版 benchmark 模板。

每日任务：
1. Day 1：整理仓库目录结构，创建 `docs/weekly` 和 `experiments` 目录。
2. Day 2：整理本地开发环境（Python、依赖、测试框架）。
3. Day 3：写一个最小 benchmark 脚本（CPU 模型也可以）。
4. Day 4：定义统一指标字段（P50/P95/throughput/memory）。
5. Day 5：写《实验规范 v1》文档（输入、输出、命名规则）。
6. Day 6-7：复盘并补齐遗漏项。

本周产出：
1. `docs/weekly/week00.md`
2. `docs/reports/benchmark_template.md`
3. 一个可执行的 benchmark 脚本雏形

验收标准：
1. 能一键运行一个小实验并保存结构化结果。
2. 别人在你的仓库中能看懂如何复现实验。

### Week 1：推理基础 + PyTorch 推理链路
本周目标：
1. 搞清推理路径（前处理 -> 模型 -> 后处理 -> 指标统计）。
2. 了解 latency 和 throughput 的核心差异。

任务：
1. 用小模型（ResNet50/BERT-base）跑 PyTorch 推理。
2. 对比 batch=1 与 batch>1 的延迟和吞吐变化。
3. 写出推理耗时拆分（tokenization、模型前向、后处理）。

本周产出：
1. 推理流程图
2. 指标对比表（至少 3 组 batch）
3. `docs/weekly/week01.md`

验收标准：
1. 你能口头解释“为什么 batch 提高时吞吐提升但单请求延迟可能变差”。

### Week 2：ONNX + torch.compile入门
**PartA:ONNX 与 ONNX Runtime**
本周目标：
1. 跑通 PyTorch -> ONNX -> ONNX Runtime。
2. 学会使用 Netron 分析模型图。

任务：
1. 导出 ResNet50/BERT 到 ONNX。（动态batch版本）
2. 使用 ONNX Runtime（CPU）跑推理并对比 PyTorch。
3. 记录算子兼容性和图优化前后差异。

本周产出：
1. ONNX 导出脚本
2. ORT vs PyTorch 性能表
3. ONNX 图结构笔记

本周搞懂：
1. 能解释 ONNX 的价值与局限（跨框架、部署友好、但并非总是更快）。
2. `opset`是导出时算子版本，会影响兼容性
3. 静态shape和动态shape：什么时候用固定batch，什么时候用dynamic axes
4. ONNX Runtime执行机制：`Execution Provider`、图优化等级、session配置(线程数、优化等级等)
5. 正确性校验：Pytorch输出和ORT输出如何做数值对齐（同一输入喂pytorch和ort，比较输出张量的误差（如max_abs_err，allclose））
6. Netron看图：输入输出、节点、算子类型、是否有冗余节点、动态维是否如配置一样生效

**PartB:torch.compile入门**
目标：
1. 理解 PyTorch 2.x 的编译路径：Dynamo trace → Inductor codegen → Triton kernel。
2. 在小模型上对比 eager vs torch.compile 的性能差异。

任务：
1. 用 `torch.compile` 编译 ResNet50/BERT，对比 eager mode 推理速度。
2. 使用 `TORCH_LOGS="graph_code"` 查看生成的图。
3. 了解 `mode="reduce-overhead"` / `mode="max-autotune"` 的区别。

产出：
1. ONNX 导出脚本 + ORT vs PyTorch 性能表
2. torch.compile 性能对比表
3. 笔记：《ONNX vs torch.compile：两条推理优化路径的对比》

验收：
1. 能解释 torch.compile 的编译链路和适用场景。
2. torch.compile(..., mode=...) 的模式差异（default / reduce-overhead / max-autotune）
3. graph break 是什么，为什么会让加速效果变差
4. 输入 shape 变化为什么会触发重新编译
5. 如何用 TORCH_LOGS 定位问题

### Week 3：Attention 与 FlashAttention 基础
本周目标：
1. 用 PyTorch 小实验理解 attention 的内存瓶颈。
2. 理解 FlashAttention 的核心思想（减少 HBM 往返）。

任务：
1. 写 naive attention基线实现（可用小尺寸张量）。
2. 用不同 sequence length 记录时延和显存峰值。
3. 对比Pytorch SDPA后端（math/flash/mem-efficient）可用哪个用哪个
4. 产出FlashAttention/FlashAttention-2/3对比笔记
5. 产出两张图：latency vs seq_len，peak memory vs seq_len

理解核心内容：
1. 为什么attention在长序列上变慢且更吃显存
2. 什么是IO瓶颈
3.   FlashAttention v1
      └── 核心贡献：Tiling + Recomputation，减少 HBM 读写
    FlashAttention v2
      └── 核心贡献：更好的并行策略，减少非矩阵乘法计算占比
    FlashAttention v3（补充阅读）
      └── 核心贡献：专为 Hopper 架构（H100）设计
          - 利用 Tensor Core 异步执行（wgmma 指令）
          - Softmax 与 GEMM 流水线重叠
          - 实测：H100 上比 FA2 快约 1.5-2x

本周产出：
1. attention 对比曲线图
2. 《FlashAttention 改进点总结，三个版本对比》

验收标准：
1. 能解释“tiled 计算为什么可能更快”。

### Week 4：vLLM 架构入门（源码阅读）
本周目标：
1. 建立对 vLLM 三件事的理解：PagedAttention、调度器、KV Cache（先了解scheduler + block_manager）
2. 理解 continuous batching 的核心机制。

任务：
1. 阅读 `vllm/attention/backends/*`
2. 阅读 `vllm/core/scheduler.py`
3. 阅读 `vllm/core/block_manager*.py`
4. 画出 request 生命周期图（进入队列 -> prefill -> decode -> 结束）
5. 理解 continuous batching 与 static batching 的核心差异
  - Iteration-level scheduling vs request-level scheduling
  - Prefill-decode 解耦：chunked prefill / splitfuse 的思想
  - 为什么 continuous batching 对吞吐提升巨大

本周产出：
1. 3 张图：KV 分页图、调度流程图、请求生命周期图
2. continuous batching原理笔记
3. `docs/weekly/week04.md`

验收标准：
1. 能解释“为什么 KV cache 不用连续内存，KV cache如何分页”。
2. 能解释"continuous batching 为什么比 static batching 吞吐更高"。
3. 能说清 chunked prefill 解决了什么问题。

### Week 5：性能测试与Profiling基础，推理框架认知
本周目标：
1. PartA: 了解latency拆分，queue vs compute，torch profiler，指标口径统一。
2. PartB:
    - TensorRT: 了解layer fusion / tactic / precision
    - SGLang: 了解RadixAttention / continuous batching
    - 对比表：vLLM vs SGLang vs TRT-LLM 的适用场景

任务：
1. 阅读 TensorRT 官方文档关键章节（网络定义、builder、优化过程）。
2. 写总结：layer fusion、kernel autotune、FP16/INT8 的作用。
3. 如果暂时无 GPU，本周只做“流程理解 + 伪代码演练”。

本周产出：
1. 测量规范v2 + 一份瓶颈定位报告
2. 《TensorRT 工作机制笔记》
3. Engine 构建流程图
4. 不追求全部搞懂，建立engine/tactic/precision认知，产出流程图+失败案例清单模板
5. 了解框架的不同：
    | 框架 | 适用场景 |	核心优势 | 局限 |
    |:---:|:---:|:---:|:---:|
    | vLLM | 通用 LLM serving | 生态最好、最易用 |	定制化难 |
    | SGLang |	Agent / 复杂 prompt |	RadixAttention、structured output | 相对较新 |
    | TRT-LLM |	NVIDIA 生产环境 |	极致性能 | 部署复杂、迭代慢 |
    | ONNX Runtime | 中小模型、边缘 |	跨平台 | 大模型支持弱 |

验收标准：
1. 能回答“TensorRT 为什么通常比 eager PyTorch 快”。

### Week 6：量化基础（先理论后实践）
本周目标：
1. 系统掌握 INT8 量化基础术语和流程。
2. 完成 PTQ 小模型试验。
3. 理解 FP8 的优势与适用场景。

任务：
1. 梳理 scale、zero point、per-tensor、per-channel。
2. 量化全谱系理论
    FP32  → FP16  → BF16  → FP8  → INT8  → INT4
    高精度                                   低精度
    慢/大                                    快/小
3. 做一个小模型 PTQ 实验（可先不做 7B）。
4. FP8专题
    FP8 有两种格式：
      E4M3：更大动态范围，适合权重
      E5M2：更大精度范围，适合梯度（训练用）
      推理时权重和激活值通常都用 E4M3

    ```
    为什么 FP8 比 INT8 更受欢迎（2025年之后）？

    INT8 量化：
      - 需要 calibration（校准数据集）
      - 存在量化误差，敏感层需要跳过
      - 工具链复杂（需要 QAT 或 PTQ 流程）

    FP8 量化：
      - H100/H200 硬件原生支持（Transformer Engine）
      - 精度损失极小（接近 FP16）
      - 不需要 calibration，直接 cast
      - vLLM / TRT-LLM 已原生集成
      
      结论：有 H100 就用 FP8，没有 H100 用 AWQ INT4
    ```

5. 明确 AWQ/GPTQ 的适用场景差异，量化选型决策表。
    | 硬件 | 精度要求 |	速度要求 | 推荐方案 |
    |:---:|:---:|:---:|:---:|
    | H100/H200 | 高 | 高 |	FP8|
    | A100 / A10 |	高 |	中 | BF16/INT8 |
    | A100 / A10 |	中 |	高 | AWQ INT4 |
    | 3090 / 4090 | 中 |	高 | AWQ INT4 / GPTQ |

本周产出：
1. 量化术语卡片（简明定义）
2. PTQ 小实验报告

验收标准：
1. 能解释“为什么量化可能变快，但准确率会下降”。

---

## 阶段 B（Week 7-12）：远程服务器接入 + 第一轮真实实验 + 项目素材积累

### Week 7：远程服务器搭建与验收
本周目标：
1. 完成远程开发链路。
2. 能在远程稳定跑基础 GPU 脚本。

任务：
1. 配置 SSH、密钥登录、基础安全策略。
2. 安装 CUDA、驱动、Docker（或使用预装镜像）。
3. 运行 GPU 自检脚本（显卡、显存、CUDA 可用性）。
4. 跑第一个 PyTorch GPU 推理样例。

本周产出：
1. 《远程服务器环境记录》文档
2. `scripts/setup/check_gpu_env.sh`

验收标准：
1. 本地可一键连接远程并启动实验脚本。

### Week 8：vLLM 基准实验（第一版）
本周目标：
1. 跑通 vLLM 推理并形成 baseline。
2. 初步了解 Speculative Decoding 概念。

任务：
1. 选择 7B 级模型（优先 Qwen 系列或同量级）。
2. 记录不同 batch/prompt length 下的吞吐与延迟，增加shared prefix ratio实验维度（shared_prefix_lengths: [0, 512, 1024, 2048]  # 模拟 system prompt / RAG context）
3. 分析 prefill 与 decode 的性能占比。
4. 阅读 Speculative Decoding 原始论文（Fast Inference from Transformers via Speculative Decoding），理解 draft-verify 流程。

项目素材积累：
- 项目A：这份 baseline 数据将作为 Agent 优化前的对比基准
- 项目C：这是回归测试平台的第一批历史数据

本周产出：
1. baseline 表格（至少 12 组组合）
2. `docs/reports/week08_vllm_baseline.md`
3. 产出中增加rag/agent场景典型负载分布
4. Speculative Decoding 原理笔记（画出 draft-verify 时序图）

验收标准：
1. 能回答当前瓶颈更偏向算力、显存还是调度。
2. 写技术博客，《vllm深度解析，如何验证2x性能提升》

### Week 9：Speculative Decoding 专项
本周目标：
1. 理解 speculative decoding 的数学原理（rejection sampling）。
2. 在 vLLM 中跑通 spec decode 实验。

任务：
1. 深入理解 draft-verify 流程的数学保证（rejection sampling 保证输出分布不变）。
2. 阅读 vLLM spec_decode/ 模块源码（P0）。
3. 实验：对比 7B 模型在有/无 spec decode 下的 throughput。
4. 了解主要变体：
  - Medusa：多头并行预测，无需 draft model
  - Eagle：feature-level draft，利用隐藏层特征
  - Lookahead Decoding：利用 Jacobi 迭代
5. 分析什么场景下 spec decode 收益最大（输出可预测性高 → acceptance rate 高）。

项目素材积累：
- 项目 A：Agent 输出多为 JSON/function call，格式可预测，spec decode 天然适配

本周产出：
1. Speculative Decoding 原理图 + 数学推导笔记
2. vLLM spec decode 实验报告（不同 draft model 的 acceptance rate 对比）
3. 《什么场景下 speculative decoding 收益最大》分析文档

验收标准：
1. 能解释"为什么 acceptance rate 决定了 spec decode 的加速比"。
2. 能说清 draft model 的选择策略。
3. 能画出 draft-verify 的完整时序图。

### Week 10：量化实战（AWQ/GPTQ）+多后端对比
本周目标：
1. 在远程 GPU 上完成一次真实量化实验。
2. 完成多后端性能对比。

任务：
1. 选择一个 7B 模型（llama2-7B）跑 AWQ/GPTQ（至少一种完整跑通）。
2. 输出精度损失和性能收益
3. 关注多轮对话下的精度累计误差，长context下量化模型的表现等
4. 形成“量化决策表”（场景 -> 推荐方案）。
5. 多后端对比：
  - 小模型（BERT/ResNet）：PyTorch vs ORT vs TRT
  - 大模型（7B）：PyTorch vs vLLM vs SGLang

项目素材积累：
- 项目 A：Agent 场景（function call）下量化对精度的影响数据
- 项目 C：多框架 benchmark 数据，直接复用为回归平台的数据源

本周产出：
1. 量化前后 benchmark 报告
2. 量化配置文件和脚本
3. 输出量化决策表拓展（如单轮问答、agent多轮、function call等情况下采用什么量化方案）
4. 多后端对比表格与结论文档
5. 技术报告：《大模型推理性能优化 Checklist》

验收标准：
1. 能说清“当前业务下是否值得量化”。

### Week 11：CUDA 基础 Kernel（从 0 到 1）
本周目标：
1. gpu架构认知，SM，warp,shard memory, occupancy, tensor core
2. 真正开始写 CUDA kernel（可先小尺寸）。

任务：
1. 画出gpu结构图，写一篇gpu执行模型笔记 ，可以用nsys分析瓶颈
2. 手写 `vector_add`、`reduce_sum`、`softmax`。
3. 对比 naive 实现与优化实现。
4. 用 Nsight Systems 进行初步分析。

项目素材积累：
- 项目B：Fused Attention Kernel前置能力

本周产出：
1. GPU 架构笔记 + 结构图
2. 三个 kernel 代码 + 每个 kernel 的性能对比表

验收标准：
1. 能解释线程组织和内存访问模式。

### Week 12：Matmul 优化第一轮
本周目标：
1. 从 naive matmul 提升到 shared memory 版本。

任务：
1. 实现 naive matmul。
2. 实现 1D/2D tiling。
3. 跑不同 tile size 并分析性能变化。
4. nsys profile初步分析，看SM occupancy为什么低，找到优化点

项目素材积累：
- 项目B：tiling 思想直接应用于 attention kernel

本周产出：
1. matmul 版本演进记录
2. Nsight 分析截图和结论

验收标准：
1. 相比 naive 至少取得明显性能提升（目标建议 >=3x）。

---

## 阶段 C（Week 13-18）：系统化优化 + 项目 A/B 正式启动

### Week 13：Matmul 高级优化 + 分布式推理基础
**PartA:Matmul高级优化**
目标：
1. 继续做 double buffering、bank conflict 优化，达到10x naive性能。
2. 对比 cuBLAS，定位差距来源。

产出：
1. 完整的nsys report
2. roofline分析
3. 产出技术博客《cuda matmul优化实战：从1x到10x的旅程》

**PartB:分布式推理基础**
目标：
1. 理解 Tensor Parallelism vs Pipeline Parallelism 的取舍。
2. 理解 NCCL 通信开销。
3. 了解 70B+ 模型的部署策略（TP=4 / TP=8 的典型配置）。

任务：
1. 阅读 vLLM 分布式相关代码：vllm/executor/、vllm/worker/。
2. 画出 TP 和 PP 的数据流图。
3. 理解 all-reduce、all-gather 等通信原语。
4. 笔记：不同模型规模 → 推荐并行策略。

产出：
1. 《分布式推理基础：TP vs PP》笔记
2. 通信开销估算表（不同模型规模 × 不同并行度）

验收：
1. 能解释"为什么 70B 模型用 TP=4 而不是 PP=4"。

### Week 14：Attention Kernel 简化版 → 项目 B Phase 1 启动 🔥
目标：
1. 用 Triton 从零实现 tiled attention（项目 B 正式启动）。
2. 阅读源码，理解 FlashAttention 的工程实现思路。

任务(项目B Phase1)：
1. 用 Triton 实现 naive attention（for loop over seq_len）。
2. 实现 tiled version（block-wise softmax with online correction）。
3. 验证正确性（vs PyTorch reference，atol=1e-3）。
4. 性能对比：naive vs tiled vs torch.nn.functional.scaled_dot_product_attention。
5. 阅读 FlashAttention 源码，理解 online softmax 的数学推导。

本周产出：
1. projects/fused_attention/ 初始代码
2. naive → tiled 的性能对比表
3. 正确性校验报告
4. profiling 数据

验收：
1. 代码正确 + 有 profiling 数据。
2. 能画出 attention tiling 的示意图并解释 online softmax。

### Week 15：Agent Infra 推理特征分析 → 项目 A Phase 1 启动 🔥
目标：
1. 分析 Agent 场景下的推理负载特征。
2. 实现一个 Agent 推理 benchmark。

任务（项目A Phase1）：
1. 搭建一个最简 Agent（ReAct 模式：think → tool_call → observe → think...）。
2. 使用 Qwen-7B（function calling 版本）作为基座模型。
3. 分析 Agent 单次任务的推理调用模式：
  - 多轮调用次数分布
  - 每轮 input/output token 数分布
  - Prefix 重叠率（system prompt + tool description + history）
4. 测试 vLLM prefix caching 在 Agent 场景的命中率。
5. 测试 structured output（JSON mode）对 decode 速度的影响。
6. 分析 function calling 的 token overhead。

本周产出：
1. projects/agent_optim/ 初始代码
2. Agent 推理负载特征报告（含数据图表）
3. Prefix cache 命中率 vs Agent 轮次关系图
4. 《Agent Infra 推理优化机会清单》

验收：
1. 能回答"Agent 场景和普通 chat 场景的推理优化侧重点有什么不同"。
2. 能给出至少 3 个 Agent 场景特有的优化方向。

### Week 16：Serving 综合
目标：
1. 跑通一个动态 batching 服务（Triton Server 或 Ray Serve，二选一深入）。
2. 理解 ensemble 模型调用。
3. 测试并发请求下的延迟变化。

任务：
1. 部署一个动态 batching 推理服务。
2. 测试并发请求处理与参数调优。
3. 跑通多副本部署和基础路由策略。
4. 设计一个支持 1000 QPS 的推理服务架构。

本周产出：
1. 并发测试结果与参数调优记录
2. 《流量突增应对方案》+ 多模型共享 GPU 调度策略
3. 完整的数据流图（从 HTTP 请求到 GPU 计算）

验收：
1. 能给出一个完整的推理服务架构设计并解释每个组件的作用。

### Week 17：Triton（OpenAI）算子实现 + 项目 B Phase 2
**Part A：常用算子实现**
目标：
1. 用 Triton 实现 RMSNorm（LLaMA/Qwen 都在用）。
2. 用 Triton 实现 SiLU × Linear 融合（FFN 层常见 fusion）。
**Part B：项目 B GQA 支持**
目标：
1. 扩展 Week 14 的 attention kernel，支持 Grouped Query Attention。
2. 处理 num_kv_heads != num_q_heads 的分组逻辑。

产出：
1. 两个 Triton 常用算子（含单测）
2. GQA attention kernel 代码 + 不同 group size 性能对比
3. 性能对比表（Triton vs PyTorch vs torch.compile）
4. 笔记：《为什么 Triton 比手写 CUDA 更适合快速迭代》

### Week 18：Agent场景端到端优化 → 项目 A Phase 2 🔥
目标：
1. 针对 Week 15 发现的优化机会，实现具体优化。
2. 形成端到端优化报告。
优化目标模型：Qwen-7B（function calling 版本）

优化维度：  
优化 1：Prefix Cache 策略优化
├── 分析 vLLM 默认 prefix cache 在 Agent 场景的命中率（Week 15 已有数据）
├── 实现分层缓存：system_prompt(L1) + tool_desc(L2) + history(L3)
└── 目标：cache hit rate 从 ~40% 提升到 ~80%

优化 2：Speculative Decoding 适配
├── Agent 输出多为 JSON/function call，格式可预测
├── 测试不同 draft model 的 acceptance rate
└── 目标：decode throughput 提升 1.5-2x

优化 3：多轮 KV Cache 管理
├── 跨轮 KV cache 复用（而非每轮重新 prefill 全部历史）
├── 实现 incremental prefill
└── 目标：第 N 轮的 TTFT 比第 1 轮降低 50%+

优化 4：Structured Output 加速
├── Constrained decoding（JSON schema guided）
├── 减少无效 token 生成
└── 测量精度与速度 tradeoff

优化 5：量化
├── FP8/AWQ 在 tool-calling 精度上的影响
└── 参考 Week 10 数据

本周产出：
1. 优化后的端到端对比报告
2. 每个优化方向的独立实验记录

验收：
1. first token latency < 100ms，throughput > 50 tokens/s/user（硬件允许时）
2. 单次 Agent 任务（5 轮交互）总延迟下降 >=30%
3. 产出《Agent 推理优化实战报告 v1》

---

## 阶段 D（Week 19-24）：项目冲刺 + 开源化 + 面试准备

### Week 19：项目设计文档定稿 + 项目 B Phase 3
**PartA:：三个项目设计文档**
为每个项目写正式设计文档：背景、目标、方案、风险、里程碑。
**Part B：项目 B Phase 3 — Decode-phase Attention 优化**
目标：
1. Decode 阶段 query 只有 1 个 token，但 KV 可能很长（memory-bound）。
2. 实现 split-K attention（将 KV 分段并行计算）。
3. 对比不同 split factor 的性能。
4. nsight profiling：证明优化前是 memory-bound。

产出：
1. split-K attention 代码 + 性能对比表
2. 项目 B 完整 README
3. nsight roofline 分析报告

### Week 20-21：主项目开发冲刺
要求：
1. 每周至少 5 次可运行提交
2. 每周末一次端到端演示

重点：
week20:
1. 将 Week 18 的各优化方向整合为统一系统。
2. 设计 Agent Benchmark Suite：
  - 场景 1：单工具调用（简单 QA）
  - 场景 2：多工具串行（数据查询 → 计算 → 格式化）
  - 场景 3：多工具并行（同时查天气 + 查日历）
  - 场景 4：长对话 Agent（10+ 轮交互）
3. 完善 prefix cache 分层策略的边界处理。
week21:
1. 完整 benchmark 所有优化组合。
2. 写出完整的优化前后对比报告（含所有场景）。
3. 尝试向 vLLM 社区提交 PR（即使是小改进）。


产出：
1. 项目 A 可运行 demo + 完整文档
2. Agent Benchmark Suite 代码
3. 完整的优化报告

### Week 22：项目 C 快速搭建 + 性能与稳定性打磨
**Part A：项目 C — 回归测试平台**
核心功能：
1. 多框架统一 Benchmark Runner
   ├── 支持 vLLM / SGLang / PyTorch eager
   ├── 统一配置格式（YAML）
   ├── 统一指标采集（TTFT, TPOT, throughput, memory）
   └── 支持多种负载模式（fixed prompt / ShareGPT / Agent trace）

2. 历史数据管理
   ├── SQLite 存储每次 benchmark 结果
   ├── 版本号 + 硬件信息 + 配置 hash → 唯一标识
   └── 支持跨版本对比查询

3. 回归检测
   ├── 自动对比当前版本 vs 上一版本
   ├── 超过阈值（如 latency +10%）自动标红
   └── 生成回归报告

4. 可视化报告
   ├── HTML 报告（Plotly 交互式图表）
   ├── 版本趋势图（latency / throughput 随版本变化）
   └── 场景热力图（batch_size × seq_len → latency）

**Part B：项目 A/B 稳定性打磨**
任务：
1. 建立回归测试（项目 A 的所有优化都要有自动化验证）。
2. 补齐 profiling 和错误处理。

产出：
1. 项目 C 可运行版本 + README
2. 项目 A/B 的性能回归报告

### Week 23：开源化与表达
任务：
1. 整理三个项目的 README、使用教程、架构图。
2. 写技术博客
  - 《Agent 推理优化实战：从负载分析到 30% 延迟降低》
  - 《Triton Attention Kernel：从 Naive 到 Split-K 的优化之路》
  - 《构建 LLM 推理回归测试平台》（可选）
3. 将项目发布到 GitHub 并整理展示页。

产出：
1. 三个项目的 GitHub 仓库（或单一 monorepo 的子目录）
2. 2-3 篇技术博客
3. 项目展示页

### Week 24：简历化与面试准备
任务：
1. 把项目改写为 STAR 结构成果描述
2. 准备 25 个高频问答
3. 模拟面试练习

---
#### 简历项目描述
```
项目 A：Agent 场景 LLM 推理优化系统
- 分析 Agent 多轮推理负载特征，发现 prefix 重叠率 >70% 但缓存命中率仅 40%
- 设计分层 Prefix Cache 策略，将 Agent 场景缓存命中率提升至 82%
- 适配 Speculative Decoding 到结构化输出场景，decode 吞吐提升 1.8x
- 实现跨轮 KV Cache 复用，第 5 轮 TTFT 从 320ms 降至 145ms
- 设计 Agent 推理 Benchmark Suite，覆盖 4 类典型 Agent 任务模式
技术栈：Python, PyTorch, vLLM, CUDA, Triton, nsight systems

项目 B：高性能 Fused Attention Kernel（Triton）
- 使用 Triton 从零实现 tiled attention，支持 online softmax 校正
- 扩展支持 GQA（Grouped Query Attention），适配 Llama/Qwen 架构
- 针对 decode 阶段（memory-bound）实现 split-K 优化，延迟降低 35%
- 全部 kernel 通过数值正确性校验（vs PyTorch SDPA，atol=1e-3）
- 使用 nsight systems 完成 roofline 分析，定位每个版本的瓶颈
技术栈：Triton, CUDA, PyTorch, nsight systems

项目 C：LLM 推理性能回归测试平台
- 设计多框架统一 Benchmark 系统，支持 vLLM/SGLang/PyTorch
- 实现自动回归检测，对比跨版本性能（TTFT/TPOT/吞吐量/显存）
- 发现 vLLM 版本升级中 decode 吞吐回归 12% 的问题
- 生成交互式 HTML 报告，包含版本趋势图和场景热力图
技术栈：Python, SQLite, Plotly, vLLM, SGLang, Docker
```
---
#### 面试高频问答清单
| # | 问题 |	对应学习周 |
|:---:|:---:|:---:|
| 1 | KV Cache 是什么，为什么需要 | Week 4 |
| 2 | PagedAttention 原理 | Week 4 |
| 3 | Continuous batching vs static batching | Week 4 |
| 4 | Prefill vs Decode 的区别与优化 | Week 8 |
| 5 | FlashAttention 为什么快 | Week 3 |
| 6 | Speculative Decoding 原理与适用场景 | Week 9 |
| 7 | INT8/FP8/AWQ/GPTQ 区别与选型 | Week 6+10 |
| 8 | Tensor Parallelism vs Pipeline Parallelism | Week 13 |
| 9 | CUDA 线程模型（grid/block/warp） | Week 11 |
| 10 | Shared memory 和 bank conflict | Week 12-13 |
| 11 | torch.compile 的工作原理 | Week 2 |
| 12 | Agent 推理和普通 chat 推理有什么不同 | Week 15 |
| 13 | 如何定位推理瓶颈（compute/memory/IO） | Week 5 |
| 14 | Roofline model 怎么用 | Week 13 |
| 15 | 如何选择推理框架 | Week 5 |
| 16 | NCCL 通信与分布式推理 | Week 13 |
| 17 | 长上下文推理的挑战 | Week 15+18 |
| 18 | Prefix caching 原理与适用场景 | Week 15+18 |
| 19 | 如何做推理服务的负载均衡 | Week 16 |
| 20 | 量化对不同任务精度的影响 | Week 10 |
| 21 | Online softmax 的数学原理 | Week 14 |
| 22 | GQA/MQA 和 MHA 的区别 | Week 14+17 |
| 23 | Chunked prefill 解决什么问题 | Week 4 |
| 24 | Structured output 加速方法 | Week 18 |
| 25 | 你做的项目中最大的技术挑战是什么 | 全程 |

## 4. 每周固定交付模板（强制执行）

每周必须提交以下 4 项：
1. 周报：`docs/weekly/weekXX.md`
2. 数据：`experiments/runs/weekXX/...`
3. 报告：`docs/reports/weekXX_<topic>.md`
4. 复盘：本周做对了什么、做错了什么、下周怎么改

周报模板（建议）：
```markdown
# Week XX

## 本周目标
## 已完成任务
## 核心实验结果（表格）
## 项目进展（A/B/C 哪个有推进）
## 问题与排查
## 本周结论（最多 5 条）
## 下周计划
```

---

## 5. 项目总览
### 项目矩阵
```
┌─────────────────────────────────────────────────────────┐
│  项目 A（主力）：Agent 场景 LLM 推理优化系统              │
│  证明：架构理解 + 性能优化 + Agent Infra 领域能力         │
│  周期：Week 15 启动 → Week 21 完成                       │
├─────────────────────────────────────────────────────────┤
│  项目 B（辅助）：高性能 Fused Attention Kernel            │
│  证明：底层 CUDA/Triton 能力 + profiling 驱动优化         │
│  周期：Week 14 启动 → Week 19 完成                       │
├─────────────────────────────────────────────────────────┤
│  项目 C（辅助）：LLM 推理回归测试平台                     │
│  证明：工程素养 + 自动化 + 对团队的实际价值               │
│  周期：Week 22 快速搭建（复用前 18 周数据）               │
└─────────────────────────────────────────────────────────┘
```
### 项目时间甘特图
```
Week:    7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24
         │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
项目 A:  │   素材积累..............│   │  P1━━│   │  P2━━━━│  冲刺━━━━│   │   │
项目 B:  │   │   │   │  素材积累...│  P1━━│   │  P2━│  P3━━│   │   │   │   │
项目 C:  │   数据积累.........................................│  搭建━│   │   │
         │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │
开源/博客:│   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │  ━━━│
简历面试: │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │   │  ━│

P1 = Phase 1, P2 = Phase 2, P3 = Phase 3
```

