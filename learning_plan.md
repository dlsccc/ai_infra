# 推理加速学习计划 V3（零基础 / 本地无 GPU / 远程服务器实验）
更新时间：2026-03-12

## 0. 计划定位与约束
当前的现实条件：
1. 本地无可用 GPU。
2. 可以在本地写代码，后续组服务器跑实验。
3. 对推理加速领域是从 0 开始。

本计划的目标：
1. 24 周完成从基础认知到工程实战的闭环。
2. 在“本地开发 + 远程 GPU 实验”模式下持续产出。
3. 最终沉淀 2-3 个可放简历和 GitHub 的项目。

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
    weekly/
    reports/
  notes/
  scripts/
    setup/
    benchmark/
    profile/
  experiments/
    configs/
    runs/
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

### Week 2：ONNX 与 ONNX Runtime（本地可做）
本周目标：
1. 跑通 PyTorch -> ONNX -> ONNX Runtime。
2. 学会使用 Netron 分析模型图。

任务：
1. 导出 ResNet50/BERT 到 ONNX。
2. 使用 ONNX Runtime（CPU）跑推理并对比 PyTorch。
3. 记录算子兼容性和图优化前后差异。

本周产出：
1. ONNX 导出脚本
2. ORT vs PyTorch 性能表
3. ONNX 图结构笔记

验收标准：
1. 能解释 ONNX 的价值与局限（跨框架、部署友好、但并非总是更快）。

### Week 3：TensorRT 理论 + ORT TensorRT EP 认知
本周目标：
1. 理解 TensorRT 的核心优化机制。
2. 知道 Engine 构建、tactic、precision 的关系。

任务：
1. 阅读 TensorRT 官方文档关键章节（网络定义、builder、优化过程）。
2. 写总结：layer fusion、kernel autotune、FP16/INT8 的作用。
3. 如果暂时无 GPU，本周只做“流程理解 + 伪代码演练”。

本周产出：
1. 《TensorRT 工作机制笔记》
2. Engine 构建流程图

验收标准：
1. 能回答“TensorRT 为什么通常比 eager PyTorch 快”。

### Week 4：vLLM 架构入门（源码阅读）
本周目标：
1. 建立对 vLLM 三件事的理解：PagedAttention、调度器、KV Cache。

任务：
1. 阅读 `vllm/attention/backends/*`
2. 阅读 `vllm/core/scheduler.py`
3. 阅读 `vllm/core/block_manager*.py`
4. 画出 request 生命周期图（进入队列 -> prefill -> decode -> 结束）

本周产出：
1. 3 张图：KV 分页图、调度流程图、请求生命周期图
2. `docs/weekly/week04.md`

验收标准：
1. 能解释“为什么 KV cache 不用连续内存”。

### Week 5：Attention 与 FlashAttention 基础
本周目标：
1. 用 PyTorch 小实验理解 attention 的内存瓶颈。
2. 理解 FlashAttention 的核心思想（减少 HBM 往返）。

任务：
1. 写 naive attention（可用小尺寸张量）。
2. 用不同 sequence length 对比耗时与显存。
3. 阅读 FlashAttention/FlashAttention-2 核心思想并做笔记。

本周产出：
1. attention 对比曲线图
2. 《FlashAttention 改进点总结》

验收标准：
1. 能解释“tiled 计算为什么可能更快”。

### Week 6：量化基础（先理论后实践）
本周目标：
1. 系统掌握 INT8 量化基础术语和流程。
2. 完成 PTQ 小模型试验。

任务：
1. 梳理 scale、zero point、per-tensor、per-channel。
2. 做一个小模型 PTQ 实验（可先不做 7B）。
3. 明确 AWQ/GPTQ 的适用场景差异。

本周产出：
1. 量化术语卡片（简明定义）
2. PTQ 小实验报告

验收标准：
1. 能解释“为什么量化可能变快，但准确率会下降”。

---

## 阶段 B（Week 7-12）：远程服务器接入 + 第一轮真实实验

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

任务：
1. 选择 7B 级模型（优先 Qwen 系列或同量级）。
2. 记录不同 batch/prompt length 下的吞吐与延迟。
3. 分析 prefill 与 decode 的性能占比。

本周产出：
1. baseline 表格（至少 12 组组合）
2. `docs/reports/week08_vllm_baseline.md`

验收标准：
1. 能回答当前瓶颈更偏向算力、显存还是调度。

### Week 9：ONNX Runtime / TensorRT 对比实验
本周目标：
1. 跑通至少一个模型的多后端对比（PyTorch / ORT / TRT）。

任务：
1. 先选小模型跑通全流程，后尝试 7B 的部分子路径。
2. 对比 FP32/FP16（可用时再尝试 INT8）。
3. 排查失败案例并记录（如算子不支持、转换失败）。

本周产出：
1. 对比表格与结论文档
2. 问题清单（失败原因 + 修复思路）

验收标准：
1. 有可复现结论，不要求“全部成功”。

### Week 10：量化实战（AWQ/GPTQ）
本周目标：
1. 在远程 GPU 上完成一次真实量化实验。

任务：
1. 选择一个 7B 模型跑 AWQ/GPTQ（至少一种完整跑通）。
2. 输出精度损失和性能收益。
3. 形成“量化决策表”（场景 -> 推荐方案）。

本周产出：
1. 量化前后 benchmark 报告
2. 量化配置文件和脚本

验收标准：
1. 能说清“当前业务下是否值得量化”。

### Week 11：CUDA 基础 Kernel（从 0 到 1）
本周目标：
1. 真正开始写 CUDA kernel（可先小尺寸）。

任务：
1. 手写 `vector_add`、`reduce_sum`、`softmax`。
2. 对比 naive 实现与优化实现。
3. 用 Nsight Systems 进行初步分析。

本周产出：
1. 三个 kernel 代码
2. 每个 kernel 的性能对比表

验收标准：
1. 能解释线程组织和内存访问模式。

### Week 12：Matmul 优化第一轮
本周目标：
1. 从 naive matmul 提升到 shared memory 版本。

任务：
1. 实现 naive matmul。
2. 实现 1D/2D tiling。
3. 跑不同 tile size 并分析性能变化。

本周产出：
1. matmul 版本演进记录
2. Nsight 分析截图和结论

验收标准：
1. 相比 naive 至少取得明显性能提升（目标建议 >=3x）。

---

## 阶段 C（Week 13-18）：系统化优化与服务化

### Week 13：Matmul 高级优化
目标：
1. 继续做 double buffering、bank conflict 优化。
2. 对比 cuBLAS，定位差距来源。

验收：
1. 写清楚“差距来自哪里”，比“盲目追绝对性能”更重要。

### Week 14：Attention Kernel 简化版
目标：
1. 写一个固定 shape 的简化 forward（可不做动态优化）。
2. 理解 FlashAttention 的工程实现思路。

验收：
1. 代码正确 + 有 profiling 数据。

### Week 15：Triton Inference Server 入门
目标：
1. 部署一个动态 batching 服务。
2. 测试并发请求下的延迟变化。

验收：
1. 提供并发测试结果与参数调优记录。

### Week 16：Ray Serve 与资源调度
目标：
1. 跑通多副本部署和基础路由策略。
2. 掌握共享 GPU 的常见策略。

验收：
1. 输出一份“流量突增应对方案”。

### Week 17：TensorRT Plugin 入门实战
目标：
1. 选一个算子（如 RMSNorm 或 RoPE）做插件原型。
2. 跑通 ONNX -> TensorRT 插件链路。

验收：
1. 插件代码可编译 + 有端到端性能对比。

### Week 18：端到端优化案例（第一次）
目标：
1. 选择一个开源模型做完整优化。
2. 从基线到优化形成完整报告。

验收：
1. 产出《模型推理优化报告 v1》。

---

## 阶段 D（Week 19-24）：项目落地与简历化输出

### Week 19：项目选题与设计文档
从以下方向选 1 个主项目 + 1 个副项目：
1. vLLM Prefix Caching 优化
2. 推理性能自动化分析工具
3. 分布式推理服务原型

产出：
1. 设计文档（背景、目标、方案、风险、里程碑）

### Week 20-21：主项目开发冲刺
要求：
1. 每周至少 5 次可运行提交
2. 每周末一次端到端演示

产出：
1. 可运行 demo + 基础文档

### Week 22：性能与稳定性打磨
任务：
1. 建立回归测试
2. 补齐 profiling 和错误处理

产出：
1. 性能回归报告

### Week 23：开源化与表达
任务：
1. 整理 README、使用教程、架构图
2. 写技术博客（优化过程 + 数据）

产出：
1. 项目展示页和技术文章

### Week 24：简历化与面试准备
任务：
1. 把项目改写为 STAR 结构成果描述
2. 准备 20 个高频问答（推理、CUDA、量化、服务）

产出：
1. 简历项目条目 v1
2. 面试问答手册

---

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
## 问题与排查
## 本周结论（最多 5 条）
## 下周计划
```

---

## 5. 成果目标（改为“相对提升”，不被硬件绑定）
不再用单一绝对值（如“必须 <100ms”），改为双指标：
1. 相对提升：同一环境下吞吐提升 >=30% 或延迟下降 >=25%
2. 工程质量：实验可复现、脚本可重跑、结论可解释

可选绝对目标（仅在硬件允许时启用）：
1. 7B 模型 first token latency 进入目标区间
2. decode throughput 达到业务要求

---

## 6. 你的学习优先级（遇到时间不够时按这个顺序）
1. 先保“实验可复现”
2. 再做“性能优化”
3. 最后追“极限指标”

一句话原则：
先成为一个“能稳定做实验并产出结论的人”，再成为“能把指标拉满的人”。
