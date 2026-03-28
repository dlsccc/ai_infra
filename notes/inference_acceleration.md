# 推理加速（持续迭代）

目标：
1. 把“概念理解 -> 实验数据 -> 面试表达”打通，避免只学名词。
2. 每周增量沉淀，24 周后形成可用于跳槽的系统化笔记。

---

## 0. 使用规则（每周都按这个更新）

每周新增 3 类内容：
1. 概念卡：新增或修订 3-5 个术语，必须写清“定义 + 怎么测 + 常见误区”。
2. 证据卡：至少 1 组真实实验数据（表格 + 结论 + 反例）。
3. 表达卡：沉淀 1 条“面试可直接讲”的 STAR/问题-分析-结论话术。

---

## 1. 学习树总览（v0）

L1 指标与测量基础：
- latency、throughput、warmup、percentile（P50/P95/P99）
- batch、concurrency、queue time、service time
- 复现性（固定输入、固定轮次、固定版本）

L2 推理执行链路：
- 前处理（tokenization / decode）
- 模型前向（compute infer）
- 后处理（logits -> 输出格式）
- 端到端拆分（client send/recv、server queue、compute）

L3 系统优化手段：
- 动态批处理、并发调度
- 图优化/后端优化（PyTorch/ONNX Runtime/TensorRT）
- 精度与量化（FP16/INT8）

L4 架构与内核（后续周深入）：
- KV Cache、PagedAttention、FlashAttention
- CUDA kernel、memory hierarchy、occupancy

---

## 2. Week1 先修概念

### 2.1 必补概念

1. 推理性能指标 
   1. Latency vs. Throughput

      | 概念 | 定义 | 计量 | 衡量视角 | 
      |:---:|:---:|:---:|:---:|
      | Latency | 单次请求输入到输出的时间 | ms/s(seconds/token) | 用户体验（一个请求等待时间） |
      | Throughput | 单位时间处理请求数 | QPS | 系统能力（一秒能够服务几个人） |

      时延较为优秀的阈值是每分钟输出250个单词，约为人类的平均阅读速度
      区分端到端延迟与模型纯计算延迟
      吞吐提升不代表单次请求延迟一定下降

      <p align="center">
      <img src="./image/throughput_latency.png" width="1000"><br>
      <em>图1：throughput & latency关系示意图</em>
      </p>

   2. 一些常计算性能指标
      - TTFT：Time To First Token，首token生成时间，衡量prefill性能指标
      - E2EL：End to End Latency，端到端时延，从输入提示词到生成所有结果返回结束
      - ITL：Inter Token Latency，解码阶段每个token生成时间 $ ITL = \frac{E2EL - TTFT}{N_{tokens}-1} $ (但是也有一些ITL采用TBT计算方式)
      - TBT：Time Between Tokens，生成token之间的时间差，指某个token生成时间 $ TBT_i = latency_i - latency_{i-1} $
      - TPOT：Time Per Output Token，所有token生成的平均时间，包括首token（不过这里有争议，现在很多普遍默认不计算首token，采用ITL计算方式）
      - QPS：Query Per Second，每秒处理请求数 $ QPS = \frac{N_{requests}}{T_{seconds}} $
      - TPS：Token Per Sencond，每秒吞吐量输出的token数  $ TPS = \frac{n_tokens}{T_y-T_x} $
      - QPM：Query Per Minute，每分钟处理请求数
      - TP90：Top Percentile，至少90%请求满足该条件，称为分位数延迟，还有TP50、TP99。P50是中位数，代表典型延迟，P95是尾延迟基准，P99是严格尾延迟。只看平均值会掩盖尾部问题，一般线上更看重P95/P99
      - RPS：Requests Per Second，每秒请求数，用于控制测试时请求注入速率，是吞吐量测试的重要参考指标
      - Ramp Up：爬坡测试，修改RPS测试服务性能
      - SLO：Service Level Objective，服务质量目标，确保为客户提供优质服务的关键
      - MFU：Model Flops Utilization，衡量模型对GPU算力资源使用效率

2. Prefilling & Decoding
- prefill：预填充，并行处理输入的所有token
   - 一次完整的前向传播，生成KV cache，输出第一个token，这个时间与输入的token有关
- decoding：解码，逐个生成下一个token
   - output token的数量决定了需要进行多少次前向传播

3. 推理工程中的一些概念和指标
   1. Warmup（预热）
      - 定义：正式计时前先执行若干轮，消除首次执行抖动。
      - 典型抖动来源：懒加载、缓存建立、内核选择、内存页映射。
   2. Batch Size
      - 一般batch增大可提升吞吐，但可能拉高单请求延迟
      - batch_size为什么吃显存？因为每增加一个batch，多一份kv cache

      <p align="center">
      <img src="./image/batch_size估算实例.png" width="1000"><br>
      <em>图1：batch_size估算实例</em>
      </p>

   3. Concurrency 并发
      - 并发升高后，系统可能从“算力瓶颈”转为“排队瓶颈”。
   4. Queue Time vs Compute Time
      - 拆分来确认是排队慢还是计算慢
      - stage breakdown阶段拆分，至少是preprocess / forward / postprocess三段

4. Pytorch推理相关
   1. `model.eval()` vs `torch.inference_mode()` vs `torch.no_grad()`

      | 特性 | `model.eval()` | `torch.no_grad()` | `torch.inference_mode()` | 
      |:---:|:---:|:---:|:---:|
      | 主要职责 | 切换模型行为模式 | 禁用梯度计算 | 禁用梯度+更深度优化 |
      | 影响Dropout | ✅ | ❌ | ❌ |
      | 影响BatchNorm | ✅ | ❌ | ❌ |
      | 禁用梯度追踪 | ❌ | ✅ | ✅ |
      | 禁用张量版本计数器 | ❌ | ❌ | ✅ |
      | 禁用inplace历史检查 | ❌ | ❌ | ✅ |
      | 引入版本 | 早期 | 早期 | Pytorch 1.9+ |
      | 使用方式 | 方法调用 | 上下文管理器/装饰器 | 上下文管理器/装饰器 |

      一般实践时用 `model.eval()` + `torch.inference_mode()`
   2. `torch.utils.benchmark.Timer`
      1. 在统计gpu时间时，不能用 `time.time()`计时，因为gpu是异步执行的

      | 类/模块 | 核心能力 |	关键方法 |
      |:---:|:---:|:---:|
      | Timer | 壁钟计时 + warmup + CUDA同步 | blocked_autorange, adaptive_autorange, timeit |
      | Timer | 指令计数 | collect_callgrind |
      | Measurement | 结果存储 + 统计量 | .median, .iqr, .merge() |
      | CallgrindStats | Callgrind 结果分析 | .stats(), .delta(), .as_standardized() |
      | unctionCounts | 函数级指令操作 | .filter(), .transform(), .denoise(), 加减法 |
      | Compare |	多结果格式化对比 | .colorize(), .print(), .trim_significant_figures() |
      | Fuzzer | 随机输入生成（隐藏功能） | .take(n) → 随机张量生成器 |




##### 参考资料
[1] https://zhuanlan.zhihu.com/p/1983137653336585901
[2] https://zhuanlan.zhihu.com/p/680459342

---

## 3. Week1 学习资料（按优先级）

P0（本周必读）：
1. PyTorch `inference_mode` 文档（理解推理时为何比 `no_grad` 更少开销）
   - https://docs.pytorch.org/docs/stable/generated/torch.autograd.grad_mode.inference_mode.html
2. PyTorch `torch.utils.benchmark`（理解 warmup、同步、统计测量）
   - https://docs.pytorch.org/docs/stable/benchmark_utils.html
   - https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html
   - https://pytorch-cn.com/tutorials/recipes/recipes/benchmark.html
3. Triton Perf Analyzer 输出解释（学习 queue/compute/client latency 拆分）
   - https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/docs/benchmarking.html
4. Triton Model Analyzer 指标定义（对齐 latency/throughput/p95 指标口径）
   - https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_analyzer/docs/metrics.html

P1（建议预读，为 Week2/Week8 铺垫）：
1. ONNX Runtime 性能调优总览
   - https://onnxruntime.ai/docs/performance/tune-performance/
2. Triton GenAI-Perf（提前认识 TTFT / ITL / token throughput）
   - https://docs.nvidia.com/deeplearning/triton-inference-server/archives/triton-inference-server-2640/user-guide/docs/perf_benchmark/genai_perf.html

---

## 4. Week1 最小完成标准（可直接打勾）

- [ ] 我能解释 latency / throughput / warmup / P50-P95 的定义和关系。
- [ ] 我能解释为何 batch 上升时吞吐上升但单请求延迟可能变差。
- [ ] 我有一张 stage breakdown 表（preprocess/forward/postprocess）。
- [ ] 我有一张 batch sweep 表（>=3 组 batch）并给出结论。
- [ ] 我记录了测量协议（warmup、测量轮次、输入、版本）。

---

## 5. 周更模板（复制到后续 Week）

````markdown
## WeekXX 增量

### A. 概念卡（新增/修订）
1. 术语：
- 定义：
- 怎么测：
- 常见误区：

### B. 证据卡（实验）
- 配置：
- 关键结果（表格）：
- 结论：
- 反例/失败点：

### C. 表达卡（面试）
- 问题场景：
- 我的分析：
- 我的动作：
- 结果与权衡：
````

---

## 6. 面向跳槽的沉淀原则

1. 每个概念都要绑定至少一个真实数据证据。
2. 每个实验都要能回答“为什么这样测、为什么这样变”。
3. 每周至少产出 1 条可复述的性能优化故事。
4. 你的最终目标不是“知道术语”，而是“能做取舍并证明结果”。
