# Week 03 - Attention vs FlashAttention Baseline Report

## 1. Experiment Metadata
| Field | Value |
|---|---|
| Date | 20260421 |
| Environment | Local GPU baseline |
| Device | GPU |
| PyTorch Version | |
| Warmup Iterations | 10 |
| Measured Iterations | 50 |
| batch_size / heads / head_dim | 1/8/64 |
| seq_lens | 64/128/256/512/1024 |

## 2. Benchmark Results

### 2.1 GPU
| Impl | Seq Len | Fwd P50 (ms) | Fwd P95 (ms) | Total P50 (ms) | Total P95 (ms) | Peak Mem P50 (MB) | Throughput (tokens/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| naive | 64 | 0.1321 | 0.1632 | 0.1956 | 0.2636 | 8.5 | 327225.1323 |
| naive | 128 | 0.108 | 0.1521 | 0.1797 | 0.2479 | 9.125 | 712250.7221 |
| naive | 256 | 0.1219 | 0.2407 | 0.1884 | 0.4169 | 11.125 | 1358695.622 |
| naive | 512 | 0.3185 | 0.3623 | 0.3927 | 0.4677 | 18.125 | 1303940.3498 |
| naive | 1024 | 1.0685 | 1.101 | 1.1724 | 1.2259 | 44.125 | 873433.987 |
| sdpa | 64 | 0.0399 | 0.0646 | 0.1085 | 0.1742 | 8.3755 | 589622.6443 |
| sdpa | 128 | 0.0378 | 0.0851 | 0.1258 | 0.2389 | 8.6255 | 1017423.3707 |
| sdpa | 256 | 0.0451 | 0.0661 | 0.1055 | 0.1914 | 9.1265 | 2427184.5016 |
| sdpa | 512 | 0.1198 | 0.148 | 0.1951 | 0.3274 | 10.1265 | 2624671.8796 |
| sdpa | 1024 | 0.3282 | 0.3334 | 0.4746 | 0.4839 | 12.1265 | 2157570.0181 |

## 3. Correctness Check (vs naive)
| Impl | Seq Len | Max Abs Err | Mean Abs Err | Note |
|---|---:|---:|---:|---|
| sdpa | 128 | 0.000977 | 5.1e-05 | |
| sdpa | 512 | 0.000732 | 2.9e-05 | |
| sdpa | 1024 | 0.000488 | 2.1e-05 | |

## 4. Curves / Plots

### 4.1 Latency vs Seq Len（`naive` vs `sdpa`）
基于 P50 的观察：
1. 随着 `seq_len` 增大，两条曲线都会上升，但 `sdpa` 的增长斜率明显更小。
2. 在前向耗时（forward）维度，`sdpa` 相对 `naive` 的加速区间为 2.66x ~ 3.31x。
3. 在总耗时（total）维度，加速区间为 1.43x ~ 2.47x；小序列下前后处理固定开销占比更高，会稀释总时延提升。

| Seq Len | Forward 加速比（`naive/sdpa`） | Total 加速比（`naive/sdpa`） |
|---:|---:|---:|
| 64 | 3.31x | 1.80x |
| 128 | 2.86x | 1.43x |
| 256 | 2.70x | 1.79x |
| 512 | 2.66x | 2.01x |
| 1024 | 3.26x | 2.47x |

### 4.2 Peak Memory vs Seq Len（GPU）
基于 P50 的观察：
1. `naive` 的显存随序列长度增长明显（8.5 MB -> 44.125 MB）。
2. `sdpa` 的显存增长更缓（8.3755 MB -> 12.1265 MB）。
3. 在 `seq_len=1024` 时，显存比值 `naive/sdpa = 3.64x`，说明长序列下 `sdpa` 的内存优势非常明显。

![Latency vs Seq](images/week03/week03_latency_vs_seq.png)

![Peak Memory vs Seq](images/week03/week03_peak_memory_vs_seq.png)

## 5. Key Findings
1. 在本实验设置（GPU, fp16, B=1, H=8, D=64）下，`sdpa` 在所有测试序列长度上均快于 `naive`：`forward_p50` 提升 2.66x~3.31x，`total_p50` 提升 1.43x~2.47x。
2. `sdpa` 的显存增长斜率更低；在 `seq_len=1024` 时，峰值显存由 `44.125 MB`（naive）降至 `12.1265 MB`（sdpa），约降低 72.5%。
3. 正确性检查稳定：`max_abs_err <= 9.77e-4`，`mean_abs_err` 在 `1e-5` 量级，符合 fp16 下预期误差范围。
4. 从实验现象看，`sdpa` 的优势在长序列时更明显，这与 FlashAttention 的 IO-aware 设计是吻合的：序列越长，避免 `N x N` 中间矩阵写回 HBM 的收益越大。

## 6. FlashAttention Notes (v1/v2/v3)
1. FA1:
   - 核心是 IO-aware tiling、online softmax 和 backward recomputation。
   - 重点不是减少 attention 的数学计算量，而是避免 materialize 完整的 `S=QK^T` 和 `P=softmax(S)`。
   - 通过把局部 `score tile` 留在 SRAM / shared memory 中，并维护行级状态 `m / l / O_i`，显著降低 HBM 访问与显存占用。
2. FA2:
   - 在 FA1 的正确性与 IO 优化基础上，进一步优化工作划分与并行策略。
   - 重点是提升 GPU 利用率、降低非 matmul 部分的开销，并让 kernel 调度更高效。
   - 如果说 FA1 解决“怎么避免大中间矩阵写回显存”，FA2 更像在解决“怎么把这件事在 GPU 上调度得更好”。
3. FA3:
   - 面向 Hopper 架构（尤其 H100）的进一步优化版本。
   - 更强调异步流水、tensor core 利用和硬件特性适配。
   - 不是推翻 FA1/FA2，而是在更新硬件上的持续工程深化。

## 7. Risks / Issues
1. 本周的实验对比对象是 `naive` 和 PyTorch `sdpa`，并不是直接运行官方 FlashAttention kernel，因此这里的实验结论更准确地说是在验证“融合 attention 后端在长序列下的速度/显存优势”，而不是对论文实现做逐 kernel 复现。
2. 当前实验规模仍然较小（B=1, H=8, D=64, seq_len<=1024），更适合作为概念验证。后续若有远程 GPU 环境，可扩展到更大 `seq_len / head_dim / batch_size` 做更贴近实际推理负载的 benchmark。

## 8. Next Step
- Week04: 进入 vLLM 架构阅读，重点是 `attention backend / scheduler / KV cache`。
- 阅读时重点带着这周的理解进入源码：query/key tile 如何切分、KV cache 如何组织、attention backend 如何选择、continuous batching 为什么会影响吞吐。
- 当前周的目标已经完成收口，不再继续补 CPU baseline，把精力集中到 Week04 的系统理解上。
