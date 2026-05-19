# Week 03

## Goals
1. Build a naive attention baseline and compare with PyTorch SDPA.
2. Measure latency and memory trends across sequence lengths.
3. Summarize FlashAttention v1/v2/v3 key improvements.

## TODO
- [x] Skip CPU baseline for this week and focus on concept closure.
- [x] Run GPU baseline (`week03_attention_gpu.yaml`).
- [x] Plot latency vs seq_len and memory vs seq_len.
- [x] Fill report `docs/reports/week03_attention_flash_baseline.md`.
- [x] Write `FlashAttention` notes.

## 实验tips
1. `torch.matmul` 在高维中的含义
- 只对最后两维做矩阵乘法，前面维度作为 batch 维。
- attention 中：`q:[B,H,L,D]` 与 `k^T:[B,H,D,L]` 相乘得到 `scores:[B,H,L,L]`。

2. `q.size(-2)` 的用法
- 取倒数第 2 个维度的大小，在 `[B,H,L,D]` 中就是 `seq_len=L`。

3. `is_causal` 和因果掩码
- `is_causal=True` 表示自回归场景，当前位置不能看未来 token。
- 常见做法是将未来位置填充为 `-inf`，使 softmax 后权重为 0。

4. `masked_fill(mask == 0, -inf)` 语义
- 约定：`mask=1` 可见，`mask=0` 不可见。
- 因果 mask 可用 `torch.tril(...)` 构造，然后广播到 `[B,H,L,L]`。

5. GPU 计时口径
- 分阶段计时优先 `torch.cuda.Event`，并在读取前 `event.synchronize()`。
- 如果使用 `perf_counter` 测 GPU 代码，必须同步，否则结果会偏小。

6. 显存峰值统计
- 每轮前：`torch.cuda.reset_peak_memory_stats(device)`。
- 读取峰值：`torch.cuda.max_memory_allocated(device)`，再转 MB/GB。
- 该值是“已分配显存峰值”，更适合基准测试记录。

7. naive vs SDPA 误差判断
- fp16 下 `max_abs_err` 在 `1e-3 ~ 1e-2` 较常见，到 `1e-1` 需重点排查。
- 对比前先保证同一输入、同一 mask、同一 dtype。

## Weekly Conclusion
- 本周的实验目标已经完成：基于 GPU 跑通了 `naive attention vs PyTorch SDPA` 的基线对比，并得到了 latency / memory 随 `seq_len` 增长的趋势图。
- 实验上观察到的现象和论文理解是一致的：`sdpa` 在长序列下不仅更快，而且显存增长明显更缓，这支持了“attention 优化的核心收益来自减少 `N x N` 中间矩阵的 HBM 访问”这一判断。
- 本周真正完成的不是“会背 FlashAttention 公式”，而是把三件事串起来了：
  1. `tiling`：把 `Q/K/V` 切成小块在 SRAM 中流式处理；
  2. `online softmax`：维护行级状态 `m / l / O_i`，在看不到整行 score 时仍然精确更新 softmax；
  3. `recomputation`：前向不存完整 `S/P`，反向按 tile 重算局部结果。
- 一个重要的纠偏是：`K^T` 是为了矩阵乘法维度匹配，不是把单个向量倒序；`m / l / O_i` 是行状态，跨 tile 的实现中需要在 HBM 和片上工作区之间反复读写，而不是永远只存在于寄存器里。
- 这一周不再补 CPU baseline。对当前学习目标来说，继续补一组 CPU 数据的收益已经低于把论文理解、实验现象和工程直觉真正收口。
- 下周进入 Week04，重点是带着本周形成的 attention / IO / tile 视角去看 vLLM 的 `attention backend`、`scheduler` 和 `KV cache`，开始从单个 kernel 的理解走向推理系统级理解。
