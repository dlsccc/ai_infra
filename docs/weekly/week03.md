# Week 03

## Goals
1. Build a naive attention baseline and compare with PyTorch SDPA.
2. Measure latency and memory trends across sequence lengths.
3. Summarize FlashAttention v1/v2/v3 key improvements.

## TODO
- [ ] Run CPU baseline (`week03_attention_cpu.yaml`).
- [x] Run GPU baseline (`week03_attention_gpu.yaml`).
- [x] Plot latency vs seq_len and memory vs seq_len.
- [x] Fill report `docs/reports/week03_attention_flash_baseline.md`.
- [ ] Write `FlashAttention` notes.

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
- 
