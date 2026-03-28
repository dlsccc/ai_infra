# Week 01 - 推理基础 + PyTorch 推理链路

## 本周目标
- 跑通真实 PyTorch 基线（非 dry-run）
- 输出 CPU baseline 的 P50/P95 与吞吐
- 增加 CPU/GPU 对照（如果 CUDA 可用）
- ResNet50 CPU batch sweep
- ResNet50 CPU vs GPU对照
- BERT-base tokenization占比实验

## 任务清单
- [ ] 安装 Week1 依赖（torch/torchvision/numpy/pandas/matplotlib）
- [ ] 跑通 `week01_resnet_cpu.yaml`
- [ ] 完成 batch sweep（建议 1/2/4/8/16）
- [ ] 可选：跑 `week01_resnet_gpu.yaml` 做对照
- [ ] 输出并填写 `docs/reports/week01_pytorch_baseline.md`

## 实验tips
1. `time.time()` vs `time.pref_counter()`
    - `time.time()`是挂钟时间，受系统设置时间影响，如果手动改时间，会受影响。一般记录日志时间点时使用
    - `time.pref_counter()`不受系统时钟回拨影响，适合benchmark和函数耗时统计
2.  `torch.cuda.synchronize()`等待gpu同步跑完

## 核心实验结果
| Case | Device | Batch | P50(ms) | P95(ms) | Throughput(samples/s) | Note |
|---|---|---:|---:|---:|---:|---|
| A | CPU | 1 |  |  |  |  |
| B | CPU | 2 |  |  |  |  |
| C | CPU | 4 |  |  |  |  |
| D | CPU | 8 |  |  |  |  |
| E | CPU | 16 |  |  |  |  |
| F | GPU | 1 |  |  |  | optional |

## 本周结论（最多 5 条）
1. 
2. 
3. 

## 下周计划（Week 02）
- ONNX 导出与 ONNX Runtime 对比
