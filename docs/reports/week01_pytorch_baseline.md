# Week 01 - PyTorch Baseline Report

## 1. Experiment Metadata
| Field | Value |
|---|---|
| Date | 20260328 |
| Environment | ai_infra |
| Model | resnet50 |
| Device | CPU / GPU |
| Warmup Iterations | 20 |
| Measured Iterations | 100 |

## 2. Batch Sweep Results
| Device | Batch | P50 (ms) | P95 (ms) | Throughput (samples/s) |
|---|---:|---:|---:|---:|
| CPU | 1 | 56.2689 | 71.1906 | 17.7718 |
| CPU | 2 | 87.0823 | 101.2324 | 22.9668 |
| CPU | 4 | 141.3436 | 169.9707 | 28.2998 |
| CPU | 8 | 261.1553 | 290.8258 | 30.6331 |
| CPU | 16 | 1751.2891 | 1900.4081 | 9.1361 |
| GPU | 1 | 6.734 | 8.0563 | 148.5001 |
| GPU | 2 | 11.0874 | 11.2805 | 180.3849 |
| GPU | 4 | 20.3296 | 33.0884 | 196.757 |
| GPU | 8 | 39.8119 | 41.0712 | 200.9449 |
| GPU | 16 | 79.4248 | 79.9078 | 201.4484 |

## 3. Stage Breakdown (P50)
| Device | Batch | Preprocess (ms) | Forward (ms) | Postprocess (ms) |
|---|---:|---:|---:|---:|
| CPU | 1 | 0.4982 | 55.727 | 0.0366 |
| GPU | 1 | 0.0381 | 4.3632 | 2.4076 |

## 4. Key Findings
1. CPU：batch 1->8 吞吐持续上升（17.8 -> 30.6 samples/s），batch=16 延迟暴涨（P50 1751ms）且吞吐掉到 9.1，说明 CPU 已进入明显退化区（常见于线程/缓存/内存带宽/热降频影响）。所以对于本台机器来说，最佳batch_size在4-8之间。
2. GPU：吞吐从 148 -> 201 后基本饱和（batch>=8 提升很小）。

## 5. Risks / Issues
暂无

## 6. Next Step
- Week02: PyTorch -> ONNX -> ONNX Runtime
