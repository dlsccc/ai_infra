# Week 02 - ONNX + ONNX Runtime Baseline Report

## 1. Experiment Metadata
| Field | Value |
|---|---|
| Date | 2026.4.14 |
| Environment | ai_infra (`D:\conda\envs\ai_infra`, `PYTHONNOUSERSITE=1`) |
| PyTorch Version | 2.6.0+cu124 |
| ONNX Version | 1.21.0 |
| ONNX Runtime Version | 1.24.4 |
| CPU | Intel64 Family 6 Model 186 Stepping 2, GenuineIntel |

## 2. Export Results (Dynamic ONNX)

### 2.1 ResNet50 Export
| Item | Value |
|---|---|
| Config | `experiments/configs/week02_export_resnet_dynamic.yaml` |
| ONNX Path | `artifacts/onnx/week02/resnet50_dynamic.onnx` |
| Opset | 17 |
| Dynamic Axes | batch（`dynamic_batch=true`, `dynamic_hw=false`） |
| Notes | 导出成功，可被 ORT CPU 正常加载并完成多 batch 推理 |

### 2.2 BERT-base Export
| Item | Value |
|---|---|
| Config | `experiments/configs/week02_export_bert_dynamic.yaml` |
| ONNX Path | `artifacts/onnx/week02/bert_base_dynamic.onnx` |
| Opset | 17 |
| Dynamic Axes | batch / seq_len |
| Notes | 导出成功，ORT CPU 可在不同 batch/seq_len 下运行 |

## 3. ONNX Runtime Inference

### 3.1 ResNet50 (ORT CPU)
| Batch | H | W | Fwd P50 (ms) | Fwd P95 (ms) | Total P50 (ms) | Total P95 (ms) | Throughput (samples/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 224 | 224 | 68.0261 | 78.4606 | 81.0969 | 91.1096 | 12.3309 |
| 4 | 224 | 224 | 269.643 | 286.8093 | 319.8502 | 338.9555 | 12.5059 |
| 8 | 224 | 224 | 311.5298 | 558.0729 | 364.7564 | 665.4297 | 21.9324 |
| 16 | 224 | 224 | 990.2577 | 1124.167 | 1186.6652 | 1330.0246 | 13.4832 |

### 3.2 BERT-base (ORT CPU)
| Batch | Seq Len | Fwd P50 (ms) | Fwd P95 (ms) | Total P50 (ms) | Total P95 (ms) | Throughput (samples/s) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 16.4657 | 18.22 | 16.5838 | 18.3816 | 60.2998 |
| 1 | 128 | 51.2185 | 58.2749 | 51.3861 | 58.4205 | 19.4605 |
| 1 | 256 | 452.496 | 487.5086 | 452.9564 | 487.9614 | 2.2077 |
| 4 | 128 | 871.0641 | 919.877 | 871.4697 | 920.3622 | 4.5899 |
| 8 | 128 | 1688.5331 | 1789.0886 | 1688.935 | 1789.4834 | 4.7367 |

## 4. ORT vs PyTorch (Optional)
| Model | Case | PyTorch Total P50 (ms) | ORT Total P50 (ms) | Relative Change |
|---|---|---:|---:|---:|
| ResNet50 | batch=1 | 56.2689 | 81.0969 | +44.12% |
| ResNet50 | batch=8 | 261.1553 | 364.7564 | +39.67% |
| BERT-base | batch=1, seq=128 | 98.5143 | 51.3861 | -47.84% |

## 5. Compatibility / Graph Notes
1. 本周已跑通 `PyTorch -> ONNX(opset 17) -> ORT(CPUExecutionProvider)` 全链路，ResNet50 与 BERT-base 均可稳定推理。
2. ResNet 导出配置为动态 batch（非动态 H/W）；BERT 导出配置为动态 batch + 动态 seq_len，符合本周实验目标。
3. ORT 的表现依赖模型和输入形状，不是“统一更快”；需要按 case 做基准验证。

## 6. Key Findings
1. ResNet50（CPU）场景下，ORT 在 batch=1/8 相比 Week01 PyTorch baseline 更慢（+44.12%、+39.67%），说明在当前硬件与线程配置下，ORT 对该模型未形成优势。
2. BERT-base 在 `batch=1, seq=128` 下，ORT 显著快于 Week01 PyTorch baseline（-47.84%），说明 ORT 在该模型/形状组合上具备优化收益。
3. BERT 对输入形状（batch、seq_len）非常敏感：`seq_len` 从 128 提到 256 或 `batch` 从 1 提到 4/8 时，时延显著上升，吞吐下降，提示 CPU 线程与内存访问已成为瓶颈。

## 7. Risks / Issues
1. 当前仅在单机 CPU、固定线程（`intra_op_num_threads=8`）下评估，结论对其他线程设置和硬件的可迁移性有限。
2. 本周主要比较的是“合成输入推理耗时”；未纳入真实 tokenizer、I/O、服务框架调度等端到端开销，后续需补充系统级测量。


## 8. Netron Observation Template

### 8.1 Basic Information
| Item | ResNet50 ONNX | BERT-base ONNX |
|---|---|---|
| File Path | `artifacts/onnx/week02/resnet50_dynamic.onnx` | `artifacts/onnx/week02/bert_base_dynamic.onnx` |
| Opset | | |
| Model Size (MB) | | |
| Node Count (approx.) | | |
| Initializer Count (approx.) | | |

### 8.2 Input / Output Signature Check
| Check Item | ResNet50 | BERT-base |
|---|---|---|
| Input Names | | |
| Input Dtype | | |
| Input Shape (show dynamic dims) | | |
| Output Names | | |
| Output Dtype | | |
| Output Shape (show dynamic dims) | | |

### 8.3 Dynamic Axes Verification
1. ResNet50 expected: dynamic batch only (`N` dynamic, `H/W` static by current config).
2. BERT expected: dynamic `batch` + dynamic `seq_len`.
3. Netron actual observation:
- ResNet50:
- BERT-base:

### 8.4 Main Operator Path (Top-5)
- ResNet50 typical path observed:
1. 
2. 
3. 
4. 
5. 

- BERT-base typical path observed:
1. 
2. 
3. 
4. 
5. 

### 8.5 Potential Redundant / Suspicious Nodes
| Model | Node Type | Why Suspicious | Potential Impact |
|---|---|---|---|
| ResNet50 | | | |
| BERT-base | | | |

### 8.6 Graph Observation Conclusion
1. 
2. 
3. 
