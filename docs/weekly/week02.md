# Week 02

## Goals
1. Export ResNet50 and BERT-base from PyTorch to dynamic ONNX.
2. Run ONNX Runtime CPU inference for multiple batch/sequence cases.
3. Summarize compatibility and performance comparison.

## TODO
- [x] Run ONNX export for ResNet50.
- [x] Run ONNX export for BERT-base.
- [ ] Inspect both ONNX graphs in Netron.
- [x] Run ORT baseline for ResNet50.
- [x] Run ORT baseline for BERT-base.
- [x] Fill report `docs/reports/week02_onnx_ort_baseline.md`.

## 实验tips
1. ONNX 导出动态轴
- 使用 `torch.onnx.export(..., dynamic_axes=...)` 声明可变维度。
- ResNet 通常开 `batch` 动态；BERT 通常开 `batch + seq_len` 动态。

2. BERT 导出推荐使用 wrapper
- 通过 `BertExportWrapper` 固定输入签名（`input_ids/attention_mask/token_type_ids`）和输出（`last_hidden_state`），防止导出后输入输出名称混乱。

3. ONNX Runtime 输入
- 不要硬编码输入名，先用 `sess.get_inputs()` 获取名称再组装 `feeds`。
- 这样能兼容模型是否带 `token_type_ids` 等差异。

4. ORT session 调优
- 先用 `ORT_ENABLE_ALL` 跑主结果，再用 `ORT_DISABLE_ALL` 做对照。
- `intra_op_num_threads` 对 CPU 影响大，建议扫描 4/8/12 寻找甜区。

5. 正确性校验
- 同一输入下对比 PyTorch 和 ORT 输出。
- 常用指标：`max_abs_err`、`mean_abs_err`、`allclose(rtol, atol)`。
- 如果误差偏大，先检查：输入一致性、dtype一致性、模型是否 `eval()`。

6. Netron 快速检查清单
- 输入输出名称和 shape 是否符合预期。
- 动态维是否以符号显示（`batch`、`seq_len`）而非固定数值。
- 是否存在大量重复 `Cast/Transpose/Identity` 可疑节点。

## Weekly Conclusion
- 
