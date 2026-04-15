# Scripts 使用说明

本文档说明 `scripts/` 目录下各脚本怎么运行，以及输出到哪里。

## 0. 前置准备

1. 进入项目根目录：`E:\ai_infra`
2. 推荐在当前终端设置：
   - PowerShell: `$env:PYTHONNOUSERSITE='1'`
3. 推荐使用 `ai_infra` 环境解释器运行 Python：
   - `D:\conda\envs\ai_infra\python.exe`

---

## 1. 脚本清单

- `scripts/benchmark/run_week01.sh`
  - Week01 ResNet50 基线入口（CPU/GPU 通过配置切换）
- `scripts/benchmark/run_week01_bert.sh`
  - Week01 BERT tokenization 基线入口（CPU/GPU 通过配置切换）
- `scripts/report/generate_week01_rows.py`
  - 从 `experiments/runs/week01/**/metrics.json` 自动生成报告表格行
- `scripts/setup/check_gpu_env.sh`
  - 简单检查 GPU/CUDA/PyTorch 环境

---

## 2. Week01 ResNet 基线

### 2.1 PowerShell（推荐）

CPU:
```powershell
$env:PYTHONNOUSERSITE='1'
& "D:\conda\envs\ai_infra\python.exe" src/benchmark/pytorch_baseline.py --config experiments/configs/week01_resnet_cpu.yaml
```

GPU:
```powershell
$env:PYTHONNOUSERSITE='1'
& "D:\conda\envs\ai_infra\python.exe" src/benchmark/pytorch_baseline.py --config experiments/configs/week01_resnet_gpu.yaml
```

### 2.2 Bash（Git Bash/WSL）

```bash
bash scripts/benchmark/run_week01.sh experiments/configs/week01_resnet_cpu.yaml
bash scripts/benchmark/run_week01.sh experiments/configs/week01_resnet_gpu.yaml
```

输出目录示例：
- `experiments/runs/week01/resnet_cpu/<timestamp>/`
- `experiments/runs/week01/resnet_gpu/<timestamp>/`

输出文件：
- `config_snapshot.yaml`
- `metrics.json`
- `summary.md`

---

## 3. Week01 BERT Tokenization 基线

### 3.1 PowerShell（推荐）

CPU:
```powershell
$env:PYTHONNOUSERSITE='1'
& "D:\conda\envs\ai_infra\python.exe" src/benchmark/bert_tokenization_baseline.py --config experiments/configs/week01_bert_cpu.yaml
```

GPU:
```powershell
$env:PYTHONNOUSERSITE='1'
& "D:\conda\envs\ai_infra\python.exe" src/benchmark/bert_tokenization_baseline.py --config experiments/configs/week01_bert_gpu.yaml
```

### 3.2 Bash（Git Bash/WSL）

```bash
bash scripts/benchmark/run_week01_bert.sh experiments/configs/week01_bert_cpu.yaml
bash scripts/benchmark/run_week01_bert.sh experiments/configs/week01_bert_gpu.yaml
```

输出目录示例：
- `experiments/runs/week01/bert_cpu/<timestamp>/`
- `experiments/runs/week01/bert_gpu/<timestamp>/`

输出文件：
- `config_snapshot.yaml`
- `metrics.json`
- `summary.md`

---

## 4. 环境检查脚本

```bash
bash scripts/setup/check_gpu_env.sh
```

用于快速检查：
1. `nvidia-smi` 是否可用
2. `nvcc` 是否可用
3. PyTorch 是否识别 CUDA

---

## 5. 常见问题

1. `ModuleNotFoundError` 或包版本错乱
   - 先确认解释器是否为 `ai_infra`：
     - `& "D:\conda\envs\ai_infra\python.exe" -V`
   - 再设置：`$env:PYTHONNOUSERSITE='1'`

2. GPU 模式报错 `torch.cuda.is_available() is False`
   - 先运行：
     - `nvidia-smi`
     - `python -c "import torch; print(torch.cuda.is_available())"`

3. Bash 脚本在 PowerShell 中不可直接执行
   - 直接使用 Python 命令方式（推荐）
   - 或在 Git Bash/WSL 中执行 `bash scripts/...`

---

## 6. Week02 ONNX + ORT

### 6.1 安装依赖（首次）

```powershell
$env:PYTHONNOUSERSITE='1'
& "D:\conda\envs\ai_infra\python.exe" -m pip install -r requirements-dev.txt
```

### 6.2 导出动态 ONNX

ResNet50:
```powershell
$env:PYTHONNOUSERSITE='1'
& "D:\conda\envs\ai_infra\python.exe" src/benchmark/onnx_dynamic_export.py --config experiments/configs/week02_export_resnet_dynamic.yaml
```

BERT-base:
```powershell
$env:PYTHONNOUSERSITE='1'
& "D:\conda\envs\ai_infra\python.exe" src/benchmark/onnx_dynamic_export.py --config experiments/configs/week02_export_bert_dynamic.yaml
```

### 6.3 ONNX Runtime 基线

ResNet50:
```powershell
$env:PYTHONNOUSERSITE='1'
& "D:\conda\envs\ai_infra\python.exe" src/benchmark/onnx_runtime_baseline.py --config experiments/configs/week02_ort_resnet_cpu.yaml
```

BERT-base:
```powershell
$env:PYTHONNOUSERSITE='1'
& "D:\conda\envs\ai_infra\python.exe" src/benchmark/onnx_runtime_baseline.py --config experiments/configs/week02_ort_bert_cpu.yaml
```

### 6.4 Bash 入口

```bash
bash scripts/benchmark/run_week02_export.sh experiments/configs/week02_export_resnet_dynamic.yaml
bash scripts/benchmark/run_week02_export.sh experiments/configs/week02_export_bert_dynamic.yaml

bash scripts/benchmark/run_week02_ort.sh experiments/configs/week02_ort_resnet_cpu.yaml
bash scripts/benchmark/run_week02_ort.sh experiments/configs/week02_ort_bert_cpu.yaml
```

### 6.5 输出目录

- ONNX 导出元数据：`experiments/runs/week02/export/...`
- ORT 结果：`experiments/runs/week02/ort/...`
- 模型文件：`artifacts/onnx/week02/*.onnx`

