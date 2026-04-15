#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-experiments/configs/week02_ort_resnet_cpu.yaml}

echo "[INFO] Running Week02 ONNX Runtime baseline: ${CONFIG_PATH}"
python src/benchmark/onnx_runtime_baseline.py --config "${CONFIG_PATH}"
