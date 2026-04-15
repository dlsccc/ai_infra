#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-experiments/configs/week02_export_resnet_dynamic.yaml}

echo "[INFO] Running Week02 ONNX export: ${CONFIG_PATH}"
python src/benchmark/onnx_dynamic_export.py --config "${CONFIG_PATH}"
