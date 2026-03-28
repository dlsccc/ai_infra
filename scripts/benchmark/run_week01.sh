#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-experiments/configs/week01_resnet_cpu.yaml}

echo "[INFO] Running Week1 baseline with config: ${CONFIG_PATH}"
python src/benchmark/pytorch_baseline.py --config "${CONFIG_PATH}"
