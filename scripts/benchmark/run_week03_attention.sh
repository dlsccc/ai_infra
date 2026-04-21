#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-experiments/configs/week03_attention_cpu.yaml}

echo "[INFO] Running Week03 attention baseline: ${CONFIG_PATH}"
python src/benchmark/attention_naive_baseline.py --config "${CONFIG_PATH}"
