#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-experiments/configs/example_benchmark.yaml}

echo "[INFO] Running benchmark with config: ${CONFIG_PATH}"
python src/benchmark/run.py --config "${CONFIG_PATH}"
