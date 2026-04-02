#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-experiments/configs/week01_bert_cpu.yaml}

echo "[INFO] Running Week1 BERT tokenization baseline: ${CONFIG_PATH}"
python src/benchmark/bert_tokenization_baseline.py --config "${CONFIG_PATH}"
