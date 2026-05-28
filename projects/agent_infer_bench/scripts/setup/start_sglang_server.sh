#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
API_KEY="${API_KEY:-EMPTY}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[SGLang] model=${MODEL_PATH}"
echo "[SGLang] endpoint=http://${HOST}:${PORT}"

exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --api-key "${API_KEY}" \
  --mem-fraction-static "${MEM_FRACTION_STATIC}" \
  --sampling-defaults openai
