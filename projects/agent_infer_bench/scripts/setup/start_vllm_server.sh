#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_KEY="${API_KEY:-EMPTY}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[vLLM] model=${MODEL_PATH}"
echo "[vLLM] endpoint=http://${HOST}:${PORT}"

exec env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  vllm serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype auto \
  --api-key "${API_KEY}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --trust-remote-code \
  --generation-config vllm
