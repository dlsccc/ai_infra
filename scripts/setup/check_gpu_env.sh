#!/usr/bin/env bash
set -euo pipefail

echo "===== GPU Environment Check ====="

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[OK] nvidia-smi found"
  nvidia-smi
else
  echo "[WARN] nvidia-smi not found"
fi

echo "----- CUDA Compiler -----"
if command -v nvcc >/dev/null 2>&1; then
  nvcc --version
else
  echo "[WARN] nvcc not found"
fi

echo "----- Python Torch CUDA Check -----"
python - <<'PY'
import sys
try:
    import torch
    print(f"python: {sys.version.split()[0]}")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device count: {torch.cuda.device_count()}")
        print(f"device 0: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"[WARN] torch check failed: {e}")
PY
