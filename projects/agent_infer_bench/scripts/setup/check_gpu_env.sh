#!/usr/bin/env bash
set -euo pipefail

echo "== System =="
uname -a || true

echo "== Python =="
python --version || true
which python || true

echo "== NVIDIA SMI =="
nvidia-smi || true

echo "== CUDA compiler =="
nvcc --version || true

echo "== PyTorch CUDA =="
python - <<'PY'
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            print(idx, props.name, round(props.total_memory / 1024**3, 2), "GiB")
except Exception as exc:
    print("torch check failed:", repr(exc))
PY

