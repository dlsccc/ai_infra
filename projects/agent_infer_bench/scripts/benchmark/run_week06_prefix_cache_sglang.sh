#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python scripts/analysis/inspect_workload_tokens.py --config configs/week06_prefix_cache_sglang.yaml --context-limit 4096
python scripts/benchmark/run_backend_baseline.py --config configs/week06_prefix_cache_sglang.yaml
