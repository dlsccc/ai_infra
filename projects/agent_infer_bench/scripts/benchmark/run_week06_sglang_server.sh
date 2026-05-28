#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python scripts/benchmark/run_backend_baseline.py --config configs/week06_plain_sglang_server.yaml
python scripts/benchmark/run_backend_baseline.py --config configs/week06_agent_sglang_server.yaml

python scripts/analysis/make_tables.py --run-dir experiments/runs/week06/plain_baseline/sglang_server
python scripts/analysis/make_tables.py --run-dir experiments/runs/week06/agent_baseline/sglang_server
