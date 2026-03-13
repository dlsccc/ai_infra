# Week 00 - Dry-run Benchmark Report

## Experiment Metadata
| Field | Value |
|---|---|
| Experiment Name | example_baseline |
| Date | 2026-03-13 |
| Mode | dry_run |
| Model | resnet50 |
| Backend | pytorch |
| Precision | fp32 |
| Batch Size | 1 |
| Warmup Iterations | 20 |
| Measured Iterations | 100 |
| Output Directory | `experiments/runs/example_baseline/20260313_100258` |

## Result Summary
| Metric | Value | Notes |
|---|---:|---|
| P50 Latency (ms) | 12.0 | dry-run simulated metric |
| P95 Latency (ms) | 14.16 | dry-run simulated metric |
| Throughput (tokens/s) | 83.333 | dry-run simulated metric |
| Peak GPU Memory (GB) | 0.68 | dry-run simulated metric |

## Analysis
1. Workflow has been validated end-to-end: config -> run -> output files.
2. This run does not represent real model performance and is only for Week 00 pipeline validation.
3. In Week 01, replace dry-run metrics with real measured metrics.

## Next Action
1. Add a real benchmark entry point in `src/benchmark/run.py`.
2. Start Week 01 PyTorch inference baseline and fill the same report template with measured numbers.
