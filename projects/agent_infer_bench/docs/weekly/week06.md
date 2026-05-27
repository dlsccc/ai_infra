# Week 06

## Goals

1. Upgrade Week 5 smoke tests into baseline v1 runs.
2. Run the same controlled workloads on vLLM and SGLang.
3. Record real observed token counts from backend outputs where available.
4. Produce the first comparison summary for Agent-style workloads.

## Planned Work

### Day 1

- [ ] Align vLLM and SGLang baseline configs.
- [ ] Add baseline v1 workloads for `multi_tool_serial` and `long_observation`.
- [ ] Confirm repeat count and output directories.

### Day 2

- [ ] Revisit token accounting in both backends.
- [ ] Separate configured token budget from actual observed token counts.
- [ ] Document current limitations in token accounting.

### Day 3

- [ ] Run vLLM baseline v1.
- [ ] Generate summary files.
- [ ] Check for obvious anomalies in per-run variance.

### Day 4

- [ ] Run SGLang baseline v1.
- [ ] Generate summary files.
- [ ] Compare workload-level trends with vLLM.

### Day 5

- [ ] Build the first comparison table.
- [ ] Write initial observations:
- plain chat vs Agent-style workloads
- single tool vs multi-turn workloads
- long observation impact

### Day 6

- [ ] Re-run suspicious points if needed.
- [ ] Confirm that the baseline is stable enough for reporting.

### Day 7

- [ ] Finish `week06_baseline_v1.md`.
- [ ] Summarize known risks and Week 7 follow-up questions.

## Notes

- Week 6 is about controlled comparison, not about final conclusions.
- TTFT is still approximate or unavailable in offline mode; keep that limitation explicit.
- Synthetic token budget values are still workload construction hints, not strict tokenizer guarantees.

