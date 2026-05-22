# Week 05

## Goals

1. Build the AgentInferBench project skeleton.
2. Validate the local mock benchmark path.
3. Prepare remote GPU environment checks.
4. Run vLLM and SGLang hello-world tests after the server is ready.

## Daily Checklist

### Day 1

- [ ] Confirm remote GPU server access.
- [ ] Run `bash scripts/setup/check_gpu_env.sh` on the server.
- [ ] Record GPU, CUDA, driver, Python, PyTorch versions.
- [ ] Run vLLM hello world.
- [ ] Run SGLang hello world.

### Day 2

- [ ] Finalize benchmark result JSON format.
- [ ] Validate mock backend locally.
- [ ] Save smoke test result.

### Day 3

- [ ] Implement or configure vLLM backend MVP.
- [ ] Run 3 small vLLM cases.

### Day 4

- [ ] Implement or configure SGLang backend MVP.
- [ ] Run the same 3 small SGLang cases.

### Day 5

- [ ] Generate Agent workload samples.
- [ ] Validate token breakdown and prefix overlap proxy.

### Day 6

- [ ] Add summary script and first tables.
- [ ] Write environment setup report.

### Day 7

- [ ] Buffer day: fix environment, scripts, and docs.

## Notes

- Keep raw large benchmark outputs out of git unless they are small curated samples.
- Do not expand into speculative decoding or quantization this week.

##### 2026/5/22 vLLM smoke
vLLM + Qwen2.5-7B-Instruct hello world succeeded
max_model_len=4096
gpu_memory_utilization=0.85
遇到的问题：
- HF 直连失败，改用 HF_ENDPOINT / hf-mirror
- vllm版本过高导致torch CUDA 13.0 与 driver CUDA 12.8 不兼容，降低vllm版本
- tokenizer 版本问题，调整 transformers/tokenizers

