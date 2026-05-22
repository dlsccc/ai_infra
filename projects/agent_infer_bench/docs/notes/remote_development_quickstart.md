# Remote Development Quickstart

## Mental Model

Use the local machine for editing, notes, and small mock tests. Use the remote GPU server only for GPU-dependent experiments.

## First-Time Flow

1. Get server IP, username, SSH port, and login method from the provider or lab admin.
2. Add or generate an SSH key locally.
3. Login from local PowerShell:

```powershell
ssh username@server_ip
```

4. On the server, clone the repo:

```bash
git clone https://github.com/dlsccc/ai_infra.git
cd ai_infra/projects/agent_infer_bench
```

5. Check the GPU environment:

```bash
bash scripts/setup/check_gpu_env.sh
```

6. Create a Python environment. Prefer conda or uv if available.
7. 检查当前驱动最高支持的cuda version
8. 安装torch，torch.version.cuda版本不能高于driver支持的能力
9. Install vLLM，通过 https://vllm.ai/releases 检查不同版本vllm对应的cuda，下载合适的版本
10. Install SGLang
11. Run a tiny model or mock benchmark before downloading large models.

## Working Rhythm

1. Edit locally.
2. Commit and push.
3. Pull on the remote server.
4. Run GPU experiments.
5. Pull results back only if they are small; otherwise keep raw files remote and commit summaries.

## Things Not To Commit

- Model weights.
- Large raw traces.
- Nsight profiles.
- Huge JSONL outputs.
- Access tokens or private keys.

