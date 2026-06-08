# Linux 速查手册：AI Infra 远程开发版

更新时间：2026-05-20  
目标：让你能在远程 GPU 服务器上熟练完成：登录、找路径、看资源、配环境、跑实验、查日志、处理错误。

## 0. Linux 命令怎么看

Linux 命令通常长这样：

```bash
命令 选项 参数
```

例子：

```bash
ls -lh /data
```

含义：

| 部分 | 含义 |
|---|---|
| `ls` | list，列出目录 |
| `-lh` | 选项，`l`=长格式，`h`=human readable |
| `/data` | 参数，要查看的路径 |

常见符号：

| 符号 | 含义 | 例子 |
|---|---|---|
| `.` | 当前目录 | `python ./run.py` |
| `..` | 上一级目录 | `cd ..` |
| `~` | 当前用户主目录 | `cd ~` |
| `/` | 根目录 | `cd /` |
| `*` | 通配符 | `ls *.py` |
| `|` | 管道，把前一个命令输出交给后一个 | `ps aux | grep python` |
| `>` | 覆盖写入文件 | `python run.py > run.log` |
| `>>` | 追加写入文件 | `echo hello >> a.txt` |
| `2>&1` | 把错误输出也写入同一个地方 | `python run.py > run.log 2>&1` |
| `&` | 后台运行 | `python run.py &` |

## 1. 常见命令先解释

这些命令你会天天见。

| 命令 | 英文/含义 | 怎么用 | 典型用途 |
|---|---|---|---|
| `echo` | 输出文本或变量 | `echo $HF_HOME` | 查看环境变量、写配置 |
| `pwd` | print working directory | `pwd` | 看当前在哪个目录 |
| `ls` | list | `ls -lh` | 看文件列表 |
| `cd` | change directory | `cd /data/projects` | 切目录 |
| `cat` | concatenate/打印文件 | `cat config.yaml` | 快速看小文件 |
| `less` | 分页查看 | `less run.log` | 看大文件 |
| `head` | 看开头 | `head -n 20 run.log` | 快速看文件开头 |
| `tail` | 看结尾 | `tail -f run.log` | 实时看日志 |
| `grep` | 文本搜索 | `grep -i error run.log` | 查错误 |
| `rg` | ripgrep，更快搜索 | `rg "TODO"` | 搜代码/文本 |
| `find` | 查找文件 | `find . -name "*.py"` | 找文件 |
| `mkdir` | make directory | `mkdir -p logs/week05` | 建目录 |
| `cp` | copy | `cp a.txt b.txt` | 复制 |
| `mv` | move | `mv old new` | 移动/改名 |
| `rm` | remove | `rm file.txt` | 删除 |
| `chmod` | change mode | `chmod +x run.sh` | 给脚本执行权限 |
| `chown` | change owner | `sudo chown -R $USER:$USER /data/projects` | 改文件归属 |
| `sudo` | superuser do | `sudo apt update` | 用管理员权限执行 |
| `apt` | Ubuntu 包管理器 | `sudo apt install git` | 装系统工具 |
| `ps` | process status | `ps aux | grep python` | 查进程 |
| `kill` | 结束进程 | `kill -9 PID` | 杀卡住的任务 |
| `top` | 资源监控 | `top` | 看 CPU/内存/进程 |
| `htop` | 更好用的 top | `htop` | 交互式看进程 |
| `df` | disk free | `df -h` | 看磁盘剩余 |
| `du` | disk usage | `du -sh *` | 看目录大小 |
| `free` | 内存查看 | `free -h` | 看内存 |
| `ssh` | secure shell | `ssh user@ip` | 登录服务器 |
| `scp` | secure copy | `scp file user@ip:/data/` | 传文件 |
| `git` | 版本控制 | `git pull` | 拉代码 |
| `nvidia-smi` | NVIDIA GPU 状态 | `nvidia-smi` | 看显存/GPU 进程 |
| `tmux` | 终端会话管理 | `tmux new -s bench` | 跑长任务不断线 |
| `curl` | 网络请求 | `curl -I https://github.com` | 测网络/请求 API |
| `wget` | 下载文件 | `wget URL` | 下载文件 |

### 几个最容易困惑的命令

| 命令 | 直觉理解 | 例子 |
|---|---|---|
| `echo` | “打印出来” | `echo $PATH` 看 PATH，`echo hello >> a.txt` 写入文件 |
| `sudo` | “用管理员身份做” | `sudo apt install tmux` |
| `ps` | “现在有哪些进程” | `ps aux | grep vllm` |
| `grep` | “从文本里找关键字” | `grep -i error logs/run.log` |
| `chmod +x` | “让脚本可以执行” | `chmod +x check_gpu_env.sh` |
| `df` | “磁盘还剩多少” | `df -h` |
| `du` | “这个目录占多少” | `du -sh ~/.cache` |
| `source` | “让当前 shell 读取配置/激活环境” | `source ~/.bashrc` |
| `export` | “设置环境变量” | `export HF_HOME=/data/hf_cache` |

## 2. 必须熟悉的目录

| 路径 | 含义 | 你怎么用 |
|---|---|---|
| `/` | 根目录 | 整个系统的起点 |
| `~` | 当前用户主目录 | 默认登录位置 |
| `/home/用户名` | 普通用户主目录 | 放少量配置和脚本 |
| `/root` | root 用户主目录 | root 登录时常见 |
| `/data` | 常见数据盘 | 放项目、模型、实验 |
| `/mnt/data` | 常见数据盘 | 同上，具体看平台 |
| `/workspace` | 常见工作目录 | 有些云平台使用 |
| `/tmp` | 临时目录 | 临时文件，不适合长期保存 |
| `/var/log` | 系统日志 | 查系统级问题 |

查看自己在哪：

```bash
pwd
```

查看有什么盘：

```bash
df -h
```

推荐项目布局：

```text
/data/projects/ai_infra
/data/hf_cache
/data/envs
/data/experiments
```

## 3. 远程服务器第一天必跑

登录后先跑这几条：

```bash
pwd
df -h
free -h
nvidia-smi
python --version
which python
```

看什么：

| 命令 | 看什么 | 判断 |
|---|---|---|
| `df -h` | 系统盘/数据盘剩余 | 数据盘最好 200GB+ |
| `free -h` | 内存 | 60GB 起步够用 |
| `nvidia-smi` | GPU 型号、显存、驱动 | 4090D 应该显示约 24GB |
| `python --version` | Python 版本 | 推荐 3.10/3.11 |
| `which python` | 当前 python 路径 | 判断是否在正确环境 |

## 4. 系统盘 vs 数据盘

| 类型 | 通常挂载 | 放什么 | 不适合放什么 |
|---|---|---|---|
| 系统盘 | `/` | 系统、驱动、apt 软件、日志 | 模型、大数据、实验结果 |
| 数据盘 | `/data` 或类似路径 | 代码、模型、HF cache、conda env、实验结果 | 系统核心文件 |

判断哪个是数据盘：

```bash
df -h
```

例子：

```text
/dev/vda1   30G   20G   10G   67% /
/dev/vdb1  300G   10G  290G    4% /data
```

这里 `/data` 就是数据盘。

## 5. 初始化 AI Infra 项目目录

假设数据盘是 `/data`：

```bash
mkdir -p /data/projects
mkdir -p /data/hf_cache
mkdir -p /data/envs
mkdir -p /data/experiments
```

设置模型缓存：

```bash
echo 'export HF_HOME=/data/hf_cache' >> ~/.bashrc
echo 'export HF_HUB_CACHE=/data/hf_cache/hub' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/data/hf_cache' >> ~/.bashrc
source ~/.bashrc
```

检查：

```bash
echo $HF_HOME
```

## 6. Git 常用流程

| 场景 | 命令 |
|---|---|
| 克隆项目 | `git clone https://github.com/dlsccc/ai_infra.git` |
| 浅克隆 | `git clone --depth 1 https://github.com/dlsccc/ai_infra.git` |
| 拉最新代码 | `git pull` |
| 看当前状态 | `git status` |
| 看改了什么 | `git diff` |
| 暂存文件 | `git add file.py` |
| 提交 | `git commit -m "message"` |
| 推送 | `git push origin main` |

GitHub 网络报 HTTP/2 错误时：

```bash
git config --global http.version HTTP/1.1
git clone --depth 1 https://github.com/dlsccc/ai_infra.git
```

远程服务器第一次拉项目：

```bash
cd /data/projects
git clone --depth 1 https://github.com/dlsccc/ai_infra.git
cd ai_infra/projects/agent_infer_bench
```

## 7. Python 环境

### venv 方式

| 操作 | 命令 |
|---|---|
| 创建环境 | `python3 -m venv /data/envs/agent-infer` |
| 激活环境 | `source /data/envs/agent-infer/bin/activate` |
| 退出环境 | `deactivate` |
| 升级 pip | `python -m pip install --upgrade pip` |
| 装依赖 | `pip install -r requirements.txt` |
| 看当前 Python | `which python` |
| 看已装包 | `pip list` |

### conda 方式

| 操作 | 命令 |
|---|---|
| 创建环境 | `conda create -n agent-infer python=3.11 -y` |
| 激活环境 | `conda activate agent-infer` |
| 看环境 | `conda env list` |
| 删除环境 | `conda remove -n agent-infer --all` |

建议：如果平台已有 conda，用 conda；没有就用 venv。

## 8. 运行脚本和日志

直接跑：

```bash
python scripts/benchmark/run_all_smoke.py
```

把输出保存到日志：

```bash
python scripts/benchmark/run_all_smoke.py > logs/smoke.log 2>&1
```

后台跑：

```bash
nohup python run.py > logs/run.log 2>&1 &
```

实时看日志：

```bash
tail -f logs/run.log
```

查错误：

```bash
grep -i error logs/run.log
grep -i oom logs/run.log
```

## 9. bashrc、source 和环境变量

`~/.bashrc` 是 bash 终端启动时会读取的配置文件。它常用来放环境变量、conda 初始化、alias 快捷命令。

| 概念 | 含义 | 例子 |
|---|---|---|
| `~/.bashrc` | 当前用户的 bash 配置文件 | `/root/.bashrc` |
| `export` | 设置环境变量，让后续命令可见 | `export HF_ENDPOINT=https://hf-mirror.com` |
| `source` | 在当前终端执行某个配置文件 | `source ~/.bashrc` |
| `alias` | 给长命令起短名字 | `alias ai_env="conda activate /root/autodl-tmp/envs/agent-infer"` |

### 为什么经常 source ~/.bashrc

写入配置：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
```

这只是把内容追加到文件里。当前终端不会自动知道，所以要：

```bash
source ~/.bashrc
```

或者重新打开一个新终端。

### Hugging Face 镜像和缓存推荐配置

```bash
cat >> ~/.bashrc <<'EOF'

# AgentInferBench cache and mirrors
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export TORCH_HOME=/root/autodl-tmp/hf_cache/torch
export MODELSCOPE_CACHE=/root/autodl-tmp/modelscope_cache
EOF

source ~/.bashrc
```

检查：

```bash
echo $HF_ENDPOINT
echo $HF_HOME
echo $HF_HUB_CACHE
```

### conda 环境路径太长怎么办

如果你用 `conda create -p /root/autodl-tmp/envs/agent-infer ...` 创建环境，那么激活时要用路径：

```bash
conda activate /root/autodl-tmp/envs/agent-infer
```

可以加 alias：

```bash
echo 'alias ai_env="conda activate /root/autodl-tmp/envs/agent-infer"' >> ~/.bashrc
source ~/.bashrc
```

以后新终端里：

```bash
ai_env
```

确认激活成功：

```bash
which python
python -c "import sys; print(sys.prefix)"
```

期望看到：

```text
/root/autodl-tmp/envs/agent-infer/bin/python
/root/autodl-tmp/envs/agent-infer
```

## 10. 多终端怎么用

远程服务器可以同时开多个终端。每个终端都是一个独立 shell，会话之间互不影响，但共享同一台机器的 GPU、CPU、磁盘。

| 场景 | 做法 |
|---|---|
| 关闭当前 SSH 终端 | `exit` 或 `Ctrl+D` |
| 本地再开一个 SSH 终端 | 新开 PowerShell，再 `ssh root@server_ip` |
| VS Code Remote 新终端 | `Terminal -> New Terminal` |
| 长任务不断线 | 用 `tmux` |

新终端通常要重新执行：

```bash
source ~/.bashrc
ai_env
cd /root/autodl-tmp/projects/ai_infra/projects/agent_infer_bench
```

如果没有 alias：

```bash
source ~/.bashrc
conda activate /root/autodl-tmp/envs/agent-infer
cd /root/autodl-tmp/projects/ai_infra/projects/agent_infer_bench
```

推荐多终端分工：

| 终端 | 用途 | 命令例子 |
|---|---|---|
| 终端 1 | 看 GPU | `watch -n 1 nvidia-smi` |
| 终端 2 | 跑 vLLM/SGLang server | `vllm serve ...` |
| 终端 3 | 跑 benchmark client | `python scripts/benchmark/run_all_smoke.py` |
| 终端 4 | 看日志 | `tail -f logs/run.log` |

注意：直接关闭普通 SSH 终端会停止该终端前台运行的任务。长任务请用 `tmux` 或 `nohup`。

## 11. tmux：长任务必会

| 场景 | 命令 |
|---|---|
| 新建会话 | `tmux new -s bench` |
| 退出但不停止 | `Ctrl+b` 然后按 `d` |
| 重新进入 | `tmux attach -t bench` |
| 查看会话 | `tmux ls` |
| 删除会话 | `tmux kill-session -t bench` |

个人启动vllm服务过程示例：
1. tmux new -s vllm
2. 运行相关服务
3. 滚动查看日志  ctrl+p 方向键  esc退出
4. tmux attach -t vllm 回到vllm tmux会话
5. ctrl+c 停掉vllm服务
6. exit 退出
7. tmux ls 查看是否还有这个服务

建议：下载模型、跑 benchmark、启动 vLLM/SGLang server 都放 tmux。

## 12. GPU 常用命令

| 场景 | 命令 |
|---|---|
| 看 GPU | `nvidia-smi` |
| 每秒刷新 | `watch -n 1 nvidia-smi` |
| 查 PyTorch CUDA | `python -c "import torch; print(torch.cuda.is_available())"` |
| 指定 GPU | `CUDA_VISIBLE_DEVICES=0 python run.py` |

`nvidia-smi` 重点看：

| 字段 | 含义 |
|---|---|
| GPU Name | 显卡型号 |
| Memory-Usage | 显存占用 |
| GPU-Util | GPU 利用率 |
| Processes | 哪些进程占 GPU |
| PID | 进程号，可用于 kill |

OOM 时先做：

```bash
nvidia-smi
```

如果有残留进程：

```bash
kill -9 PID
```

然后降低：

```text
max_model_len
concurrency
max_num_seqs
output_length
```

## 13. 进程和端口

| 场景 | 命令 |
|---|---|
| 看所有进程 | `ps aux` |
| 找 Python 进程 | `ps aux | grep python` |
| 杀进程 | `kill -9 PID` |
| 看端口 | `ss -lntp` |
| 查 8000 端口 | `ss -lntp | grep 8000` |

例子：

```bash
ps aux | grep vllm
ss -lntp | grep 8000
```

如果 vLLM server 占着端口：

```bash
kill -9 PID
```

## 14. 磁盘管理

| 场景 | 命令 |
|---|---|
| 看磁盘剩余 | `df -h` |
| 看当前目录大小 | `du -sh .` |
| 看各子目录大小 | `du -sh * | sort -h` |
| 找 1GB 以上文件 | `find . -type f -size +1G` |
| 清 pip cache | `pip cache purge` |
| 清 conda cache | `conda clean -a` |
| 看 HF cache 大小 | `du -sh $HF_HOME` |

磁盘危险信号：

```text
系统盘 / 剩余 < 10GB
数据盘使用率 > 90%
HF cache 放到了系统盘
```

## 15. 文件查看和搜索

| 场景 | 命令 |
|---|---|
| 看小文件 | `cat file.md` |
| 看大文件 | `less file.log` |
| 看前 50 行 | `head -n 50 file.log` |
| 看后 50 行 | `tail -n 50 file.log` |
| 实时看日志 | `tail -f file.log` |
| 搜关键字 | `grep "error" file.log` |
| 递归搜代码 | `rg "prefix"` |
| 列出所有文件 | `rg --files` |

安装 `rg`：

```bash
sudo apt install -y ripgrep
```

## 16. 权限

| 场景 | 命令 |
|---|---|
| 看权限 | `ls -l` |
| 给脚本执行权限 | `chmod +x run.sh` |
| 改目录归属 | `sudo chown -R $USER:$USER /data/projects` |
| 管理员安装软件 | `sudo apt install git` |

权限格式：

```text
-rwxr-xr-x
```

含义：

| 字母 | 含义 |
|---|---|
| `r` | read，读 |
| `w` | write，写 |
| `x` | execute，执行 |

脚本不能运行时：

```bash
chmod +x scripts/setup/check_gpu_env.sh
```

或者直接：

```bash
bash scripts/setup/check_gpu_env.sh
```

## 17. 网络和下载

| 场景 | 命令 |
|---|---|
| 测 GitHub | `curl -I https://github.com` |
| 下载文件 | `wget URL` |
| curl 下载 | `curl -L URL -o file` |
| 测端口服务 | `curl http://127.0.0.1:8000/health` |

端口转发：

```bash
ssh -L 8000:127.0.0.1:8000 user@server_ip
```

含义：把远程服务器的 8000 端口映射到本地 8000。

## 18. VS Code Remote SSH

推荐主开发方式：

```text
VS Code Remote SSH 打开远程项目
远程 terminal 跑实验
本地浏览器/文档写报告
GitHub 同步代码
```

流程：

1. 本地 VS Code 安装 `Remote - SSH`。
2. 连接服务器。
3. 打开远程目录：

```text
/data/projects/ai_infra/projects/agent_infer_bench
```

4. 在 VS Code terminal 中运行：

```bash
python scripts/benchmark/run_all_smoke.py
```

## 19. Week5 推荐命令顺序

第一次登录服务器后：

```bash
pwd
df -h
free -h
nvidia-smi
python --version
which python
```

初始化数据盘：

```bash
mkdir -p /data/projects /data/hf_cache /data/envs /data/experiments
echo 'export HF_HOME=/data/hf_cache' >> ~/.bashrc
echo 'export HF_HUB_CACHE=/data/hf_cache/hub' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/data/hf_cache' >> ~/.bashrc
source ~/.bashrc
```

拉项目：

```bash
cd /data/projects
git config --global http.version HTTP/1.1
git clone --depth 1 https://github.com/dlsccc/ai_infra.git
cd ai_infra/projects/agent_infer_bench
```

跑环境检查：

```bash
bash scripts/setup/check_gpu_env.sh
```

创建环境：

```bash
python3 -m venv /data/envs/agent-infer
source /data/envs/agent-infer/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

跑 mock benchmark：

```bash
python scripts/benchmark/run_all_smoke.py
python scripts/analysis/make_tables.py --run-dir experiments/runs/week05/mock_smoke
```

## 20. 常见错误速查

| 现象 | 先看什么 | 常见处理 |
|---|---|---|
| clone 失败 HTTP/2 | 网络/Git 配置 | `git config --global http.version HTTP/1.1` |
| 磁盘满 | `df -h`, `du -sh *` | 清 cache，移动到数据盘 |
| Python 包找不到 | `which python`, `pip list` | 激活正确环境 |
| CUDA 不可用 | `nvidia-smi`, `torch.cuda.is_available()` | 检查 PyTorch/CUDA |
| OOM | `nvidia-smi` | 降低上下文/并发，杀残留进程 |
| 端口占用 | `ss -lntp | grep 8000` | `kill -9 PID` |
| 日志太长 | `tail -f`, `grep error` | 只看关键错误 |
| 脚本没权限 | `ls -l` | `chmod +x script.sh` 或 `bash script.sh` |

## 21. 你必须背熟的 20 条

```bash
pwd
ls -lh
cd /data/projects
df -h
du -sh *
free -h
nvidia-smi
watch -n 1 nvidia-smi
python --version
which python
source /data/envs/agent-infer/bin/activate
git pull
git status
tail -f logs/run.log
grep -i error logs/run.log
ps aux | grep python
kill -9 PID
ss -lntp
tmux new -s bench
tmux attach -t bench
```

## 22. 最重要的一句话

Linux 日常不是会多少命令，而是遇到问题能迅速定位到是哪一类：

```text
路径错了？
权限错了？
环境错了？
磁盘满了？
GPU 被占了？
网络断了？
进程卡了？
端口冲突？
```

每次排错先看：

```bash
pwd
df -h
nvidia-smi
which python
git status
```

这五条能帮你解决大部分远程开发初期问题。
