# 工程化经验手册

目标：用最简洁的方式记录“以后还会用到”的工程实践。

## 记录规则
1. 每条经验尽量控制在 10-20 行。
2. 固定结构：是什么、什么时候用、最小示例、常见坑。
3. 只记录已实践过或近期会实践的内容。

---

## 01. YAML 配置文件

是什么：
- YAML 是配置格式（`.yaml` / `.yml`），适合存放实验参数和运行选项。

什么时候用：
1. 同一脚本要跑多组参数（batch、precision、model path）。
2. 希望“改参数不改代码”。
3. 需要可复现实验（报告可直接引用配置文件）。

最小示例：
```yaml
experiment_name: example_baseline
model: resnet50
backend: pytorch
precision: fp32
batch_size: 1
warmup_iters: 20
measure_iters: 100
```

在当前仓库的用法：
1. 配置放在 `experiments/configs/*.yaml`。
2. 运行脚本读取配置（如 `scripts/benchmark/run_example.sh`）。
3. 结果写入 `experiments/runs/...`。

常见坑：
1. 缩进错误（YAML 对缩进敏感）。
2. 数字/字符串类型混淆（`"1"` 和 `1`）。
3. 字段重名导致参数被覆盖。

---

## 02. 常用指令：环境管理

| 指令 | 用途 | 什么时候用 | 备注 |
|---|---|---|---|
| `conda create -n ai_infra python=3.11 -y` | 创建独立环境 | 新项目初始化 | 避免污染 base 环境 |
| `conda activate ai_infra` | 激活环境 | 每次开始开发前 | 激活后再装包/运行脚本 |
| `python -m pip install --upgrade pip` | 升级 pip | 新环境建好后 | 减少安装兼容问题 |
| `pip install -r requirements-dev.txt` | 安装开发依赖 | 拉起项目时 | 安装你维护的直接依赖 |
| `conda env export --from-history > environment.yml` | 导出 Conda 基础蓝图 | 包变更后 | 文件简洁，适合长期维护 |
| `pip freeze > requirements-lock.txt` | 导出当前完整快照 | 包变更后 | 用于严格复现，文件较长 |
| `conda env list` | 查看环境列表 | 查环境是否存在 | 快速确认环境名 |
| `conda list` | 查看当前 Conda 包 | 排查冲突时 | 仅看当前激活环境 |
| `pip list` | 查看当前 pip 包 | 对比依赖时 | 与 `requirements-lock.txt` 对照 |
| `where python` | 检查 Python 路径 | 怀疑激活错环境时 | Windows 下常用 |
| `python -V` | 查看 Python 版本 | 环境核对 | 应与计划版本一致 |
| `pip -V` | 查看 pip 对应解释器 | 环境核对 | 防止 pip 指到别的 Python |

---

## 03. 常用指令：环境复建

| 场景 | 指令 | 作用 |
|---|---|---|
| 用 Conda 蓝图重建基础环境 | `conda env create -f environment.yml` | 按基础依赖恢复环境 |
| 激活复建后的环境 | `conda activate ai_infra` | 进入目标环境 |
| 用锁文件恢复 pip 依赖 | `pip install -r requirements-lock.txt` | 精确复现当时包版本 |
| 验证解释器是否正确 | `where python` | 避免装错环境 |
| 验证版本是否匹配 | `python -V` / `pip -V` | 快速确认运行上下文 |
| 验证最小流程是否可跑 | `python src/benchmark/run.py --config experiments/configs/example_benchmark.yaml` | 复建后做冒烟测试 |

---

## 04. 维护流程（建议固定执行）

1. 改依赖前先激活环境：`conda activate ai_infra`。
2. 安装/升级后先跑最小冒烟命令，确认未破坏流程。
3. 刷新两份文件：
   - `conda env export --from-history > environment.yml`
   - `pip freeze > requirements-lock.txt`
4. 提交前检查 lock 文件变化是否符合预期。

---

## 05. 常用指令：计算机性能指标查看（Windows/Linux）

说明：
1. 优先使用低权限可执行的命令。
2. 部分命令可能需要管理员权限。

| 指标类别 | Windows 指令 | Linux 指令 | 用途 | 备注 |
|---|---|---|---|---|
| GPU 概览 | `nvidia-smi` | `nvidia-smi` | 查看 GPU 型号、驱动、显存占用、利用率 | NVIDIA 环境必备 |
| GPU 实时监控 | `nvidia-smi -l 1` | `nvidia-smi -l 1` | 每 1 秒刷新一次 GPU 状态 | 压测时常用 |
| GPU 指标导出 | `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader` | `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader` | 结构化输出核心指标 | 适合写日志 |
| GPU 进程占用 | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` | 查看谁在占用显存 | 定位显存泄漏 |
| CPU 逻辑线程数 | `python -c "import os; print(os.cpu_count())"` | `nproc` | 查看可用逻辑 CPU 数 | 便于设定线程参数 |
| CPU 使用率 | `Get-Counter '\\Processor(_Total)\\% Processor Time'` | `mpstat 1 1` | 查看当前 CPU 总占用 | Linux 需 `sysstat` |
| 内存可用量 | `Get-Counter '\\Memory\\Available MBytes'` | `free -h` | 查看可用内存 | 判断是否内存瓶颈 |
| 进程 CPU Top10 | `Get-Process | Sort-Object CPU -Descending | Select-Object -First 10 ProcessName,CPU,WorkingSet` | `top -b -n 1 -o %CPU` | 找高 CPU 占用进程 | 排查后台干扰 |
| 磁盘容量 | `Get-PSDrive -PSProvider FileSystem` | `df -h` | 查看各盘容量与剩余空间 | 模型/日志落盘前检查 |
| 磁盘吞吐 | `Get-Counter '\\PhysicalDisk(_Total)\\Disk Read Bytes/sec','\\PhysicalDisk(_Total)\\Disk Write Bytes/sec'` | `iostat -dx 1 3` | 观察磁盘读写压力 | Linux 需 `sysstat` |
| PyTorch 设备可用性 | `python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"` | `python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"` | 验证 PyTorch 是否识别 GPU | 训练/推理前先跑 |
| PyTorch 设备信息 | `python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"` | `python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"` | 输出当前 GPU 名称 | 多环境核对 |

常见坑：
1. `nvidia-smi` 正常不代表 PyTorch 一定能用 CUDA，还需要单独检查 `torch.cuda.is_available()`。
2. 后台进程占用 GPU/CPU 会污染 benchmark 结果，测前先清场。

---

## 06. 条目模板（后续追加）

````markdown
## XX. 主题名称

是什么：
- 

什么时候用：
1. 
2. 

最小示例：
```text
...
```

常见坑：
1. 
2. 

实践建议：
1. 
````

