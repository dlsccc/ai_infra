# Week 00 - 初始化与基线

## 本周目标
- 建立可复现的开发与实验目录结构
- 产出 benchmark 模板与周报模板
- 打通最小实验流程（配置 -> 运行 -> 结果记录 -> 周报）

## Day-by-day 任务

### Day 1
- [x] 创建目录结构：`docs/`、`scripts/`、`experiments/`、`src/`、`tests/`
- [x] 检查 Git 状态并整理初始化结构（提交可与后续改动合并）

### Day 2
- [x] 本地 Python 环境检查
- [x] 写依赖清单：`requirements-dev.txt`

### Day 3
- [x] 准备 `experiments/configs/example_benchmark.yaml`
- [x] 准备 `scripts/benchmark/run_example.sh`

### Day 4
- [x] 固化指标定义：P50/P95、throughput、peak memory
- [x] 在 `docs/reports/benchmark_template.md` 中写清字段解释

### Day 5
- [x] 用模板填一版假数据（验证流程完整）：`docs/reports/week00_dryrun.md`
- [x] 记录遇到的问题和下一步动作

### Day 6-7
- [x] 复盘并补齐遗漏
- [x] 输出 `Week 00` 结论

## 本周产出检查
- [x] `docs/weekly/week_template.md`
- [x] `docs/reports/benchmark_template.md`
- [x] `experiments/configs/example_benchmark.yaml`
- [x] `scripts/benchmark/run_example.sh`
- [x] `src/benchmark/run.py`
- [x] `docs/reports/week00_env_check.md`
- [x] `docs/reports/week00_dryrun.md`

## 本周结论
1. 最小实验链路已打通：YAML 配置 -> Python 脚本 -> 结构化输出。
2. 当前结果是 dry-run 指标，仅用于验证工程流程，不用于性能判断。
3. Week 01 可以直接切换为真实测量，并沿用同一份报告模板。

## 下周计划
- 进入 Week 01：推理基础 + PyTorch 推理链路
- 目标：产出第一份“真实测量”基线报告（非 dry-run）
