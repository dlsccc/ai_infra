# Agent Workload Characterization

## 1. 文档目的

这份文档用于回答 Week 6 Day 3 的核心问题：

1. 当前 `AgentInferBench` 中的 Agent workload 是如何构造的。
2. 每一轮请求的输入文本结构是什么。
3. 从现有数据出发，为什么 Agent 推理不是普通 chat 的简单重复。
4. 基于这些数据，可以提出哪些后续可验证的优化假设。

本文基于以下实验结果：

- `docs/reports/week06_plain_baseline.md`
- `docs/reports/week06_agent_baseline.md`

## 2. 当前 Agent workload 的构造方式

当前 workload 生成逻辑位于：

- `agent_bench/workloads/generator.py`
- `agent_bench/workloads/schemas.py`

### 2.1 通用构造原则

所有 Agent-style workload 都遵循同一套基本拼接模式：

1. 固定 system prompt
2. 固定 tool schema 列表
3. 逐轮累积的 history
4. 当前轮 user turn
5. 统一的 `Assistant:` 结尾

也就是说，当前每一轮请求并不是只发送“这一轮新增内容”，而是把：

- system
- tools
- 历史 observation
- 历史 assistant tool call
- 当前 user turn

全部拼成一个完整 prompt 再送给服务端。

这正是 Agent workload 与普通单轮 chat 的根本区别之一。

### 2.2 tool schema 是如何生成的

tool 定义来自 `make_tools(tool_count, tool_tokens)`，每个 tool 的结构类似：

```json
{
  "name": "tool_00",
  "description": "... filler text ...",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string"
      }
    }
  }
}
```

当前实现里，tool schema 的特点是：

1. tool 数量固定。
2. 每个 tool 的 description 用 filler text 填充。
3. tool schema 在同一个 trace 内保持稳定顺序。

这意味着 Agent workload 的第一轮就会携带一段不短的固定前缀。

### 2.3 history 是如何累积的

在 `generator.py` 中，每一轮结束后都会把一条固定格式的 assistant tool call 写入 history：

```text
Assistant tool call: {"name": "tool_00", "arguments": {"query": "example"}}
```

从第二轮开始，如果该 workload 配置了 `observation_tokens > 0`，还会把上一轮 observation 加入 history：

```text
Observation turn X: ...
```

因此随着 turn 增长，history 会逐轮变长。

## 3. 三类 Agent workload 的具体定义

当前 Week 6 Agent baseline 配置为：

| Workload | system tokens | tool count | tool tokens | turns | observation tokens |
|---|---:|---:|---:|---:|---:|
| `single_tool` | 64 | 2 | 16 | 2 | 0 |
| `multi_tool_serial` | 64 | 2 | 16 | 3 | 32 |
| `long_observation` | 64 | 2 | 16 | 3 | 96 |

### 3.1 `single_tool`

目标：

- 模拟一次简单工具调用
- 观察固定 system/tool 前缀本身的成本

特点：

1. 只有 2 轮。
2. 没有长 observation。
3. history 增长非常有限。

它更像“Agent 形式的最小可用请求”，适合测：

- 固定前缀成本
- 第一轮和第二轮的差异

### 3.2 `multi_tool_serial`

目标：

- 模拟多轮串行工具调用
- 观察 history 累积带来的输入增长

特点：

1. 有 3 轮。
2. 每一轮 observation 比较短。
3. 每轮 history 中都会多出上一轮 observation 和 assistant tool call。

它更像“多轮决策链逐渐变长”的 workload。

### 3.3 `long_observation`

目标：

- 模拟工具返回内容较长的 Agent 场景
- 观察 prefill-heavy 压力

特点：

1. 同样是 3 轮。
2. observation 比 `multi_tool_serial` 更长。
3. 因此 prompt 膨胀速度最快。

这类 workload 最能体现：

- 长 observation 如何推高输入 token
- 为什么 Agent workload 会迅速从“可控多轮”变成“上下文膨胀问题”

## 4. 每轮请求输入的文本结构

### 4.1 `plain_chat` 的结构

普通 chat 的输入非常简单：

```text
You are a helpful assistant.
User request: ...
```

它没有：

- tool schema
- assistant tool call history
- observation history

所以它更接近普通单轮对话。

### 4.2 Agent workload 的通用结构

当前 Agent workload 的每一轮请求，可以抽象成：

```text
System:
<system prompt filler>

Tools:
<tool_00 schema>
<tool_01 schema>
...

<history so far>
User turn N: <current user text>
Assistant:
```

其中 `<history so far>` 在不同 turn 下会逐渐变成：

```text
Assistant tool call: {...}
Observation turn 0: ...
Assistant tool call: {...}
Observation turn 1: ...
...
```

### 4.3 `single_tool` 的轮次结构

#### Turn 0

```text
System:
...

Tools:
...

User turn 0: ...
Assistant:
```

#### Turn 1

```text
System:
...

Tools:
...

Assistant tool call: {"name": "tool_00", "arguments": {"query": "example"}}
User turn 1: ...
Assistant:
```

它的增长主要来自：

- 多出一条 assistant tool call

### 4.4 `multi_tool_serial` 的轮次结构

#### Turn 0

```text
System:
...

Tools:
...

User turn 0: ...
Assistant:
```

#### Turn 1

```text
System:
...

Tools:
...

Assistant tool call: {...}
Observation turn 0: ...
User turn 1: ...
Assistant:
```

#### Turn 2

```text
System:
...

Tools:
...

Assistant tool call: {...}
Observation turn 0: ...
Assistant tool call: {...}
Observation turn 1: ...
User turn 2: ...
Assistant:
```

它的增长主要来自：

- tool call history 逐轮增加
- observation 逐轮累积

### 4.5 `long_observation` 的轮次结构

文本结构与 `multi_tool_serial` 相同，但 `Observation turn X` 更长。

所以它与 `multi_tool_serial` 的主要差别不是模板，而是：

- 同样的 history 框架下，observation 占比更高
- 因此 prompt 膨胀更快

## 5. 用现有数据回答：为什么 Agent 推理不是普通 chat 的简单重复

结论可以直接写成：

**Agent 推理不是普通 chat 的简单重复，因为它的负载结构发生了变化，而不只是请求次数增加。**

这一点可以用当前数据中的 3 条证据支撑。

### 5.1 证据一：Agent 的固定前缀成本更高

在 plain baseline 中，普通 chat 的真实输入 token 大致为：

- `319`
- `1435`
- `2971`

在 Agent baseline 中，仅第一轮就已经达到：

- `single_tool` turn 0: `845`
- `multi_tool_serial` turn 0: `845`
- `long_observation` turn 0: `845`

这说明 Agent workload 即使还没有进入真正的多轮累积阶段，仅仅是：

- system prompt
- tool schema
- Agent 模板

就已经带来了显著更高的固定前缀成本。

因此它不是普通 chat 的“同样请求多发几次”，而是一开始就拥有不同的输入结构。

### 5.2 证据二：Agent 的输入会按 turn 持续累积，plain chat 不会

当前 Agent baseline 的输入增长曲线非常清楚：

- `single_tool`: `845 -> 874`
- `multi_tool_serial`: `845 -> 1056 -> 1267`
- `long_observation`: `845 -> 1440 -> 2035`

其中：

- `single_tool` 的增长最小
- `multi_tool_serial` 呈稳定增长
- `long_observation` 增长最快

这说明 Agent workload 的时延问题来自逐轮扩张的上下文，而不是同一类单轮 chat 的重复执行。

### 5.3 证据三：Agent 同时具有“前缀复用机会”和“上下文膨胀压力”

多轮 workload 的 prefix overlap ratio 会逐轮上升，例如：

- `multi_tool_serial`: `0.0000 -> 0.5540 -> 0.6289`
- `long_observation`: `0.0000 -> 0.4260 -> 0.5859`

这说明 Agent 确实存在前缀复用机会。

但与此同时，输入 token 依然显著增长，尤其是 `long_observation`。

这意味着 Agent workload 的本质不是：

- “请求重复，所以缓存自然生效”

而是：

- “虽然有重复前缀，但 history 与 observation 同时在膨胀上下文”

因此 Agent 推理是一个同时包含：

1. 高固定前缀
2. 多轮 history 累积
3. observation 膨胀
4. prefix reuse 机会

的复合负载，而不是 plain chat 的简单重复。

## 6. 从数据中提炼出的 3 个可验证优化假设

### 假设一：`long_observation` 是最值得优先优化的场景

理由：

- 它的输入增长最快
- 总时延也随 turn 快速升高
- 它最能代表真实 Agent 工具返回内容过长时的压力

可验证方式：

1. 固定 backend 与 turns。
2. 仅调整 `observation_tokens`。
3. 观察 `input_tokens / TTFT / total_latency / TPOT` 的变化。

如果 observation 长度下降能显著改善总时延，那么说明 observation 管理是高优先级优化方向。

### 假设二：稳定的 prompt layout 与 tool ordering 会提升多轮复用价值

理由：

- 当前多轮 workload 已经出现明显 prefix overlap
- 如果 prompt 布局和 tool schema 顺序保持稳定，理论上更有利于 prefix cache 复用

可验证方式：

1. 对比 stable tool ordering 与 random ordering。
2. 对比固定 prompt layout 与动态 layout。
3. 观察 `prefix_overlap_ratio`、`TTFT`、`total_latency` 是否更稳定。

这对应后续的 canonicalization / stable tool ordering 方向。

### 假设三：prefix cache 对 prefill-heavy workload 的收益会显著高于普通 workload

理由：

- 当前 plain baseline 中 TTFT 增长相对平缓
- `long_observation` 与多轮 Agent workload 更接近 prefill-heavy 场景
- 因此 prefix cache 的收益很可能集中在长前缀、多轮累积场景，而不是所有请求都平均受益

可验证方式：

1. 在 vLLM 中做 prefix cache 开/关实验。
2. 对比 `single_tool / multi_tool_serial / long_observation / plain_chat`。
3. 观察 `TTFT / total_latency / 首轮与后续轮差异` 的变化。

这正对应 Week 6 Day 4 的实验目标。

## 7. 当前阶段结论

基于现有 Week 6 结果，可以给出一个比较清晰的 characterization 结论：

1. Agent workload 在第一轮就已经具有高于普通 chat 的固定前缀成本。
2. 随着 turn 增长，history 和 observation 会推动输入上下文持续膨胀。
3. `long_observation` 是最能代表 Agent 推理特殊性的 workload。
4. Agent workload 同时具有 prefix reuse 机会与 prompt 膨胀压力，因此它不是普通 chat 的简单重复，而是不同的系统负载形态。
5. 当前数据已经足够支持进入后续 prefix cache 开关实验，并为 canonicalization、stable tool ordering、context management 提供了明确动机。
