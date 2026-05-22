### 1. 
Agent 场景下，如何通过 prompt layout / context management / routing / KV cache policy 提升 prefix/KV reuse，并在不显著损害任务效果的情况下降低 TTFT/JCT。

| 方向 | 你关心的问题 |	代表工作 |
| --- | --- | --- |
| LLM Serving / KV Cache 基础 |	KV cache 为什么重要，prefix cache 怎么省 prefill | vLLM/PagedAttention、vLLM APC |
| Prefix/KV Reuse for Agent/RAG | 多轮/Agent/RAG 里怎么复用 KV | SGLang/RadixAttention、KVFlow、LMCache、CacheBlend |
| Cache-aware Routing |	请求怎么路由到有缓存的实例 | NVIDIA Dynamo、vLLM Production Stack、llm-d |
| Context Management / Compression | 超上下文时如何保留信息又降成本 | MemGPT、LLMLingua、LongLLMLingua、Selective Context |

### 2.
1. Serving 和 prefix cache 基础
[1] https://arxiv.org/abs/2309.06180
VLLM
重点：PagedAttention、KV cache block 管理、serving 吞吐。
你要引用它解释：LLM serving 中 KV cache 是核心资源。
[2] https://docs.vllm.ai/en/v0.9.2/features/automatic_prefix_caching.html
重点：新请求如果和已有请求共享相同 prefix，就能复用 KV cache，跳过共享部分计算。
这正是你“prompt 标准化为什么有用”的直接依据。注意：这里的标准化是你的工程推论，因为 exact/shared prefix 更容易命中。
[3] https://arxiv.org/abs/2312.07104
SGLang
重点：RadixAttention 用前缀树复用 KV cache，特别适合程序化、多调用、结构化 LLM workflow。
这和 Agent 很贴。

2. Agent/RAG 的 KV cache 复用
[1] https://arxiv.org/abs/2507.07400
KVFlow 
重点：multi-agent workflow 里普通 prefix caching 不够，需要 workflow-aware KV cache 管理。
这篇和你的项目最像。
[2] https://arxiv.org/abs/2510.09665
LMCache
重点：把 KV cache 做成 serving 系统里的独立缓存层，支持跨 query / 跨 engine / offload。
适合你理解“生产系统怎么管理 KV cache”。
[3] https://blog.lmcache.ai/2025-03-31-eurosys/
重点：RAG 里很多可复用内容并不总是严格 prefix，传统 prefix cache 不能完全利用。
可以作为你讨论“prefix cache 的局限”的材料。

3. Routing / cache locality
[1] https://docs.nvidia.com/dynamo/archive/0.8.0/router/kv_cache_routing.html
重点：把请求路由到拥有相关 KV cache 的 worker，同时考虑负载均衡。
这能支撑你后面的 session/cache-aware routing simulator。
[2] https://docs.vllm.ai/projects/production-stack/en/latest/use_cases/kv-cache-aware-routing.html
重点：多实例 vLLM 下，router 如何利用 cache locality。
这说明你的 routing 方向是生产系统里真实存在的问题。
[3] https://llm-d.ai/docs/guide/Installation/precise-prefix-cache-aware
重点：基于 vLLM KV events 做更精确的 prefix cache aware routing。
这可以作为工程侧对照。

4. Context management / compression
[1] https://research.memgpt.ai/
重点：把 LLM context 看成有限主存，把长期记忆放到外部存储，按需 paging。
适合你写 context management 的系统视角。
[2] https://arxiv.org/abs/2310.05736
重点：prompt compression，目标是在减少 token 的同时保留语义完整性。
适合压缩策略综述。
[3] https://arxiv.org/abs/2310.06839
重点：长上下文场景下，压缩和信息位置会影响模型对关键内容的感知。
适合讨论“压缩不只是变短，还要保留关键证据”。
[4] https://arxiv.org/abs/2310.06201
重点：识别并裁剪上下文中的冗余内容，提升推理效率。
适合放在“冗余删除/上下文压缩”分类下。

### 3. 关键词
prefix/cache
```
LLM prefix caching
KV cache reuse LLM serving
automatic prefix caching vLLM
RadixAttention SGLang
KV cache aware routing LLM
workflow-aware KV cache agent
multi-agent LLM serving KV cache
```
prompt标准化
```
prompt canonicalization LLM caching
prompt template cache hit rate LLM
structured prompt prefix caching
prompt layout prefix cache
tool schema ordering prefix cache
```
context management
```
LLM context management agent memory
LLM agent memory summarization retrieval
prompt compression LLM
long context compression LLM
conversation memory LLM agents
virtual context management LLM
```
routing
```
KV cache aware routing
prefix cache aware routing
LLM serving request routing cache locality
vLLM production stack KV cache routing
NVIDIA Dynamo KV cache routing
llm-d prefix cache aware routing
```

### 4.
综述方向：
Agent 推理中的 Context Management 与 KV Cache Reuse

背景
Agent 多轮调用导致上下文增长、重复 prefix、工具返回膨胀，推理成本主要体现在 prefill、KV cache 和 JCT。

KV Cache 与 Prefix Caching
讲 vLLM APC、PagedAttention、SGLang RadixAttention。

Agent/RAG 场景下的 Cache Reuse
讲 KVFlow、LMCache、CacheBlend。指出普通 exact-prefix cache 的局限。

Cache-aware Routing
讲 Dynamo、vLLM Production Stack、llm-d。引出 session/prefix routing。

Context Management / Compression
讲 MemGPT、LLMLingua、LongLLMLingua、Selective Context。
分类：sliding window、summary、structured memory、retrieval、prompt compression。

对你项目的启发
你的假设：

stable system/tool prefix 放前面，提升 prefix reuse 机会。
tool schema canonicalization 减少等价 prompt 的字面差异。
dynamic summary 放在 stable prefix 后面，减少对 prefix cache 的破坏。
session-aware routing 提升多轮 Agent 的 cache locality。
synthetic workload + public trace 能验证这些假设。
Gap / 创新空间
现有工作多关注底层 KV cache 或长上下文压缩；你的项目关注 Agent prompt layout/context policy 对 prefix reuse、TTFT/JCT 的影响，并做可复现实验。

|Paper/System|	Problem|	Method|	Metrics|	Relation to My Project	|Gap|
|---|---|---|---|---|---|
|vLLM APC|	shared prefix 重复 prefill|	自动 KV prefix cache|	TTFT/throughput|	解释为什么 stable prefix 有用|	不讨论 prompt layout|
|SGLang|	结构化 LLM program 重复调用	|RadixAttention	|latency/throughput|	Agent workflow 很相关|	需对比 vLLM|
|KVFlow	|multi-agent cache 管理|	workflow-aware prefix cache|	JCT/cache hit|	直接相关|	可复现难度高|
|MemGPT|	context 超窗口|	virtual context/memory tiers|	task success|	context management 视角	|不关注 serving cache|
|LLMLingua	|prompt 太长|	prompt compression	|token/latency/quality	|压缩策略参考|	可能破坏 prefix 稳定|


我不是提出新的 KV cache engine，
而是研究 Agent workload 中 prompt/context policy 对 KV reuse 和 JCT 的影响，
并提出 engine-external 的 canonicalization + layout + routing 策略。

baseline: 原始 prompt layout
variant 1: stable tool ordering
variant 2: prompt canonicalization
variant 3: dynamic summary 放前/放后
variant 4: session-aware routing simulator

指标
input tokens
prefix overlap ratio
TTFT
JCT
P95 latency
cache hit proxy
tool call JSON validity

看能否证明：
相同语义的 Agent prompt，因为 layout/schema 顺序不同导致 prefix reuse 机会下降；
canonicalization/stable layout 能提高 overlap，降低 TTFT/JCT；
但在高动态 observation 或低复用场景下收益下降。
