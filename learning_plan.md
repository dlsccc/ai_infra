# 总目标
1. 读懂并修改 vLLM 核心代码
2. 写 CUDA kernel（tiled matmul + 简单 attention）
3. 用 TensorRT 写 plugin
4. 能分析 GPU 性能瓶颈（nsys / nvprof）
5. 自己做一个 mini LLM inference engine

# 阶段1:1-4周 推理系统认知建立
**第 1 周：ONNX + 推理流程全链路**  
**Day 1–2**  
PyTorch 模型导出 ONNX  
理解 graph structure  
用 Netron 可视化  
输出：  
画出模型 graph 流程图  
**Day 3–4**  
ONNX → TensorRT engine  
跑 FP32 / FP16  
比较 latency  
输出：  
一份性能对比表  
**Day 5–7**  
读 TensorRT engine 构建流程：  
重点理解：  
layer fusion  
tactic selection  
kernel autotune  
你要能回答：  
TensorRT 为什么比 PyTorch inference 快？  
**第 2 周：LLM 推理框架精读**  
读vLLM  
**Day 1–3**  
paged attention  
block manager  
必须搞懂：  
为什么不用连续内存？  
KV cache 如何分页？  
**Day 4–7**  
读 scheduler：  
continuous batching  
request queue  
输出：  
画出一次 token 生成的流程图  
**第 3 周：KV cache + Attention 深入**  
FlashAttention  
重点理解：  
为什么减少 HBM 访问？  
为什么 tiled 算法更快？  
任务：  
手写一个 naive attention  
再写一个分块版本（不一定 CUDA）  
**第 4 周：量化**  
INT8 量化流程  
scale / zero point  
per-channel vs per-tensor  
任务：  
用 PTQ 跑一个 7B 模型  
对比显存和速度  
# 阶段2:5-10周 正式CUDA训练
**第 5 周：GPU 架构**  
SM  
warp（32 线程）  
shared memory  
occupancy  
tensor core  
输出：  
画 GPU 结构图  
写一篇“GPU 执行模型笔记”  
**第 6 周：CUDA 基础编程**  
vector add  
reduce sum  
scan  
重点：  
blockDim  
gridDim  
memory coalescing  
**第 7–8 周：Matmul 优化**  
naive matmul  
shared memory tiling  
double buffering  
减少 bank conflict  
目标：  
性能接近cuBLAS70% 即合格。  
**第 9–10 周：性能分析**  
nsys  
roofline model  
memory bound vs compute bound  
任务：  
分析你自己的 matmul：  
回答：  
SM 利用率多少？  
global memory 吞吐多少？  
为什么不是 100%？  
# 阶段3：11-16周 推理级kernel
**第 11–12 周：写 Attention CUDA Kernel**  
QK matmul  
softmax  
V 乘法  
KV cache 支持  
**第 13–14 周：融合优化**  
matmul + bias + activation 融合  
减少 kernel launch  
**第 15–16 周：TensorRT plugin**  
自定义 layer  
注册 plugin  
跑 ONNX → TensorRT  
# 阶段4:17-24周 打造项目
**mini-llm-inference-engine**  
要求：  
支持 batching  
KV cache  
CUDA attention  
FP16  
简单 INT8  
benchmark 输出  
GitHub 必须可展示  
