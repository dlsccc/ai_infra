# 阶段1:1-6周-推理全栈认知
**week1:ONNX + TensorRT**  
Day 1–2  
PyTorch 模型导出 ONNX  
理解 graph structure  
用 Netron 可视化  
输出：  
画出模型 graph 流程图  
Day 3–4  
ONNX → TensorRT engine  
跑 FP32 / FP16  
比较 latency  
输出：  
一份性能对比表  
Day 5–7  
读 TensorRT engine 构建流程：  
重点理解：  
layer fusion  
tactic selection  
kernel autotune  
能回答：  
TensorRT 为什么比 PyTorch inference 快？  
完成性能对比实验:    
```
models = ["ResNet50","BERT-base","Llama-7B"]
frameworks = ["PyTorch","ONNX Runtime","TensorRT"]
```
**week2:vLLM**  
Day 1–3  
paged attention  
block manager  
必须搞懂：  
为什么不用连续内存？  
KV cache 如何分页？  
Day 4–7  
读 scheduler：  
continuous batching  
request queue  
prefix caching原理  
输出：  
画出一次 token 生成的流程图  
画出完整的request生命周期图    
写技术博客《vLLM如何做到2x吞吐提升》  
**week3-4:KV Cache + Attention**  
FlashAttention  
重点理解：  
为什么减少 HBM 访问？  
为什么 tiled 算法更快？    
进行对比实验：    
```
import torch
from flash_attn import flash_attn_func
#测试不同sequence_length下的性能
seq_lens = [512,1024,2048,4096]
#输出memory占用+latency曲线图
```
输出：  
- 阅读flashattention论文并做presentation  
- 分析flashattention-2改进点

**week5:量化**  
INT8 量化流程  
scale / zero point  
per-channel vs per-tensor  
任务：
用 PTQ 跑一个 7B 模型  
对比显存和速度  
AWQ/GPTQ量化方法对比  
实测llama2-7B在不同量化下的准确率  
**week6:分布式推理**  
```
from vllm import LLM
#1.单卡 vs Tensor Parallel性能对比
#2.理解NCCL ALLReduce开销
#3.计算通信/计算比
```
输出：  
- 技术报告《大模型推理性能优化checklist》  
- 回答2A100 vs 1H100该选哪个

# 阶段2:7-10周-CUDA核心训练  
**week7:GPU架构+基础编程**  
**DAY1-3:GPU架构认知**  
SM  
warp（32 线程）  
shared memory  
occupancy  
tensor core  
NVIDIA GPU架构白皮书精读  
输出：  
画 GPU 结构图  
写一篇“GPU 执行模型笔记”  
**DAY4-7:基础kernel**  
```
必须手写3个kernel
1. vector_add(理解grid/block/thread)
2. reduce_sum(理解warp shuffle)
3. softmax(理解shared memory)
```
- 能解释每行代码的memory访问模式
- 能用nsys分析瓶颈

**week8-9:Matmul优化（关键）**  
naive matmul  
shared memory tiling  
double buffering  
减少 bank conflict  
目标：  
性能达到naive版本的10x即可
学习路径：
1. naive实现（global memory only）
2. shared memory tiling(1D blocking)
3. 2D blocking + double buffering
时间分配：
- Day1-7：写代码
- Day8-10：性能分析
```
nsys profile -o report ./matmul
#分析：SM occupancy、memory throughput、warp stall
```
- Day11-14：技术博客《如何优化CUDA Matmul》

**week10:Attention Kernel**  
阅读FlashAttention源码，写简化版forward
只支持固定shape，不做动态优化
# 阶段3:11-16周-系统设计+工程实战
**week11-12:推理服务系统设计**  
1. Triton Inference Server
```
# 部署一个支持dynamic batching的模型
# 理解ensemble模型调用
# 优化并发请求处理
```
2. Ray Serve(分布式部署)
```
from ray import serve
# 多副本部署
# GPU资源调度
# 请求路由策略
```
核心任务：  
- 设计一个支持1000QPS的推理服务架构
- 画出完整的数据流图（从http请求到gpu计算）

能回答：
- 如何处理流量突增？
- 多模型共享GPU调度策略

**week13-14:TensorRT Plugin开发**  
自定义 layer  
注册 plugin  
跑 ONNX → TensorRT  
做一个真实场景：
  1. 为某个特殊算子写plugin(如RoPE、RMSNorm)
  2. 测试end-to-end性能提升

输出：  
完整的plugin代码，benchmark报告  
**week15-16:性能调优实战**  
优化一个开源模型的推理性能  
模型选择：  
chatGML3-6B、Qwen-7B  
优化方向：  
1. 模型转换（Pytorch -- ONNX -- TensorRT）
2. 量化（FP16/INT8）
3. 算子融合
4. KV cache优化

目标：
1. first token latency < 100ms
2. throughput > 50tokens/s/user
3. 写一份完整优化报告

# 阶段4:17-24周-项目
### 项目1：vLLM核心功能增强（week17-20）
**optionA:Prefix Caching优化**
```
#为vLLm实现更高效的prefix cache
#场景：RAG应用中system prompt重复率高
#目标：减少30%重复计算
```
**optionB:Multi-LoRA支持改进**
```
#优化vLLM的LoRA切换延迟
#目标：支持100+个LoRA动态加载
```
**optionC:新型量化算法集成**
```
#集成AWQ/SqueezeLLM到vLLM
#提供完整的精度评测
```
输出：  
pull Request到vLLM官方仓库  
输出技术博客，说明设计思路  
### 项目2：推理性能分析工具（week21-22）
```
# llm_profiler.py
class LLMProfiler:
    def profile(self, model_path, batch_sizes, seq_lengths):
        """
        自动测试：
        - Prefill latency
        - Decode throughput  
        - Memory usage
        - GPU utilization
        生成报告（HTML+图表）
        """
        pass
```
技术栈：  
nsys/nvprof集成  
数据可视化（Plotly）  
支持多种推理框架  
### 项目3：分布式推理系统（week23-24）
支持Ray的多机多卡推理服务  
```
1. 支持70B模型跨4卡部署（Tensor Parallel）
2. 多个7B模型并行服务（Model Parallel）
3. 自动failover和负载均衡
```
技术要点：  
NCCL通信优化  
GPU拓扑感知调度  
请求队列优化  

# 简历项目描述
### 项目1：vLLM Prefix Caching优化
```
- 为vLLM实现细粒度prefix cache, 减少RAG场景重复计算30%
- 设计LRU替换策略，支持10000+不同prefix开发
- 提交PR到官方仓库，获得120+ stars
技术栈：Python, CUDA, Pytorch, profiling(nsys)
```
### 项目2：大模型推理性能自动化分析工具
```
- 开发端到端prefiling工具，支持vLLM/TensorRT/ONNX Runtime
- 自动生成性能报告（latency P50/p99, gpu利用率，memory）
- 帮助团队将平均推理延迟从150ms降至80ms
```
