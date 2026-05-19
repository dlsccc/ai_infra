# FlashAttention

## 1. 这篇论文要解决什么问题

标准 attention 前向可以写成：

\[
O = \text{softmax}(QK^T)V
\]

如果只看单个 head，设：

- `Q, K, V ∈ R^(N x d)`
- `S = QK^T ∈ R^(N x N)`
- `P = softmax(S) ∈ R^(N x N)`

那么标准实现通常会经历：

1. 读 `Q, K`，计算 `S`
2. 把 `S` 写回 HBM
3. 再读 `S`，计算 `P`
4. 把 `P` 写回 HBM
5. 再读 `P, V`，计算 `O`

问题不只是算术复杂度，而是 `S / P` 这两个 `N x N` 中间矩阵会造成很重的 HBM 访问。

论文的核心观点：

- 标准 attention 的瓶颈很大程度上是 **memory IO**，不是纯粹的 FLOPs
- 只要避免 materialize 完整的 `S` 和 `P`，就能显著降低显存占用和 HBM 往返

---

## 2. 核心思路：tiling + online softmax + recomputation

FlashAttention 前向的关键是三件事：

1. `tiling`
   - 不一次性计算完整 `QK^T`
   - 而是把 `Q` 切成 query tiles，把 `K / V` 切成 key/value tiles
   - 每次只在 SRAM / shared memory 中处理一个小块

2. `online softmax`
   - 因为分块计算时，看不到整行 score
   - 需要在每个 query 行上维护行状态 `m / l / O_i`
   - 随着新的 `K/V tile` 到来，在线更新 softmax

3. `recomputation`
   - 前向不保存完整 `S` 和 `P`
   - 反向需要时，再按 tile 重算局部 `S_ij` 和 `P_ij`
   - 用额外计算换显存

---

## 3. Tiling 到底在做什么

设：

- `Q_i` 是一个 query tile，形状 `Br x d`
- `K_j, V_j` 是一个 key/value tile，形状 `Bc x d`

前向每一步处理：

\[
S_{ij} = Q_i K_j^T
\]

这是一个局部 score tile，形状：

\[
Br \times Bc
\]

它只在片上 SRAM 中短暂停留，不会写回 HBM 变成完整的 `S`。

然后基于这个局部 tile 更新每一行的 softmax 状态，最终直接得到 `O_i`。

所以 FlashAttention 的数据流是：

- `Q_i` 尽量留在片上
- `K_j / V_j` 一块一块流式经过
- `S_ij` 只在片上临时存在
- 不写回完整 `S`
- 不写回完整 `P`

---

## 4. Online softmax 的本质

标准 stable softmax 对一行 score `x = [x_1, ..., x_N]` 会写成：

\[
m = \max_j x_j
\]

\[
l = \sum_j e^{x_j - m}
\]

\[
softmax(x)_j = \frac{e^{x_j - m}}{l}
\]

FlashAttention 的难点在于：一开始拿不到整行，只能分块看到这行 score。

因此对每个 query 行维护：

- `m_i`：当前已经看到的最大值
- `l_i`：在当前最大值坐标系下的指数和
- `O_i`：当前累计输出

如果旧状态是：

\[
(m_{old}, l_{old}, O_{old})
\]

新块的局部状态是：

\[
(\hat m, \hat l, \hat O)
\]

则合并时：

\[
m_{new} = \max(m_{old}, \hat m)
\]

\[
l_{new} =
e^{m_{old}-m_{new}} l_{old}
+
e^{\hat m-m_{new}} \hat l
\]

输出的更新也要做同样的重标定：

\[
O_{new} =
\frac{
e^{m_{old}-m_{new}} l_{old} O_{old}
+
e^{\hat m-m_{new}} \hat P V_j
}{
l_{new}
}
\]

这里最关键的理解是：

- **旧状态是在旧最大值坐标系下算的**
- 如果新块出现了更大的最大值，旧状态必须整体乘上一个指数缩放因子

这就是 online softmax 的数学本质。

---

## 5. 前向过程怎么理解

前向对每个 `Q_i`：

1. 初始化行状态
   - `m_i = -inf`
   - `l_i = 0`
   - `O_i = 0`

2. 遍历所有 `K_j / V_j`
   - 计算局部 `S_ij = Q_i K_j^T`
   - 计算当前 tile 的局部最大值和局部 softmax 贡献
   - 用 online softmax 更新 `m_i / l_i / O_i`

3. 全部 `K/V tiles` 处理完后，得到最终 `O_i`

重要的是：

- 前向不保存完整 `S`
- 前向不保存完整 `P`
- 只保存后向需要的摘要状态（如 `m_i / l_i`，有的实现也保存 `lse`）

---

## 6. 反向传播是怎么处理的

前向没有保存完整 `S / P`，所以反向不能像标准 attention 一样直接拿缓存好的 `P` 来算。

反向核心就是：

- **利用前向保存的行级状态**
- **重新按 tile 计算局部 `S_ij` 和 `P_ij`**

标准 attention 反向里有：

\[
dV = P^T dO
\]

\[
dP = dO V^T
\]

softmax 反向：

\[
dS_{ij} = P_{ij}\left(dP_{ij} - \sum_t P_{it}dP_{it}\right)
\]

然后：

\[
dQ = dS K,\quad dK = dS^T Q
\]

FlashAttention 反向按 tile 做：

1. 重算
   \[
   S_{ij} = Q_i K_j^T
   \]

2. 利用前向保存的 `m_i / l_i` 重建局部概率
   \[
   P_{ij} = \exp(S_{ij} - m_i) / l_i
   \]

3. 计算
   \[
   dV_j += P_{ij}^T dO_i
   \]

4. 计算
   \[
   dP_{ij} = dO_i V_j^T
   \]

5. 对 softmax 做反向，得到
   \[
   dS_{ij}
   \]

6. 再累计
   \[
   dQ_i += dS_{ij} K_j
   \]
   \[
   dK_j += dS_{ij}^T Q_i
   \]

所以反向的“重计算”不是把整个前向又跑一遍，而是：

- **只重算局部 `S_ij / P_ij`**
- 不保存完整 `S / P`

---

## 7. 复杂度与内存访问

### 7.1 标准 attention

前向：

1. `S = QK^T`
   \[
   O(N^2 d)
   \]

2. `P = softmax(S)`
   \[
   O(N^2)
   \]

3. `O = PV`
   \[
   O(N^2 d)
   \]

总计算复杂度：

\[
O(N^2 d)
\]

额外中间显存：

- `S ∈ R^(N x N)`
- `P ∈ R^(N x N)`

所以额外中间存储：

\[
O(N^2)
\]

### 7.2 FlashAttention

FlashAttention 仍然要算所有 `q_i · k_j`，也仍然要完成 softmax 和加权求和。

因此总算术复杂度仍然是：

\[
O(N^2 d)
\]

但它不保存完整 `S / P`，只保存：

- 输出 `O`
- 行状态 `m / l`
- 当前 tile 的片上 scratch

因此额外中间存储降为：

\[
O(Nd)
\]

### 7.3 核心区别

- **算术复杂度没有降阶**
- **显存占用从 `O(N^2)` 降到 `O(Nd)`**
- **HBM 访问大幅减少**

所以 FlashAttention 的收益本质上是：

- **IO-aware optimization**
- 不是“数学上减少了 attention 该算的东西”

---

## 8. FA1 / FA2 / FA3 的区别

### FA1

核心贡献：

- IO-aware tiling
- online softmax
- backward recomputation

本质：

- 避免 materialize 完整 `S / P`
- 用片上 SRAM 和重计算降低 HBM 压力

### FA2

核心贡献：

- 更好的 work partitioning 和并行策略
- 降低非 matmul 部分的开销
- 提高 GPU 利用率

理解方式：

- FA1 更像“把 attention 算对、算省内存”
- FA2 更像“把这个 kernel 在 GPU 上调度得更合理、更饱满”

### FA3

核心贡献：

- 面向 Hopper 架构（尤其 H100）的进一步优化
- 更深的异步流水
- 更贴近 Tensor Core / 硬件特性的调度设计

理解方式：

- FA3 不是推翻 FA1/FA2
- 而是在新硬件上把 pipeline 和 tensor core 利用进一步做深

---

## 9. 这周我真正搞懂了什么

1. `K^T` 是为了矩阵乘法维度匹配，不是把单个向量元素倒序。
2. `tiling` 的作用不是减少注意力数学本身，而是避免 `S/P` 落到 HBM。
3. `online softmax` 的关键是旧状态换坐标系时的指数缩放。
4. `m / l / O_i` 是行级摘要状态，不是完整概率矩阵。
5. 反向传播的核心是保存摘要状态、重算局部 `S/P`。
6. FlashAttention 的本质收益是 IO reduction，而不是 FLOPs reduction。

---

## 10. 我后面看代码时要带着的问题

后面看 `vLLM`、`PyTorch SDPA backend` 或 Triton kernel 时，重点带着这些问题：

1. query tile / key tile 的切分粒度怎么定？
2. `m / l` 或 `lse` 是怎么保存的？
3. backward 是如何重算局部 `P_ij` 的？
4. 实现里哪些状态跨 tile 需要写回 HBM，哪些只存在片上？
5. 调度优化到底是在提升 matmul 占比，还是在减少额外标量开销？

---

## 11. 一句话总结

**FlashAttention = 用 tiling 把 attention 改写成片上流式计算，用 online softmax 保证分块计算结果与标准 softmax 完全一致，再用 backward recomputation 把显存占用从 `O(N^2)` 压到 `O(Nd)`。**
