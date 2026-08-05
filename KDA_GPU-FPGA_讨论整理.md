# KDA：算法原理、Kimi K3 实现与 GPU–FPGA 协同加速分析

> 本文整理我们围绕 Kimi Delta Attention（KDA）的讨论，重点回答三个问题：KDA 在算什么、为什么它在 GPU 上可能受限，以及把它放到 FPGA 时，真正合理的系统边界在哪里。

## 目录

1. [核心结论](#1-核心结论)
2. [从线性注意力到 KDA](#2-从线性注意力到-kda)
3. [KDA 的完整计算流程](#3-kda-的完整计算流程)
4. [Kimi K3 中的 KDA 配置](#4-kimi-k3-中的-kda-配置)
5. [Prefill 与 Decode 的执行差异](#5-prefill-与-decode-的执行差异)
6. [为什么 KDA Decode 在 GPU 上容易受限](#6-为什么-kda-decode-在-gpu-上容易受限)
7. [FPGA 为什么可能适合 KDA](#7-fpga-为什么可能适合-kda)
8. [已有 GDN FPGA 工作及其启示](#8-已有-gdn-fpga-工作及其启示)
9. [GPU–FPGA 切分方案比较](#9-gpufpga-切分方案比较)
10. [MTP 验证与 KDA State Replay](#10-mtp-验证与-kda-state-replay)
11. [KDA Context Parallelism（KCP）](#11-kda-context-parallelismkcp)
12. [FPGA SmartNIC 执行 KCP Scan](#12-fpga-smartnic-执行-kcp-scan)
13. [研究方向与实验路线](#13-研究方向与实验路线)
14. [最终判断](#14-最终判断)
15. [参考资料](#15-参考资料)

---

## 1. 核心结论

KDA 比普通 LLM GEMV 更有理由考虑 FPGA 加速，因为它的特殊性不在于“也是 memory-bound”，而在于：

1. 每个 head 都维护一个固定大小的递归状态矩阵；
2. 该状态跨 token 持续存在；
3. GPU 在每个 decode step 中通常要从 HBM 读出状态、更新后再写回；
4. FPGA 可以将热状态长期保存在 BRAM/URAM 中，避免每个 token 的 HBM 往返；
5. KDA 的状态更新尺寸固定、控制规则、易于构造流式流水线。

但“算子适合 FPGA”不等于“系统上适合从 GPU 中切出来”。Kimi K3 有 69 个 KDA 层，如果 GPU 每经过一个 KDA 层就同步调用一次独立 FPGA，将产生每 token 69 次设备往返，并拆散 GPU 已融合的投影、卷积、归一化和门控 kernel。同步开销可能吞掉 FPGA 的局部加速收益。

因此，讨论后的判断是：

- **微架构层面最直接的方向**：支持多层、多请求和 MTP replay 的 persistent KDA state engine；
- **系统边界最干净但风险更高的方向**：让 FPGA SmartNIC 在原有 collective 路径中执行 KCP fragment scan；
- **不推荐**：把完整 KDA 层连同大规模 Q/K/V/O 投影放到 PCIe FPGA；
- **决定项目是否成立的第一步**：先 profile 真实 K3 serving，确认 KDA recurrence 或 KCP merge 确实处于关键路径。

---

## 2. 从线性注意力到 KDA

### 2.1 普通线性注意力

最简单的线性注意力用一个固定大小的矩阵保存历史：

\[
S_t=S_{t-1}+k_tv_t^{\mathsf T}
\]

\[
o_t=S_t^{\mathsf T}q_t
\]

其中：

- \(q_t,k_t\in\mathbb R^{d_k}\)；
- \(v_t,o_t\in\mathbb R^{d_v}\)；
- \(S_t\in\mathbb R^{d_k\times d_v}\)。

状态大小不随序列长度增长，但旧信息只会不断累加，缺少主动删除或修改已有关联的能力。

### 2.2 DeltaNet：用预测误差修改记忆

Delta rule 先从旧状态中查询当前 key 已经关联的 value：

\[
\hat v_t=S_{t-1}^{\mathsf T}k_t
\]

再计算需要写入的修正量：

\[
\Delta v_t=\beta_t(v_t-\hat v_t)
\]

最后更新状态：

\[
S_t=S_{t-1}+k_t\Delta v_t^{\mathsf T}
\]

这不是盲目加入 \(k_tv_t^{\mathsf T}\)，而是根据“目标 value 与旧记忆预测值之间的误差”修改关联。

这种方法其实就是在假设：
\[
S^{\mathsf T}k_j=
\underbrace{|k_j|^2v_j}_{\text{目标项}}+
\underbrace{\sum_{i\neq j}v_i(k_i^{\mathsf T}k_j)}_{\text{其他 key 的干扰}}
\]
其中的“其他 key 的干扰”的数值很小，k进行了归一化。这就是迫使模型学习哪些 token 应产生相似的 key，从而共享信息，哪些信息需要通过不同方向的 key 分开存储，但这终究会受到矩阵容量的限制，如果信息过多，仍然会出现多个key高度相关，数值上被反复抵消，导致模型无法记住所有信息。
### 2.3 GDN：增加每 head 的标量遗忘门

Gated DeltaNet（GDN）加入标量衰减 \(\alpha_t\)：

\[
S_t=
\alpha_t
\left(I-\beta_tk_tk_t^{\mathsf T}\right)S_{t-1}
+\beta_tk_tv_t^{\mathsf T}
\]

这里一个 head 在一个 token 上只有一个 \(\alpha_t\)，所以整个状态矩阵使用相同衰减系数。

### 2.4 KDA：将标量门细化为逐通道门

KDA 将 \(\alpha_t\) 从标量扩展为向量：

\[
\alpha_t\in[0,1]^{d_k}
\]

其状态更新为：

\[
\boxed{
S_t=
\left(I-\beta_tk_tk_t^{\mathsf T}\right)
\operatorname{Diag}(\alpha_t)S_{t-1}
+\beta_tk_tv_t^{\mathsf T}
}
\]

输出为：

\[
\boxed{o_t=S_t^{\mathsf T}q_t}
\]

KDA 的关键改进是：不同 key 通道可以采用不同遗忘强度，比 GDN 的“整个 head 一起忘记”更细粒度。

### 2.5 四种机制的关系

| 机制 | 状态更新特点 | 遗忘/修改能力 |
|---|---|---|
| 简单线性注意力 | 直接累加 \(kv^{\mathsf T}\) | 无主动遗忘 |
| DeltaNet | 按预测误差修正已有 key–value 关联 | 可修改关联 |
| GDN | Delta rule + 每 head 标量衰减 | 整个 head 统一遗忘 |
| KDA | Delta rule + 每 key 通道衰减 | 逐通道精细遗忘 |

---

## 3. KDA 的完整计算流程

### 3.1 `Diag(α)` 如何构造

若：

\[
\alpha=
\begin{bmatrix}
0.9\\0.5\\0.1
\end{bmatrix}
\]

则：

\[
\operatorname{Diag}(\alpha)=
\begin{bmatrix}
0.9&0&0\\
0&0.5&0\\
0&0&0.1
\end{bmatrix}
\]

对于状态矩阵 \(S\)：

\[
\operatorname{Diag}(\alpha)S
\]

等价于用 \(\alpha_i\) 缩放 \(S\) 的第 \(i\) 行。因此实现时不会真的构造一个包含大量零的对角矩阵，而是直接执行：

\[
S_{ij}\leftarrow\alpha_iS_{ij}
\]

软件中可写为：

```python
S_decay = alpha[:, None] * S
```

FPGA 中可以在读取第 \(i\) 行状态时，将 \(\alpha_i\) 广播给该行的全部元素。

### 3.2 等价的逐步状态更新

KDA 公式可以改写为更适合硬件理解的形式。

先衰减旧状态：

\[
\widetilde S_t=\operatorname{Diag}(\alpha_t)S_{t-1}
\]

再查询旧关联：

\[
\hat v_t=\widetilde S_t^{\mathsf T}k_t
\]

计算 delta correction：

\[
\Delta v_t=\beta_t(v_t-\hat v_t)
\]

更新状态：

\[
S_t=\widetilde S_t+k_t\Delta v_t^{\mathsf T}
\]

计算输出：

\[
o_t=S_t^{\mathsf T}q_t
\]

这一形式清楚展示了状态访问：至少需要围绕 \(S\) 完成衰减、查询、写回和输出计算。高效实现会通过代数变换与数据流融合减少完整状态遍历次数。

### 3.3 ShortConv

KDA 的 Q/K/V 路径在投影后还会沿 token 维执行短窗口因果一维卷积。对第 \(c\) 个通道：

\[
\bar y_{t,c}=
\sum_{r=0}^{K-1}w_{c,r}y_{t-r,c}+b_c
\]

它只读取当前和过去的 token，不读取未来 token，因此是 causal convolution。

Kimi K3 官方配置给出：

```json
"short_conv_kernel_size": 4
```

所以 decode 时除 KDA 矩阵状态外，还要为卷积保存最近 \(K-1=3\) 个位置的投影结果。ShortConv 负责精确的局部模式，KDA 状态负责压缩长期历史，而周期性的 MLA 层保留全局精确访问能力。

### 3.4 Swish / SiLU

Swish 定义为：

\[
\operatorname{Swish}(x)
=x\operatorname{Sigmoid}(x)
=\frac{x}{1+e^{-x}}
\]

它通常也称为 SiLU。与 ReLU 不同，它不会把所有负数直接截为零，并且处处可微。

KDA 中可将 Q/K/V 的前处理概括为：

\[
\text{Linear projection}
\rightarrow
\text{ShortConv}
\rightarrow
\text{Swish}
\rightarrow
\text{Q/K L2Norm}
\]

K3 还在 recurrent output 后使用 head-wise RMSNorm 和数据依赖的 output gate，最后再通过输出投影回到 residual hidden size。

---

## 4. Kimi K3 中的 KDA 配置

### 4.1 文本主干结构

Kimi K3 文本主干配置为：

| 参数 | 数值 |
|---|---:|
| Hidden size | 7168 |
| Decoder 层数 | 93 |
| KDA 层数 | 69 |
| Gated MLA 层数 | 24 |
| KDA heads / 层 | 96 |
| KDA head dimension | 128 |
| ShortConv kernel size | 4 |

层排列方式为：

\[
23\times(3\text{ KDA}+1\text{ Gated MLA})
+1\text{ Gated MLA}
\]

最后额外放置一个 Gated MLA 层，使文本主干以全局注意力结束。

### 4.2 “每层都有 96 个 head”应如何理解

文本主干每个序列混合层都配置了 96 个 head，但两种层的 head 含义不同：

| 层类型 | 96 个 head 的含义 |
|---|---|
| KDA 层 | 96 个独立 KDA recurrent heads，每个维护一个 \(128\times128\) 状态 |
| Gated MLA 层 | 96 个 Query/输出 heads，通过 MLA 共享压缩后的 KV latent |

MLA 的 96 个 head 不等价于传统 MHA 中 96 套独立 KV cache。

另外：

\[
96\times128=12288\neq7168
\]

并不矛盾。7168 是 residual stream 的 hidden size；KDA 内部可以投影到 96 个 128 维 head，随后再通过输出投影映射回 7168 维。

### 4.3 KDA 状态大小

单个 head 的状态为：

\[
S^{(h)}\in\mathbb R^{128\times128}
\]

BF16 下单 head 状态大小：

\[
128\times128\times2=32768\text{ bytes}=32\text{ KiB}
\]

一个 KDA 层的 96 个状态：

\[
96\times32\text{ KiB}=3\text{ MiB}
\]

69 个 KDA 层的全局状态：

\[
69\times3\text{ MiB}=207\text{ MiB/request}
\]

若采用 TP16，每个 TP rank 负责：

\[
96/16=6\text{ heads}
\]

于是每个 rank 的状态为：

\[
6\times128\times128\times2=192\text{ KiB/layer}
\]

\[
69\times192\text{ KiB}\approx12.94\text{ MiB/request/rank}
\]

以上仅计算 BF16 recurrent state，未计 ShortConv state、replay log、双缓冲、地址表和中间结果。若关键状态使用 FP32，容量还会翻倍。

### 4.4 TP rank 与 CP rank 不同

| 并行方式 | 切分维度 | rank 的含义 |
|---|---|---|
| Tensor Parallel（TP） | head 或矩阵维度 | 每个 rank 只处理部分 heads/权重 |
| Context Parallel（CP） | 序列长度 | 每个 rank 只处理一个 token 片段 |
| Expert Parallel（EP） | MoE experts | 每个 rank 保存和执行部分专家 |

例如 \(TP16\times CP8\) 表示：每个 TP rank 负责 6 个 KDA heads，而同一 head shard 的序列又分布在 8 个 CP ranks 上。

---

## 5. Prefill 与 Decode 的执行差异

### 5.1 Decode：单 token 递归更新

Decode 时，当前 token 必须依赖上一个 token 的状态：

\[
S_{t-1}\rightarrow S_t\rightarrow S_{t+1}
\]

单请求内无法在 token 维同时计算多个未来 token。其特点是：

- 每次只处理一个 token；
- 每个 head 的状态尺寸固定；
- 计算量相对状态搬运量较低；
- 状态必须在 token 之间保留；
- batch-1 或小 batch 下 GPU 并行度较差。

这正是 persistent-state FPGA 最有潜力的场景。

### 5.2 Prefill：chunkwise 并行

Prefill 已知全部输入 token，可以将序列分块，并通过 KDA 的 chunkwise/WY 表示提升并行性。每个 chunk 内的大矩阵运算能够转化为更适合 Tensor Core 的 GEMM。

因此：

- prefill 的 dense GEMM 更适合 GPU；
- KDA 并不意味着 prefill 必须逐 token 串行；
- 把完整 prefill KDA 迁移到 FPGA，通常会与 GPU 争夺其擅长的矩阵吞吐；
- FPGA 更有价值的边界可能是跨 chunk 状态传播和通信，而不是 chunk 内主 GEMM。

---

## 6. 为什么 KDA Decode 在 GPU 上容易受限

### 6.1 固定状态不等于低开销

KDA 把随序列增长的 KV cache 换成固定大小状态：

\[
O(L)\rightarrow O(1)\quad\text{（相对于序列长度）}
\]

但每个 token 仍然要访问整个状态矩阵。对 batch-1 decode，单个 \(128\times128\) 状态上的计算量不足以摊薄内存访问。

已有 GDN FPGA 论文对类似 recurrence 的分析显示，H100 上其算术强度约为 \(0.87\) FLOP/B，明显低于 H100 的 ridge point。KDA 增加逐通道 gate 后仍具有相同的核心问题：状态读写量大，而每个元素上执行的运算有限。

### 6.2 GPU cache 不能等同于显式片上常驻

GPU 有较大的 L2 cache，但它是硬件管理的共享缓存：

- 不能简单保证某个请求的 69 层 KDA 状态跨 kernel、跨 token 永远留在 L2；
- 真实 serving 还会同时运行投影、MLA、MoE、通信和其他请求；
- 多请求会让状态工作集迅速超过 cache 容量。

FPGA BRAM/URAM 则是软件显式管理的 scratchpad，可以把状态分配到确定的 bank，并保证它一直保留到请求结束或主动换出。

### 6.3 batch 增大后 GPU 优势会上升

FPGA 的优势主要针对小 batch：

- batch 增大后，GPU 可把多个请求的 head/update 合并；
- 状态计算更接近 batched GEMM；
- Tensor Core 利用率上升；
- FPGA 片上存储却需要为每个请求复制一份状态。

K3 在 TP16、BF16 下约需 \(12.94\) MiB/request/rank。即使单个请求能够勉强放入某些高端 FPGA 的片上 SRAM，多请求也会很快迫使状态 spill 到 HBM，重新变成带宽受限。

---

## 7. FPGA 为什么可能适合 KDA

| KDA 特征 | 与 FPGA 的匹配点 |
|---|---|
| 固定 \(128\times128\) 状态 | 易于 bank 化和静态地址规划 |
| 跨 token 长期复用 | BRAM/URAM 可 persistent 保存 |
| 行级逐通道 gate | 可在状态流经时广播并融合 |
| 规则的矩阵–向量与 rank-1 update | 易于构造定制数据流 |
| 单 token、小 batch | FPGA 不依赖大规模 SIMT 并行度 |
| 状态必须顺序更新 | 深流水比大量小 kernel launch 更自然 |

真正的优势不是 FPGA HBM 带宽高于 GPU——通常并不是——而是：

\[
\boxed{\text{让状态根本不离开片上存储}}
\]

如果状态无法片上常驻，FPGA 也要反复访问自己的 HBM，此时相对高端 GPU 的优势会显著减弱甚至消失。

---

## 8. 已有 GDN FPGA 工作及其启示

2026 年论文 *A Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode on FPGA* 已经提出：

- 将 GDN 的完整 recurrent state 跨 token 常驻于 FPGA BRAM；
- 将 recurrence 融合为五阶段流水；
- 把朴素的三次状态遍历压缩为一次读和一次写；
- 在 AMD Alveo U55C 上实现；
- 对论文采用的 Qwen3-Next 配置保存 32 个 \(128\times128\) FP32 状态，总计 2 MiB；
- 最快配置报告 63 \(\mu\text{s}\)/token，相对 H100 PCIe GPU reference 为 4.5× 低延迟，并报告最高 60× 能效优势。

### 8.1 persistent state 的两层含义

算法层面：

\[
S_t=f(S_{t-1},x_t)
\]

生成 token \(t\) 后，\(S_t\) 不被丢弃，而是下一 token 的输入。

硬件层面：状态不只是“逻辑上存在”，而是跨 token 始终保留在 FPGA BRAM/URAM 中：

\[
\text{片上 }S_{t-1}
\rightarrow\text{更新}
\rightarrow\text{片上 }S_t
\]

### 8.2 这篇论文覆盖了什么，又没有覆盖什么

它已经覆盖了最核心的基本思想：

> 用 FPGA 片上 persistent state 消除 linear-attention decode 的 HBM round-trip。

若继续研究 KDA，创新点不能只停留在“把 KDA state 放 BRAM”。还至少需要处理：

- KDA 的逐通道 \(\alpha\)；
- K3 的 96 heads、69 个 KDA 层及 TP shard；
- ShortConv state；
- 多请求状态调度和冷热状态迁移；
- MTP 拒绝后的 rollback/replay；
- GPU–FPGA P2P 与异步调度；
- 与 K3 fused kernel 的端到端比较。

此外，论文的 2 MiB GDN state 与 K3 的全模型状态规模不同，不能直接把其“完整状态轻松放入 BRAM”的结论照搬到 K3。

---

## 9. GPU–FPGA 切分方案比较

### 9.1 方案一：只卸载 KDA recurrence

GPU 完成：

- Q/K/V 与 gate 投影；
- ShortConv；
- Swish、L2Norm；
- output projection；
- MLA、MoE 与其余网络。

FPGA 完成：

- 保存 \(S_t\)；
- 逐通道衰减；
- delta retrieval/correction；
- rank-1 state update；
- 计算 \(o_t=S_t^{\mathsf T}q_t\)。

优点是 FPGA 只处理最有差异化的 persistent-state 部分，且无需保存巨量投影权重。

问题是每个 token 仍需穿过 69 个 KDA 层：

\[
\text{GPU}\rightarrow\text{FPGA}\rightarrow\text{GPU}
\]

重复 69 次。即使每次传输的数据不大，逐层依赖产生的 69 次同步、P2P 启动和调度延迟也很危险。

以 TP16 为例，每层每 rank 有 6 个 head。若仅粗略计算 BF16 的 \(q,k,v,\alpha\) 和输出：

\[
(4+1)\times6\times128\times2
\approx7.5\text{ KiB/layer/token}
\]

\(\beta\) 和元数据相对较小。数据量本身未必巨大，但低延迟系统中更致命的是 69 个串行边界，而非累计字节数。

此外，现有 GPU 实现会融合：

- recurrent state replay；
- bonus token 处理；
- 下一轮 draft window；
- ShortConv、归一化与 gate。

只切走 recurrence 会破坏这种融合。

### 9.2 方案二：把完整 KDA 层放到 FPGA

GPU 只发送和接收：

\[
x_t,y_t\in\mathbb R^{7168}
\]

FPGA 内部完成投影、ShortConv、激活、recurrence、output gate 和 output projection。

接口更干净，但权重代价很大。仅用 Q/K/V/O 四个近似满投影粗略估算：

\[
4\times7168\times(96\times128)
\approx352\text{ M parameters/layer}
\]

BF16 下约为：

\[
704\text{ MB/layer}
\]

69 层约：

\[
48.6\text{ GB}
\]

这还未计 gate、卷积及其他参数。该估算只用于说明量级，真实 K3 投影结构和量化格式会改变准确数值。

这些大矩阵乘法和权重流正是高端 GPU 擅长的部分。FPGA 的矩阵吞吐与 HBM 带宽通常不占优势，因此不推荐把完整 KDA 层整体迁出 GPU。

### 9.3 方案三：跨请求异步流水

可让：

- FPGA 处理请求 A 的 KDA state；
- GPU 同时处理请求 B 的 MLA/MoE；
- 完成后调度器再恢复请求 A；
- 不同 microbatch 在两类设备上交错执行。

这样一部分设备同步可以被其他请求的计算隐藏。

但它主要提升吞吐，而非 batch-1 单请求延迟，并且需要：

- 足够多的独立请求；
- continuous batching 支持异构事件依赖；
- FPGA 同时保存多个请求的多层状态；
- 热状态不频繁 spill 到 HBM。

这与 FPGA 片上容量存在直接冲突：并发越高，越难让所有请求的 KDA state 常驻片上。

### 9.4 三种方案总结

| 方案 | 接口 | 核心优势 | 主要问题 | 判断 |
|---|---|---|---|---|
| 只卸载 recurrence | 传 projected inputs，返回 recurrent output | 保留 FPGA persistent-state 优势 | 69 次逐层同步、破坏融合 | 微架构合理，系统边界较差 |
| 完整 KDA 层 | 传/回 7168 维 hidden state | 设备边界简洁 | 巨量权重流，与 GPU 争夺 GEMM | 不推荐 |
| 跨请求异步流水 | recurrence 接口不变，但隐藏等待 | 提升整体吞吐 | 需高并发和大状态容量 | 可作为服务系统优化 |

---

## 10. MTP 验证与 KDA State Replay

### 10.1 MTP 验证是什么

更准确地说，是 MTP/EAGLE 风格的 Draft 模型提出多个候选 token，完整 Target 模型一次验证这些候选。

假设 Draft 提出：

\[
d_1,d_2,d_3,d_4
\]

Target 验证后只接受前两个，并在第三个位置给出修正 token \(x_3\)，则最终推进：

\[
P\rightarrow d_1\rightarrow d_2\rightarrow x_3
\]

### 10.2 为什么 KDA 状态不能直接保留

验证四个候选时，Target 的 KDA 状态已经依次前进：

\[
S_0\rightarrow S_1\rightarrow S_2\rightarrow S_3\rightarrow S_4
\]

如果只接受 \(d_1,d_2\)，正确状态应为 \(S_2\)，不能保留混入被拒绝 token 的 \(S_4\)。

为每个候选位置保存完整状态：

\[
S_0,S_1,S_2,S_3,S_4
\]

会显著放大状态容量和内存流量。

### 10.3 K3 的 replay 思路

K3 不为每个 draft token 保存一份完整 state，而是：

1. 保留验证前的 base state \(S_0\)；
2. 缓存候选 token 的较小 projected inputs；
3. 得知接受长度后，只重放 accepted tokens；
4. 从 \(S_0\) 重建正确状态；
5. 写回 verified token 和 bonus token 对应状态。

若接受两个 token：

\[
S_0
\xrightarrow{\text{replay }d_1}S_1
\xrightarrow{\text{replay }d_2}S_2
\]

### 10.4 为什么这与 FPGA 契合

FPGA 可以维护：

\[
S_{\text{base}}
+\{\text{projected input}_1,\ldots,\text{projected input}_k\}
\]

并在片上执行 accepted-prefix replay，无需：

- 将多个完整状态复制到 HBM；
- 从 GPU 重新发送整层 hidden state；
- 让状态频繁离开 persistent engine。

因此，**persistent state + speculative replay** 比单纯复刻已有 GDN FPGA recurrence 更可能形成新的 KDA 加速研究点。

---

## 11. KDA Context Parallelism（KCP）

### 11.1 为什么普通序列切分会串行

假设把一条长序列分给 4 个 CP ranks：

| CP rank | 本地 token 片段 |
|---:|---|
| 0 | 第 1 段 |
| 1 | 第 2 段 |
| 2 | 第 3 段 |
| 3 | 第 4 段 |

rank 1 的第一个 token 需要 rank 0 的最终状态：

\[
S_{\mathrm{in},1}=S_{\mathrm{out},0}
\]

若直接逐段传递，就会退化为：

\[
\text{rank 0完成}\rightarrow
\text{rank 1开始}\rightarrow
\text{rank 2开始}\rightarrow
\text{rank 3开始}
\]

这没有真正的 context parallelism。

### 11.2 将本地序列精确表示为 fragment

一段 token 对任意输入状态的整体作用可以写成仿射变换：

\[
\boxed{
S_{\mathrm{out},i}=M_iS_{\mathrm{in},i}+E_i
}
\]

其中：

- \(M_i\in\mathbb R^{d_k\times d_k}\)：该片段如何变换传入状态；
- \(E_i\in\mathbb R^{d_k\times d_v}\)：入口状态为 0 时，该片段自身积累的状态；
- \(F_i=(M_i,E_i)\)：rank \(i\) 的 fragment。

这不是有损的文本压缩，而是对整段 token 状态作用的精确等价表示。无论本地片段包含 1K、16K 还是 250K token，边界 fragment 的形状固定。

K3 中 \(d_k=d_v=128\)，所以每个 head 的：

\[
M_i,E_i\in\mathbb R^{128\times128}
\]

### 11.3 fragment 如何组合

设：

\[
F_a(S)=M_aS+E_a
\]

\[
F_b(S)=M_bS+E_b
\]

则先经过 \(F_a\)，再经过 \(F_b\)：

\[
F_b(F_a(S))
=M_bM_aS+(E_b+M_bE_a)
\]

所以组合算子为：

\[
\boxed{
(M_b,E_b)\circ(M_a,E_a)
=(M_bM_a,\ E_b+M_bE_a)
}
\]

它满足结合律，因此可以做 prefix scan。

### 11.4 当前 all-gather + merge 流程

每个 rank 先独立计算本地：

\[
F_i=(M_i,E_i)
\]

随后 all-gather，使每个 rank 得到所有 fragments：

\[
[F_0,F_1,\ldots,F_{P-1}]
\]

rank \(r\) 再在本地 GPU 上合并它之前的 fragments：

\[
S\leftarrow0
\]

\[
S\leftarrow M_jS+E_j,qquad j=0,\ldots,r-1
\]

从而得到本地片段的入口状态 \(S_{\mathrm{in},r}\)。例如：

\[
S_{\mathrm{in},1}=E_0
\]

\[
S_{\mathrm{in},2}=M_1E_0+E_1
\]

\[
S_{\mathrm{in},3}=M_2(M_1E_0+E_1)+E_2
\]

严格地说，当前 FLA 文档描述的是 all-gather 后各 rank 在本地顺序 merge；由于组合满足结合律，也可以进一步实现树状并行 scan。

---

## 12. FPGA SmartNIC 执行 KCP Scan

> 本节是我们提出的研究设想，不是 K3 当前已经采用的实现。

### 12.1 当前 GPU 路径

当前抽象流程为：

1. 各 GPU 计算本地 \((M_i,E_i)\)；
2. all-gather 所有 fragments；
3. 每张 GPU 执行自己的 prefix merge；
4. 得到本 rank 的入口状态；
5. 完成本地 KDA 计算。

潜在问题包括：

- 每张 GPU 都接收所有 rank 的 fragments；
- 多个 rank 重复执行部分 prefix merge；
- TP 后每个 GPU 只有少量 heads 时，小矩阵链不一定充分利用 GPU；
- merge 存在跨 fragment 的顺序依赖；
- 可能产生小 kernel 和同步开销。

### 12.2 SmartNIC 方案

如果 FPGA 位于 NIC/collective 数据路径中，可以让它：

- 接收各 GPU 的本地 fragments；
- 维护文档、请求、layer、head 和 rank 顺序；
- 执行 ring scan 或 tree scan；
- 计算每个 rank 的 exclusive prefix；
- 只把该 rank 需要的入口状态发送给对应 GPU。

例如 rank 2 不再接收全部 \([F_0,F_1,F_2,F_3]\)，而只接收：

\[
S_{\mathrm{in},2}=M_1E_0+E_1
\]

该方案试图融合：

\[
\boxed{
\text{collective 通信}
+\text{顺序状态传播}
+\text{小矩阵 scan}
}
\]

它的系统边界比“每个 KDA 层额外调用一次 PCIe FPGA”更干净，因为 FPGA 位于本来就必须经过的网络路径，而不是额外插入一个设备往返。

### 12.3 为什么它不一定比 GPU 快

fragment 组合包括：

\[
M_bM_a
\]

和：

\[
M_bE_a
\]

二者都是 \(128\times128\) 稠密矩阵乘法，并且连续组合可能需要较高精度来限制误差累积。稠密矩阵乘法本来就是 GPU Tensor Core 的优势项。

所以 SmartNIC 方案成立的条件不是“FPGA 更适合流水线”这一句，而必须实测证明：

\[
\boxed{
\text{减少的网络流量、GPU同步与重复merge时间}
>
\text{FPGA矩阵计算、缓存与NIC调度开销}
}
\]

### 12.4 Ring 与 Tree

| 实现 | 优点 | 问题 |
|---|---|---|
| Ring scan | 与网络拓扑自然，单步控制简单 | 延迟随 rank 数按 \(O(P)\) 增长 |
| Tree scan | 理论深度 \(O(\log P)\) | 路由、缓存与上/下扫控制更复杂 |
| GPU all-gather + merge | Tensor Core 强、软件栈成熟 | 全量复制 fragment，GPU 重复 merge |
| FPGA SmartNIC scan | 可边收边算、融合通信 | FP32/高精度小 GEMM 未必有性能优势 |

### 12.5 值得继续研究的必要条件

SmartNIC-KCP 只有在以下条件同时出现时才更有希望：

\[
\text{CP规模较大}
+\text{每GPU本地head较少}
+\text{all-gather/merge为显著瓶颈}
+\text{网络传输可与FPGA计算重叠}
\]

如果 profiling 表明 KCP merge 只占很小比例，或者已被 collective 完全隐藏，那么这个方向不值得做。

---

## 13. 研究方向与实验路线

### 13.1 方向 A：Persistent KDA State Engine + MTP Replay

建议设计包含：

- KDA 逐通道 gate 数据流；
- fused decay、retrieval、delta update 和 output；
- ShortConv state 管理；
- request–layer–head 三级状态地址空间；
- TP-aware state placement；
- base state + projected-input replay log；
- accepted-prefix replay；
- 热请求片上常驻、冷请求 HBM spill；
- GPU–FPGA P2P 和异步事件接口。

优势：FPGA 的片上状态能力可直接消除核心 HBM 流量，硬件收益逻辑最明确。

风险：已有 GDN persistent-state FPGA 论文，必须通过 KDA 特性、K3 系统集成和 MTP replay 建立足够创新性。

### 13.2 方向 B：FPGA SmartNIC KCP Scan

建议研究：

- fragment 的网络格式与布局；
- 多 head、多 layer 的批处理方式；
- ring/tree scan 微架构；
- BF16/FP32/混合精度误差；
- 与 RDMA/NCCL/KCP runtime 的协同；
- 边接收边计算的 overlap；
- 相对 GPU all-gather + merge 的通信量和关键路径延迟。

优势：FPGA 位于原有通信路径，系统切分边界干净，创新性更强。

风险：核心 merge 仍是稠密矩阵乘，FPGA 算力可能不占优；只有在网络和同步主导时才成立。

### 13.3 方向 C：KDA Prefix-State Cache / Migration Layer

让 FPGA 或 CXL/SmartNIC 侧内存管理：

- KDA recurrent states；
- MLA paged KV cache；
- prefix 命中边界；
- state prefetch、eviction 和 request migration。

该方向可以提高长上下文服务容量和缓存复用率，但更偏系统与存储管理，不一定带来显著计算加速。

### 13.4 必须先完成的 profiling

在投入 RTL/HLS 前，应先回答：

1. K3 decode 中 KDA recurrence 占每 token 延迟的多少？
2. QKV/ShortConv/gate/replay 当前融合到了什么程度？
3. 不同 batch、TP、并发数下，KDA state 的 HBM 读写量是多少？
4. KCP 的 all-gather、merge GEMM 和等待分别占多少？
5. KDA recurrence 或 KCP merge 是否处于端到端关键路径？
6. GPU–FPGA P2P 的单次启动延迟和有效带宽是多少？
7. 片上状态容量允许同时驻留多少请求？

### 13.5 建议的评估矩阵

| 维度 | 建议取值 |
|---|---|
| 阶段 | prefill / decode / speculative verification |
| Batch | 1、小 batch、continuous batching |
| 序列长度 | 短、中、超长上下文 |
| 并行 | 不同 TP、CP、EP 组合 |
| 状态位置 | GPU HBM、FPGA BRAM/URAM、FPGA HBM |
| 通信 | Host staging、PCIe P2P、SmartNIC/RDMA |
| 精度 | BF16、FP32 state、混合精度 |
| 指标 | TTFT、TPOT、吞吐、P99、能耗、状态容量 |

端到端基线必须包含真实 GPU fused kernel；只与朴素逐算子 GPU 实现比较不足以证明系统价值。

---

## 14. 最终判断

| 评价维度 | KDA recurrence |
|---|---|
| 固定尺寸数据流 | 很适合 FPGA |
| 控制流规则性 | 很适合 FPGA |
| 小 batch GPU 利用率 | FPGA 有潜在优势 |
| 片上状态复用 | FPGA 的核心优势 |
| Prefill dense GEMM | GPU 更有优势 |
| 多请求大 batch | GPU 优势上升，FPGA 容量压力增大 |
| 单独从 K3 逐层切出 | 系统边界不理想 |
| 每 token 设备往返 | 69 个 KDA 层使同步风险很高 |
| 端到端加速 | 取决于 KDA 占比、融合破坏和异步隐藏能力 |

总的结论是：

> **KDA 是目前比普通 GEMV 更有理由放到 FPGA 上的 LLM 算子，因为它的关键瓶颈是反复读写跨 token 存在的固定 recurrent state，而 FPGA 可以让热状态长期驻留片上。**

但同时：

> **“GPU 执行 K3 其他部分，独立 FPGA 同步执行每个 KDA 层”不是理想的系统方案。69 次逐层往返、KDA/MLA/MoE 交错和 fused kernel 拆分，很可能吞掉局部算子加速。**

根据最新讨论，方向优先级应从两个维度理解：

| 目标 | 更推荐的方向 |
|---|---|
| 追求最直接、可解释的硬件收益 | Persistent KDA state engine + MTP replay |
| 追求更干净的异构系统边界和更强新颖性 | FPGA SmartNIC KCP scan，但先以 profiling 证明瓶颈 |
| 追求服务容量而非算力加速 | KDA prefix-state cache / migration |
| 直接把整个 KDA 层搬到 FPGA | 不推荐 |

最稳妥的研究策略不是立即选定某一种架构，而是先对 K3 的 KDA decode 与 KCP 分别做端到端 profiling：

- 若 KDA state HBM round-trip 主导 decode，则优先做 persistent engine；
- 若大规模 CP 中 all-gather/merge 和 GPU 等待主导，则 SmartNIC scan 才具备研究价值；
- 若两者都未进入关键路径，则 KDA+FPGA 不应作为主要加速方向。

---

## 15. 参考资料

1. Kimi Team, [Kimi K3: Open Frontier Intelligence](https://arxiv.org/abs/2607.24653), 2026.
2. Moonshot AI, [Kimi K3 官方模型配置](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json).
3. Kimi Team, [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692), 2025.
4. Gupta et al., [A Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode on FPGA](https://arxiv.org/abs/2603.05931), 2026.
5. Flash Linear Attention, [Context Parallel of Linear Attention / KCP 实现说明](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/README.md).

