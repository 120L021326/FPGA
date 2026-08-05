# A Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode on FPGA

> 论文：Neelesh Gupta, Peter Wang, Rajgopal Kannan, Viktor K. Prasanna, *A Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode on FPGA*, arXiv:2603.05931v1, 2026.
>
> 本文整理范围：论文的问题背景、Gated DeltaNet（GDN）算法、persistent-state FPGA 数据流、HLS 映射、实验结果、实验可信度与系统局限，并补充此前关于“预测误差”和“状态矩阵为何可视为线性映射”的讨论。

---

## 1. 一句话概括

这篇论文并不是简单地用 FPGA 替代 GPU 执行普通 GEMV，而是抓住了 Gated DeltaNet decode 的特殊性质：

- 每个 token 都要完整读写固定大小的循环状态矩阵；
- 计算量很小，外部存储流量却很大；
- 状态在相邻 token 之间持续存在，但无需随着序列长度增长；
- 单层 2 MiB 的状态可以常驻 FPGA BRAM。

因此，论文把原本需要逐 token 往返 HBM 的状态变成片上 persistent state，并通过代数变换将三次状态遍历减少为两次。

论文声称的最佳结果是单个 GDN layer、batch=1、FP32 decode 延迟为 $63.2\ \mu s$，相对 H100 PyTorch reference 加速 $4.5\times$。但该结果只来自 HLS 综合周期估计；真正完成布局布线的 $H_{\text{iter}}=2$ 配置在 263 MHz 下为 $161.7\ \mu s$，相对 H100 的严格可实现加速约为 $1.76\times$。

---

## 2. 论文解决的核心问题

### 2.1 标准注意力的 KV cache

标准 causal self-attention 在 decode 时需要保存历史 token 的 key 和 value。随着上下文长度 $n$ 增长，KV cache 容量也按 $O(n)$ 增长。

线性注意力、DeltaNet、Mamba 等 subquadratic sequence model 用固定大小的循环状态替代不断增长的 KV cache，使状态容量相对于序列长度为 $O(1)$。

但“状态容量不随序列增长”不等于“每个 token 的状态访问很便宜”。以 GDN 为例，每个新 token 都必须：

1. 读取旧状态；
2. 根据当前 $q,k,v$ 更新状态；
3. 写回新状态；
4. 用状态产生当前输出。

状态虽然固定，却会被每个 token 反复完整遍历。

### 2.2 论文采用的 GDN 配置

论文针对 Qwen3-Next 风格的单个 GDN layer：

| 参数 | 数值 |
|---|---:|
| query heads $h_q$ | 16 |
| key heads $h_k$ | 16 |
| value heads $h_v$ | 32 |
| head dimension $d$ | 128 |
| 数据类型 | FP32 |
| batch size | 1 |
| 每个 value head 的状态 | $128\times128$ FP32 |

总状态容量为：

\[
h_vd^2\times4
=32\times128\times128\times4
=2{,}097{,}152\ \text{bytes}
=2\ \text{MiB}.
\]

GPU 每个 token 至少需要读取和写回一次状态，因此状态流量为：

\[
2\ \text{MiB read}+2\ \text{MiB write}
=4\ \text{MiB/token/layer}.
\]

而整次 recurrence 只有约 $4.2$ MFLOPs。按是否计入 token 输入等细节，其算术强度约为 $0.87\sim1.0$ FLOP/B，远低于 H100 FP32 roofline 的 ridge point，因此结构上属于 memory-bound workload。

### 2.3 为什么 FPGA 有机会

论文比较的平台为：

| 平台 | 外部带宽 | 片上存储 | 特点 |
|---|---:|---:|---|
| NVIDIA H100 PCIe | 约 2.0 TB/s HBM3 | 50 MB L2 | cache 由硬件管理，软件不能严格保证跨 kernel 的状态驻留 |
| AMD Alveo U55C | 约 460 GB/s HBM2e | 约 17.6 MB BRAM | BRAM 可作为软件显式管理、跨调用保留的 scratchpad |

U55C 的 HBM 带宽明显低于 H100，所以论文的论点并不是：

\[
BW_{\text{FPGA}}>BW_{\text{GPU}}.
\]

真正的优势是：

\[
\boxed{\text{state external-memory traffic on FPGA}=0}
\]

即将完整状态跨 token 常驻 BRAM，把外部带宽问题转化为片上多 bank 并行访问问题。

---

## 3. 从普通线性注意力理解状态矩阵

### 3.1 普通线性注意力

忽略 feature map 和归一化项，普通线性注意力可以写成：

\[
S_t=S_{t-1}+k_tv_t^{\mathsf T},
\]

\[
o_t=S_t^{\mathsf T}q_t.
\]

展开状态：

\[
S_t=\sum_{i=1}^{t}k_iv_i^{\mathsf T}.
\]

因此输出为：

\[
o_t
=\left(\sum_{i=1}^{t}k_iv_i^{\mathsf T}\right)^{\mathsf T}q_t
=\sum_{i=1}^{t}v_i(k_i^{\mathsf T}q_t).
\]

这里：

- $k_i^{\mathsf T}q_t$ 是历史 key 与当前 query 的内积相似度；
- 每个历史 value $v_i$ 按该相似度参与输出；
- 它可以理解为没有 softmax 的 attention。

### 3.2 为什么说 $S^{\mathsf T}$ 是线性映射

“把状态矩阵看成线性映射”只是指：

\[
x\mapsto S^{\mathsf T}x
\]

是从 key/query 空间到 value 空间的线性函数，并不意味着它像 Python 字典一样严格完成 $k_i\mapsto v_i$。

若状态中只有一个 token：

\[
S=kv^{\mathsf T},
\]

则对任意输入 $x$：

\[
S^{\mathsf T}x
=(kv^{\mathsf T})^{\mathsf T}x
=v(k^{\mathsf T}x).
\]

它先计算 $x$ 与 $k$ 的内积，再用该内积缩放 $v$。当输入正好为 $k$ 时：

\[
S^{\mathsf T}k
=v(k^{\mathsf T}k)
=\lVert k\rVert_2^2v.
\]

所以用户之前提出的疑问是正确的：一般情况下它并不严格等于 $v$。只有 $k$ 是单位向量时才有：

\[
S^{\mathsf T}k=v.
\]

### 3.3 多个 key 时的串扰

若：

\[
S=\sum_i k_iv_i^{\mathsf T},
\]

用 $k_j$ 查询得到：

\[
S^{\mathsf T}k_j
=\sum_i v_i(k_i^{\mathsf T}k_j)
=\lVert k_j\rVert^2v_j
+\sum_{i\ne j}v_i(k_i^{\mathsf T}k_j).
\]

第二项是其他 key 对当前查询的影响。它并不总是噪声：相关历史 token 的 value 聚合本来就是注意力想要的效果；只有无关信息因 key 相似而混入时，才构成有害串扰。

因此，$S^{\mathsf T}k$ 更准确的含义是：

> 状态中的线性预测器根据输入 $k$ 给出的 value 估计，而不是无条件精确的键值查找结果。

---

## 4. Delta rule 中“预测误差”的含义

### 4.1 预测不是预测下一个 token

DeltaNet 中：

\[
r_t=S_{t-1}^{\mathsf T}k_t
\]

表示旧状态收到当前 key 后给出的 value 估计。这里的“预测”只是：

\[
k_t\longmapsto \widehat v_t=r_t,
\]

不是预测下一个 token。

当前 value projection 产生的 $v_t$ 被作为本次希望状态建立的关联目标，于是：

\[
e_t=v_t-r_t
\]

就是预测误差。

### 4.2 标准 Delta 更新

标准 DeltaNet 的核心更新可以写成：

\[
S_t
=S_{t-1}
+\beta_tk_t(v_t-S_{t-1}^{\mathsf T}k_t)^{\mathsf T}.
\]

其直观含义是：

- 若旧状态完全不知道该关联，$r_t\approx0$，写入内容接近完整 $v_t$；
- 若旧状态已经预测准确，$r_t\approx v_t$，几乎不再重复写入；
- 若旧状态预测错误，只写入使预测向 $v_t$ 修正的残差。

普通线性注意力更像不断累积外积统计量；DeltaNet 更像容量有限、可在线纠错的关联记忆。

### 4.3 为什么它等价于一步梯度下降

定义当前样本上的平方误差：

\[
\mathcal L_t(S)
=\frac12\left\|S^{\mathsf T}k_t-v_t\right\|_2^2.
\]

对 $S$ 求梯度：

\[
\nabla_S\mathcal L_t
=k_t(S^{\mathsf T}k_t-v_t)^{\mathsf T}.
\]

以 $\beta_t$ 为步长做一次梯度下降：

\[
\begin{aligned}
S_t
&=S_{t-1}-\beta_t\nabla_S\mathcal L_t\\
&=S_{t-1}
+\beta_tk_t(v_t-S_{t-1}^{\mathsf T}k_t)^{\mathsf T}.
\end{aligned}
\]

因此可以把它理解成：推理过程中，状态 $S_t$ 是一个在线更新的小型线性模型；模型权重本身没有训练，变化的是循环状态。

### 4.4 归一化与一次修正

令：

\[
r_t=S_{t-1}^{\mathsf T}k_t,
\qquad
S_t=S_{t-1}+\beta_tk_t(v_t-r_t)^{\mathsf T}.
\]

更新后用同一个 key 回读：

\[
S_t^{\mathsf T}k_t
=r_t+\beta_t\\lVert k_t\rVert^2(v_t-r_t).
\]

若 $\lVert k_t\rVert=1$ 且 $\beta_t=1$，则：

\[
S_t^{\mathsf T}k_t=v_t.
\]

但这次更新也会影响其他 key。对任意历史 key $k_j$：

\[
S_t^{\mathsf T}k_j
=S_{t-1}^{\mathsf T}k_j
+\beta_t(k_t^{\mathsf T}k_j)(v_t-r_t).
\]

因此 Delta rule 并没有消除有限维关联记忆的容量限制：

- 若 $k_t^{\mathsf T}k_j\approx0$，对旧关联影响很小；
- 若二者高度相似，纠正当前关联时也会改变旧关联；
- 模型训练需要共同学习 key 的表示、写入强度与遗忘策略。

---

## 5. Gated DeltaNet 的完整 recurrence

论文对单个 value head 给出的 decode recurrence 为：

\[
r_t=S_{t-1}^{\mathsf T}k_t,
\tag{1}
\]

\[
\Delta v_t=\beta_t(v_t-r_t),
\tag{2}
\]

\[
S_t=g_tS_{t-1}+k_t\Delta v_t^{\mathsf T},
\tag{3}
\]

\[
o_t=\frac1{\sqrt d}S_t^{\mathsf T}q_t.
\tag{4}
\]

其中：

- $q_t,k_t,v_t\in\mathbb R^d$；
- $S_t\in\mathbb R^{d\times d}$；
- $\beta_t\in(0,1)$ 控制本次预测误差写入多少；
- $g_t$ 控制旧状态整体保留多少；
- $A_{\log}$ 和 $dt_{\text{bias}}$ 是 learned per-head parameters；
- $\alpha_t,b_t$ 是 token-dependent inputs。

论文将两个门控写成：

\[
g_t
=\exp\left[-\sigma(\alpha_t)\,e^{A_{\log}}\,
\operatorname{softplus}(dt_{\text{bias}})\right],
\]

\[
\beta_t=\sigma(b_t).
\]

最直观的总结是：

\[
\boxed{
\text{新状态}
=\text{衰减后的旧记忆}
+\text{当前预测误差的选择性写入}
}
\]

其中 $g_t$ 和 $\beta_t$ 的作用不同：

| 门控 | 作用对象 | 直观含义 |
|---|---|---|
| $g_t$ | 整个旧状态 | 旧记忆保留多少 |
| $\beta_t$ | 当前误差 | 当前关联修改多少 |

---

## 6. 朴素实现为什么需要三次状态遍历

按公式直接实现单个 head：

1. $r=S^{\mathsf T}k$：完整读取一次状态；
2. $S\leftarrow gS+k\Delta v^{\mathsf T}$：再次读取并写回状态；
3. $o=\frac1{\sqrt d}S^{\mathsf T}q$：第三次读取更新后的状态。

若列方向并行度为 $P_K=16$，每次遍历 $128\times128$ 状态需要：

\[
\frac{d^2}{P_K}
=\frac{128^2}{16}
=1024\ \text{cycles}.
\]

三次遍历的理论成本为：

\[
T_{\text{head}}^{\text{naive}}
=3\times1024
=3072\ \text{cycles}.
\]

---

## 7. 代数变换：把三次遍历减少为两次

将状态更新代入输出：

\[
\begin{aligned}
S_t^{\mathsf T}q_t
&=\left(g_tS_{t-1}+k_t\Delta v_t^{\mathsf T}\right)^{\mathsf T}q_t\\
&=g_tS_{t-1}^{\mathsf T}q_t
+\Delta v_t(k_t^{\mathsf T}q_t).
\end{aligned}
\]

等价地写成：

\[
S_t^{\mathsf T}q_t
=g_tS_{t-1}^{\mathsf T}q_t
+(q_t^{\mathsf T}k_t)\Delta v_t.
\tag{5}
\]

于是第一次读取旧状态时，可以同时计算：

\[
r_t=S_{t-1}^{\mathsf T}k_t
\]

和：

\[
\hat o_t=g_tS_{t-1}^{\mathsf T}q_t.
\]

之后用一个向量修正项得到输出：

\[
\Delta v_t=\beta_t(v_t-r_t).
\]

\[
o_t=\frac1{\sqrt d}{(\hat o_t+(q_t^{\mathsf T}k_t)\Delta v_t)}.
\]

因此无需为了输出再次读取更新后的 $S_t$。状态访问变为：

1. 一次 read pass：同时计算 $S_{t-1}^{\mathsf T}k_t$ 和 $S_{t-1}^{\mathsf T}q_t$；
2. 一次 read-modify-write pass：计算并写回 $S_t$。

理论状态遍历成本从 $3072$ cycles 降为约 $2048$ cycles；计入短向量运算和流水启动，论文 HLS 报告中的实际 iteration interval 约为 $2106$ cycles，约带来 $1.46\times$ 改善。

---

## 8. 五阶段 fused datapath

论文把单个 head 的计算组织为五个阶段。

### Phase 1：query-key 点积

\[
a_t=q_t^{\mathsf T}k_t.
\]

这里用 $a_t$ 表示临时点积，避免与门控输入 $\alpha_t$ 混淆。

### Phase 2：一次状态读取完成两组矩阵向量乘

在同一次 BRAM read pass 中，对旧状态同时累加：

\[
r_t=S_{t-1}^{\mathsf T}k_t,
\]

\[
\hat o_t=g_tS_{t-1}^{\mathsf T}q_t.
\]

同一个状态元素被同时用于与 $k_t$ 和 $q_t$ 相乘，从而共享状态读取。

### Phase 3：Delta correction

\[
\Delta v_t=\beta_t(v_t-r_t).
\]

### Phase 4：输出修正

\[
o_t=\frac1{\sqrt d}(\hat o_t+a_t\Delta v_t).
\]

### Phase 5：状态更新

\[
S_t=g_tS_{t-1}+k_t\Delta v_t^{\mathsf T}.
\]

这个五阶段结构的核心不是把公式机械拆成五块，而是让同一次状态读取同时服务 retrieval 和 partial output，再把最终输出变成向量级修正。

---

## 9. 为什么 tiled accumulation 能实现 II=1

状态矩阵列方向按：

\[
P_K=16
\]

并行处理。每一行共有 $128$ 个元素，因此需要：

\[
\frac{128}{16}=8
\]

个 tile。

对同一个输出 partial sum 而言，第一次累加后要经过其他 tile/累加链的轮转，隔 8 个周期才再次访问同一组累加寄存器：

\[
\text{recurrence distance}=8.
\]

FP32 adder 的反馈延迟约为 5 cycles。因为：

\[
8>5,
\]

前一次加法在下一次依赖访问前已经完成，所以 HLS 能做到每周期启动一个 tile：

\[
II=1.
\]

本质上，这是利用多组交错的独立累加链隐藏 FP32 adder 的反馈延迟，而不是要求单条依赖链在一个周期内完成浮点加法。

---

## 10. 利用 GVA 做 head 并行

论文中的配置是 2:1 Grouped Value Attention（GVA）：

- 16 组 query/key heads；
- 32 个 value heads；
- 每一组 $q,k$ 服务两个 value heads；
- 两个 value heads 有各自独立的 $v$、状态矩阵与输出。

因此一对 value heads 可以：

- 共享 $q,k$ 的加载与广播；
- 并行执行两个独立状态矩阵的计算；
- 不复制 q/k buffer 和相应的数据通路。

论文用 $H_{\text{iter}}$ 表示一次 dataflow iteration 内并行处理的 value-head 数量：

| $H_{\text{iter}}$ | 每轮 GVA pair | 遍历 32 个 value heads 的迭代数 $N_{\text{iter}}=32/H_{\text{iter}}$ |
|---:|---:|---:|
| 2 | 1 | 16 |
| 4 | 2 | 8 |
| 8 | 4 | 4 |
| 16 | 8 | 2 |

理论上增大 $H_{\text{iter}}$ 可以减少总迭代次数，但实际会增加 DSP、LUT、FF、扇出和布线压力，因此并不是越大越好。

---

## 11. BRAM 状态组织与 banking

以 $H_{\text{iter}}=8$ 为例，逻辑状态可理解为：

\[
S[\text{iter}][h][i][j],
\]

其中：

- $\text{iter}=0,\ldots,3$：四个 head groups；
- $h=0,\ldots,7$：每组并行处理八个 value heads；
- $i,j=0,\ldots,127$：状态矩阵坐标。

设计采用两类 partition。

### 11.1 Head 维 complete partition

并行的八个 value heads 各自拥有独立 BRAM bank group，避免 head 之间争抢端口。

### 11.2 矩阵列维 cyclic partition

沿列维以 $P_K=16$ 循环分 bank：

\[
\text{bank}(j)=j\bmod16.
\]

每个 head 因此具有 16 个可并行访问的 bank，每周期可读取 16 个 FP32 元素。

对 $H_{\text{iter}}=8$：

\[
8\ \text{heads}\times16\ \text{banks/head}
=128\ \text{BRAM arrays}.
\]

四个 iteration 的状态位于相同物理 bank 的不同地址深度中，不需要把物理 bank 数再乘四。

设计使用 dual-port BRAM 支持数据流阶段之间的读写需求。这里真正发挥的是 FPGA 可显式构造大量、确定性的片上存储端口，而不是 U55C 的外部 HBM 带宽。

---

## 12. Prepare–Compute–Store 三层 dataflow

每个 head group 的处理分成三个大阶段：

1. **Prepare**：从输入 buffer 取对应的 $q,k,v$，计算 $g_t,\beta_t$；
2. **Compute**：执行五阶段 fused GDN datapath；
3. **Store**：把该组输出经 AXI 写回外部存储。

HLS dataflow 允许不同 head groups 重叠：

- group $n-1$：Store；
- group $n$：Compute；
- group $n+1$：Prepare。

稳态 interval 由最慢的 Compute 阶段决定，约为：

\[
II_{\text{group}}\approx2106\ \text{cycles}.
\]

需要特别强调：这种重叠发生在**同一个 token 的不同 head group 之间**，不是相邻生成 token 之间的流水。自回归依赖仍要求下一个 token 等待当前 token 以及模型剩余算子完成。

---

## 13. 延迟模型

论文给出的总延迟模型为：

\[
L
=\frac{h_v}{H_{\text{iter}}}T_{\text{iter}}
+T_{\text{load}},
\]

其中：

- $h_v=32$；
- 对 $H_{\text{iter}}\le8$，$T_{\text{iter}}\approx2106$ cycles；
- $T_{\text{load}}\approx8800\sim10600$ cycles；
- $T_{\text{load}}$ 是通过 AXI 加载每 token 输入的成本。

例如 $H_{\text{iter}}=8$：

\[
L\approx4\times2106+10554
=18978\ \text{cycles}.
\]

在 300 MHz 下：

\[
\frac{18978}{300\times10^6}
\approx63.26\ \mu s.
\]

输入加载占比约为：

\[
\frac{10554}{18978}\approx55.6\%.
\]

因此，在状态 HBM 流量被消除后，$q,k,v$ 和门控参数的外部输入加载成为新的主要瓶颈之一。

论文给出的 token I/O 为：

\[
49{,}664\ \text{bytes}
\approx48.5\ \text{KiB/token/layer}.
\]

---

## 14. 实验设置与结果

### 14.1 实验设置

| 项目 | 配置 |
|---|---|
| FPGA | AMD Alveo U55C |
| HLS | Vitis HLS 2025.1 |
| 综合目标频率 | 300 MHz |
| GPU | NVIDIA H100 PCIe |
| 数据类型 | 全部 FP32 |
| workload | Qwen3-Next 风格的单个 GDN layer，batch=1 decode |
| GPU baseline | NVLabs 官方 PyTorch GatedDeltaNet reference |
| 正确性验证 | C/RTL co-simulation 对照 golden reference |

### 14.2 延迟结果

| 配置 | 周期数 | 延迟 | 相对 H100 | 证据级别 |
|---|---:|---:|---:|---|
| H100 PCIe | — | $285\ \mu s$ | $1\times$ | GPU 实测 |
| $H_{\text{iter}}=2$，300 MHz | 42,538 | $141.7\ \mu s$ | $2.0\times$ | HLS 综合周期估计 |
| $H_{\text{iter}}=2$，263 MHz | 42,538 | $161.7\ \mu s$ | 约 $1.76\times$ | 完成布局布线 |
| $H_{\text{iter}}=4$，300 MHz | 26,252 | $87.4\ \mu s$ | $3.3\times$ | 综合；布局布线失败 |
| $H_{\text{iter}}=8$，300 MHz | 18,978 | $63.2\ \mu s$ | $4.5\times$ | 仅综合估计 |
| $H_{\text{iter}}=16$，300 MHz | 23,206 | $77.4\ \mu s$ | $3.7\times$ | 仅综合估计 |

### 14.3 为什么 $H_{\text{iter}}=16$ 反而变慢

理论模型假设每个 iteration 都维持约 2106 cycles。但 16 个 head 完全展开后：

- 数据通路显著变宽；
- BRAM 到计算单元的扇出和连线增多；
- DSP、FF、LUT 数量大幅增加；
- pipeline depth 与 routing pressure 增大；
- 实际 iteration interval 膨胀到约 6300 cycles。

虽然 iteration 数由 4 次下降到 2 次，总延迟反而高于 $H_{\text{iter}}=8$。这说明 FPGA 的最佳并行度由计算、存储 banking、扇出和物理布线共同决定，而不是只看全芯片资源百分比。

---

## 15. 资源与布局布线

| 配置 | BRAM | DSP | FF | LUT |
|---|---:|---:|---:|---:|
| $H_{\text{iter}}=2$ | 12% | 6% | 7% | 7% |
| $H_{\text{iter}}=4$ | 25% | 12% | 12% | 13% |
| $H_{\text{iter}}=8$ | 25% | 23% | 24% | 25% |
| $H_{\text{iter}}=16$ | 25% | 46% | 54% | 52% |

$H_{\text{iter}}=4$ 虽然全芯片资源比例不高，但在 SLR0 内形成高密度 BRAM—逻辑连接，随后又跨越 SLR 边界，最终出现 88,725 条无法布通的信号。

因此：

- $H_{\text{iter}}=2$：完成布局布线，实际达到 263 MHz；
- $H_{\text{iter}}=4$：综合时序通过，但布局布线失败；
- $H_{\text{iter}}=8,16$：论文没有给出完成物理实现的结果。

所以摘要中的“$4.5\times$ faster”来自 $H_{\text{iter}}=8$ 的 HLS 综合估计。若采用最严格的“已布局布线且满足时序”标准，当前实验证明的加速应是约 $1.76\times$。

---

## 16. 能耗结果应该如何理解

论文使用：

\[
E=P\times L.
\]

H100 按 350 W 整卡 TDP 计算：

\[
350\ \text{W}\times285\ \mu s
=99.8\ \text{mJ/token}.
\]

完成布局布线的 $H_{\text{iter}}=2$ FPGA 按 Vivado 估计的 9.96 W 片上功耗计算：

\[
9.96\ \text{W}\times161.7\ \mu s
=1.61\ \text{mJ/token}.
\]

由此得到约：

\[
\frac{99.8}{1.61}\approx62\times.
\]

但这个比较并不严格公平：

- GPU 使用整卡 TDP，而不是 kernel 执行期间的实测功耗；
- FPGA 使用 Vivado 片上功耗估计，不是完整板卡输入功耗；
- FPGA 数值没有完整计入板卡、HBM、PCIe 等系统部分；
- 小 kernel 执行时 H100 不一定达到 350 W。

论文也按 U55C 的 150 W board TDP 给出保守估计。在整卡 TDP 对整卡 TDP的口径下，能效提升约为 $7.6\times\sim10.5\times$，这个范围更适合作为保守结论。

---

## 17. GPU baseline 的主要疑问

### 17.1 285 μs 对应的有效带宽异常低

只看约 4 MiB 的状态读写，在 2 TB/s 下的理想传输时间约为：

\[
\frac{4\ \text{MiB}}{2\ \text{TB/s}}
\approx2.1\ \mu s.
\]

但论文测得 H100 reference 延迟为 $285\ \mu s$，若粗略按 4.2 MB 计算，有效带宽仅：

\[
\frac{4.2\ \text{MB}}{285\ \mu s}
\approx14.7\ \text{GB/s},
\]

不到峰值带宽的 1%。这说明 285 μs 很可能不仅是物理 HBM 带宽造成的，还包含：

- 多个小 PyTorch/CUDA kernel 的 launch overhead；
- 中间张量构造与额外访存；
- recurrence 未充分融合；
- batch=1 下并行度和 occupancy 较低；
- 小矩阵/向量操作利用率低；
- reference implementation 的软件框架开销。

论文证明了 GDN recurrence 的理论算术强度很低，但没有充分证明高度融合、专门优化的 CUDA/Triton kernel 仍需要 285 μs。

### 17.2 “没有跨 head 并行性”的表述不准确

时间维度 $t\rightarrow t+1$ 确实存在严格 recurrence，但 32 个 value heads 的状态相互独立，可以在 GPU 上并行执行。论文正文把 batch=1 recurrence 描述为不具有 cross-head parallelism，这个说法至少不够严谨。

更合理的判断是：

- token 之间不能自由并行；
- head 之间可以并行；
- 但单 token 总工作量较小，未必能充分占满 GPU；
- baseline 的实际并行映射与融合质量决定了 285 μs 是否具有代表性。

### 17.3 GPU L2 并非绝对不能缓存状态

论文强调 H100 L2 是硬件管理 cache，不能像 BRAM 一样由程序显式保证状态跨 kernel 驻留，这一点成立。但从“不能保证”直接推导为“每 token 必然完整往返 HBM”仍然偏强。实际命中率还会受 kernel 融合、访问模式、并发工作集与 cache policy 影响。

FPGA BRAM 的确定性仍是优势，但公平 GPU baseline 应包括：

- 尽可能融合的单 kernel recurrence；
- CUDA Graph 或 persistent kernel 等降低 launch overhead 的方案；
- 对 L2/HBM 实际流量的 profiler 测量；
- 真实功耗而非只用 TDP。

---

## 18. 距离完整 Qwen3-Next 推理还有多远

### 18.1 只实现单个 GDN recurrence

论文假设 $q,k,v$ 和门控输入已经生成，没有包括：

- Q/K/V projection GEMM；
- output projection；
- RMSNorm/LayerNorm；
- FFN 或 MoE；
- softmax-attention layers；
- logits、sampling 等其他 decode 环节。

所以 $63.2\ \mu s$ 的准确含义是：

> 一个 GDN layer 的状态更新和输出阶段的估计延迟。

它不是完整模型生成一个 token 的延迟。

### 18.2 每个 GDN layer 都有独立状态

2 MiB 是单层状态。若共有 $L_{\text{GDN}}$ 个 GDN layers，逻辑状态容量为：

\[
2L_{\text{GDN}}\ \text{MiB}.
\]

U55C 虽有约 17.6 MB BRAM，但不能据此认为可轻松常驻 8 层：banking、buffer、端口复制和布局会产生额外物理开销。论文中单层设计在 $H_{\text{iter}}\ge4$ 时已经使用约 25% BRAM，且 $H_{\text{iter}}=4$ 已遇到 SLR 布线失败。

因此，多层状态若需要在 BRAM 与外部存储间切换，persistent-state 消除状态流量的优势会被显著削弱。

### 18.3 没有实现 prefill 状态初始化

真实 decode 开始时，状态不是零，而是 prefill 处理完整 prompt 后得到的：

\[
S_{\text{prompt}}.
\]

论文把 prefill 支持列为未来工作，没有展示：

- chunkwise parallel GDN prefill 如何映射到 FPGA；
- GPU prefill 生成的多层状态如何传给 FPGA；
- 首 token 前的初始化时间；
- 多请求状态如何分配、保存和切换。

### 18.4 没有评估真实 GPU–FPGA 协同通信

若 GPU 负责 projection、FFN/MoE 和其他层，而 FPGA 只负责 recurrence，那么每个 GDN layer、每个 token 都会发生：

\[
\text{GPU}\rightarrow(q,k,v,\text{gates})\rightarrow\text{FPGA},
\]

以及：

\[
\text{FPGA}\rightarrow o\rightarrow\text{GPU}.
\]

论文中的 AXI gmem 输入/输出不等于完整 GPU–FPGA PCIe 路径，也没有计入：

- GPU 到 FPGA 的数据搬运；
- FPGA 到 GPU 的返回；
- 跨设备同步；
- 每层、每 token 的调度开销；
- 多层串行切换。

因此，这项工作目前是一个 FPGA kernel/算子加速器，而不是已经闭环的 GPU–FPGA 协同推理系统。

---

## 19. 与“什么算子真正适合 FPGA”的关系

此前对 GPU–FPGA 协同的判断标准是：

1. FPGA 应处理 GPU 结构上不擅长的工作，而不只是 GPU 利用率暂时较低的普通 GEMV；
2. FPGA 的外部带宽并不高，必须发挥片上状态、定制数据流或非规则控制优势；
3. 计算边界应尽量干净，不能为了一个小算子在每层、每 token 频繁跨 PCIe；
4. 需要和优化后的 GPU kernel 比较，而不是只和框架 reference 比较。

这篇论文在**算子层面**比“让 FPGA 执行普通权重 GEMV”更符合这些标准。它的关键收益来自：

\[
\text{persistent recurrent state}
+\text{explicit BRAM banking}
+\text{custom fused dataflow},
\]

而不是更高的外部带宽。

但在**系统层面**，切分边界仍不干净：

- recurrence 嵌在每个 GDN layer 内部；
- 输入依赖 GPU 上的 projection；
- 输出又要立即进入后续 projection/FFN；
- 这种跨设备边界可能每层、每 token 重复出现。

因此最准确的评价是：

> 论文找到了一个真正适合 FPGA 片上状态驻留的算子，但尚未找到足够干净的完整 GPU–FPGA 系统切分边界。

---

## 20. 论文贡献、局限与最终评价

### 20.1 主要贡献

1. 指出线性注意力虽然消除了随上下文增长的 KV cache，却引入了每 token 必须完整读写固定状态矩阵的新瓶颈；
2. 利用 FPGA BRAM 的显式、持久存储，使 2 MiB 单层 GDN 状态跨 token 常驻片上；
3. 通过
   \[
   S_t^{\mathsf T}q_t
   =g_tS_{t-1}^{\mathsf T}q_t
   +(q_t^{\mathsf T}k_t)\Delta v_t
   \]
   将状态遍历由三次减少为两次；
4. 利用 2:1 GVA 共享 q/k，并在多个 value heads 间展开并行；
5. 用 BRAM banking、tiled II=1 accumulation 和 Prepare–Compute–Store dataflow 构造定制数据通路。

### 20.2 已证明的窄结论

论文较可靠地支持下面的结论：

> 对单层、FP32、batch=1、decode-only 的 GDN recurrence，如果该层 2 MiB 状态能够长期驻留 FPGA BRAM，那么定制数据流可以消除状态外存往返，并优于论文采用的 H100 PyTorch reference。

### 20.3 尚未充分证明的结论

论文没有充分证明：

- 完整 Qwen3-Next 推理可获得 $4.5\times$ 加速；
- 多个 GDN layers 的状态都能同时常驻 BRAM；
- $H_{\text{iter}}=8$ 可以完成布局布线并达到 300 MHz；
- 高度融合的优化 GPU kernel 仍然需要 $285\ \mu s$；
- 计入 GPU–FPGA 通信与同步后仍有明显端到端收益；
- $60\times$ 是公平的系统级能效提升；
- prefill、多请求和状态切换可以低成本支持。

### 20.4 总体评价

这篇论文的核心 idea 很有价值，也是 FPGA 与新型 LLM 架构结合中较“对路”的方向之一：它没有试图在外部带宽上击败 GPU，而是利用 FPGA 可控的片上状态把外部状态流量直接消掉。

但实验系统性仍然偏弱，尤其是 GPU baseline、布局布线完成度、多层状态容量和跨设备通信。阅读其结果时，应把三个层次明确分开：

| 层次 | 结论 |
|---|---|
| 算法性质 | GDN decode 状态访问密集、算术强度低，成立 |
| FPGA 算子设计 | persistent BRAM + 两遍状态数据流有效，成立 |
| 完整系统优势 | 对优化 GPU 和端到端混合 LLM 是否仍显著占优，尚未证明 |

最终可以将它视为一个有启发性的算子级原型，而不能直接视为完整 Qwen3-Next 或 GPU–FPGA 协同系统已经实现 $4.5\times/60\times$ 提升的证据。

---

## 21. 关键公式速查

### 普通线性注意力

\[
S_t=S_{t-1}+k_tv_t^{\mathsf T},
\qquad
o_t=S_t^{\mathsf T}q_t.
\]

### 状态作为线性映射

\[
S^{\mathsf T}x
=\sum_i(k_i^{\mathsf T}x)v_i.
\]

### Gated DeltaNet

\[
r_t=S_{t-1}^{\mathsf T}k_t,
\]

\[
\Delta v_t=\beta_t(v_t-r_t),
\]

\[
S_t=g_tS_{t-1}+k_t\Delta v_t^{\mathsf T},
\]

\[
o_t=\frac1{\sqrt d}S_t^{\mathsf T}q_t.
\]

### 两遍状态访问的代数重写

\[
S_t^{\mathsf T}q_t
=g_tS_{t-1}^{\mathsf T}q_t
+(q_t^{\mathsf T}k_t)\Delta v_t.
\]

### 延迟模型

\[
L
=\frac{h_v}{H_{\text{iter}}}T_{\text{iter}}
+T_{\text{load}}.
\]
