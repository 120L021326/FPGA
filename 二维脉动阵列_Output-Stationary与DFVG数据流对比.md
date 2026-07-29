# 两种二维脉动阵列数据流：Output-Stationary 与 DFVG Weight-Stationary

> [返回主线：FPGA 在 LLM 领域的应用调研](./FPGA调研.md)

## 1. 问题背景

二维脉动阵列描述的是 PE（Processing Element）按照二维网格排列，并让数据在相邻 PE 之间有规律地传递。  
但“二维脉动阵列”只描述了计算单元的拓扑，并不唯一决定数据流。

同一个二维 PE 阵列可以采用不同的数据驻留策略：

- **Output-Stationary（输出驻留）**：partial sum/ACC 固定在 PE 中，activation 和 weight 流动；
- **Weight-Stationary（权重驻留）**：weight 固定在 PE 中，activation 和 partial sum 流动；
- 还可以采用 input-stationary、row-stationary 等其他形式。

本文重点比较：

1. ACC 固定在 PE 中的 output-stationary 二维脉动阵列；
2. DFVG 论文采用的、接近 weight-stationary 的二维脉动阵列。

---

## 2. 统一矩阵乘法表示

将矩阵乘法写成：

$$
Y=XW,
$$

其中：

$$
X\in\mathbb{R}^{B\times K},
\qquad
W\in\mathbb{R}^{K\times N},
\qquad
Y\in\mathbb{R}^{B\times N}.
$$

各维度含义如下：

| 符号 | 含义 |
|---|---|
| \(B\) | 同时处理的 token、request 或 speculative branch 数量 |
| \(K\) | 输入 hidden size，也就是点积的归约维度 |
| \(N\) | 输出 hidden size 或输出通道数 |

每个输出元素为：

$$
Y_{b,n} =
\sum_{k=0}^{K-1}
X_{b,k}W_{k,n}.
$$

两种阵列最终计算的是同一个公式，区别在于：

- 哪一个维度映射到阵列的空间；
- 哪一个维度沿时间展开；
- weight、activation 和 partial sum 分别驻留还是流动。

---

## 3. 形式一：ACC 固定在 PE 中的 Output-Stationary 阵列

### 3.1 基本思想

在 output-stationary 数据流中，一个 PE 在一段计算时间内负责一个输出元素：

$$
Y_{b,n}.
$$

PE 内部保存对应的累加器：

$$
ACC_{b,n}.
$$

随着 \(k\) 维的数据逐周期进入阵列，PE 执行：

$$
ACC_{b,n}
\leftarrow
ACC_{b,n}
+
X_{b,k}W_{k,n}.
$$

遍历完整个 \(K\) 维后：

$$
ACC_{b,n} =
\sum_{k=0}^{K-1}
X_{b,k}W_{k,n}
=Y_{b,n}.
$$

因此，“output-stationary”中的 stationary 指的是：

> 当前输出元素对应的 partial sum 长时间保存在同一个 PE 中，直到完整点积结束。

### 3.2 典型数据行为

| 数据 | 行为 |
|---|---|
| Activation \(X_{b,k}\) | 在阵列中传播或广播 |
| Weight \(W_{k,n}\) | 在阵列中传播或广播 |
| Partial sum/ACC | 固定在负责该输出的 PE 中 |
| \(K\) 维 | 主要沿时间展开 |
| 最终输出 | 累加完成后从 PE 读出 |

可以将一个 PE 的行为概括为：

```text
activation ──► [ PE: local ACC += activation × weight ] ──► activation
                         ▲
                         │
                       weight
```

其中 weight 的具体传播方向取决于实现，但 partial sum 不会为了完成同一个点积而逐级流向其他 PE。

### 3.3 阵列通常映射的维度

output-stationary 阵列一般将输出矩阵的一个 tile 映射到 PE：

$$
B_{\text{tile}}\times N_{\text{tile}}.
$$

第 \((b,n)\) 个 PE 保存：

$$
ACC_{b,n}.
$$

而输入 hidden/K 维沿时间展开：

$$
k=0,1,2,\ldots,K-1.
$$

所以即使：

$$
K=4096
\quad\text{或}\quad
K=8192,
$$

也不要求阵列在物理空间中具有 4096 或 8192 行。PE 可以在 4096 或 8192 个计算步骤中不断更新同一个 ACC。

“hidden state 可以无限大”更准确的说法是：

> \(K\) 维不受阵列空间尺寸的直接限制，可以映射到时间维；只要控制器、数据供给和累加精度允许，PE 就可以继续累加。

### 3.4 Decode 场景下的映射

普通自回归 decode 通常每个 request 每步只有一个新 token：

$$
B=1.
$$

在 speculative decoding 或 DFVG 中，多个 branch/node 可以形成：

$$
B>1.
$$

但 output-stationary 阵列的空间维并不一定只是 query sequence length，而通常是：

$$
B_{\text{tile}}\times N_{\text{tile}}.
$$

因此：

- \(B_{\text{tile}}\) 可以对应并行 token、request 或 branch；
- \(N_{\text{tile}}\) 对应一部分输出 hidden 维；
- \(K\) 沿时间完整流过；
- 如果 \(N>N_{\text{tile}}\)，仍然需要对输出维 \(N\) 进行切分。

所以 output-stationary 并非完全不需要 tiling，而是：

> 它通常不需要把 \(K\) 维的 partial sum 在多个 PE 或 output buffer 之间反复合并，但仍可能切分 \(B\) 维和 \(N\) 维。

### 3.5 主要优点

#### 3.5.1 对大 hidden size 更灵活

\(K\) 维可以完全沿时间遍历，不要求阵列空间覆盖 hidden size。

#### 3.5.2 partial sum 不需要在 PE 之间流动

同一个输出的中间结果始终留在本地：

$$
ACC_{b,n}^{(0)}
\rightarrow
ACC_{b,n}^{(1)}
\rightarrow
\cdots
\rightarrow
ACC_{b,n}^{(K)}.
$$

这可以减少 partial sum 的阵列内通信。

#### 3.5.3 不需要频繁产生 hidden tile 的外部中间结果

如果 PE 能够连续处理完整的 \(K\) 维，就不需要每处理一小段 hidden 维便把 partial sum 写入 output buffer，再在后续轮次读出并继续累加。

#### 3.5.4 适合 branch 数较小且上限明确的场景

如果：

$$
B\le B_{\text{tile}},
$$

阵列能够同时保存所有 branch 的 ACC，就可以让同一个权重广播给多个 branch：

$$
ACC_{1,n}\mathrel{+}=X_{1,k}W_{k,n},
$$

$$
ACC_{2,n}\mathrel{+}=X_{2,k}W_{k,n},
$$

$$
\cdots
$$

$$
ACC_{B,n}\mathrel{+}=X_{B,k}W_{k,n}.
$$

这样同样可以实现跨 branch 的权重复用。

### 3.6 主要缺点

#### 3.6.1 每个有效输出都需要本地累加状态

阵列需要维护：

$$
B_{\text{tile}}\times N_{\text{tile}}
$$

个本地 ACC。

如果累加精度高于输入精度，例如 BF16 乘法使用 FP32 或较宽定点累加，则每个 PE 都需要较宽的累加寄存器和相关控制逻辑。

#### 3.6.2 动态 branch 数可能造成 PE 空闲

如果阵列按最大 branch 数设计：

$$
B_{\max}\times N_{\text{tile}},
$$

但某一轮只有少量分支：

$$
B\ll B_{\max},
$$

则为 branch 维预留的很多 PE 会空闲。

#### 3.6.3 activation 和 weight 的分发网络可能更复杂

为了让不同 PE 在同一周期获得正确的：

$$
X_{b,k}
\quad\text{和}\quad
W_{k,n},
$$

通常需要规则的行列传播、广播或多级分发网络。随着阵列扩大，广播扇出、布线拥塞和时序收敛可能成为问题。

---

## 4. 形式二：DFVG 的 Weight-Stationary + Partial-Sum 流动阵列

### 4.1 基本思想

DFVG 论文中的 systolic PE array 更接近 weight-stationary 数据流：

- weight 加载到 PE 的本地 weight buffer，并在一段计算期间保持不变；
- activation 沿水平方向在 PE 之间传播；
- partial sum 沿垂直方向在 PE 之间传播；
- 多个 core 的结果经过加法树合并；
- 跨多个计算轮次的结果继续在 output buffer 中累加。

单个 PE 执行：

$$
p_{\text{out}} =
p_{\text{in}}+a\times w_{\text{local}}.
$$

其数据行为可概括为：

```text
                         partial sum in
                                │
                                ▼
activation in ──► [ PE: p + a × local weight ] ──► activation out
                                │
                                ▼
                        partial sum out
```

### 4.2 各类数据的行为

| 数据 | 行为 |
|---|---|
| Activation \(X_{b,k}\) | 水平方向传播 |
| Weight \(W_{k,n}\) | 驻留在 PE 的本地 weight buffer |
| Partial sum \(p\) | 垂直方向逐级传播 |
| 跨 tile 累加结果 | 保存在 output buffer 中 |
| 最终输出 | 从阵列底部输出，经加法树和 output buffer 合并 |

### 4.3 阵列空间映射

这种设计更接近把阵列空间映射为：

$$
K_{\text{tile}}\times N_{\text{tile}}.
$$

每个 PE 保存一个或一组权重：

$$
W_{k,n}.
$$

对于当前 branch \(b\)，partial sum 沿 \(K_{\text{tile}}\) 个 PE 逐级累加：

$$
p_{r,n} =
p_{r-1,n}
+
X_{b,k_0+r}W_{k_0+r,n}.
$$

经过阵列的一次局部归约后，只得到一个 hidden tile 的结果：

$$
p_{b,n}^{(t)} =
\sum_{k=tK_{\text{tile}}}^{(t+1)K_{\text{tile}}-1}
X_{b,k}W_{k,n}.
$$

### 4.4 为什么必须切分 hidden/K 维

真实模型通常具有很大的 hidden size：

$$
K=4096,\ 8192,\ \ldots
$$

而 FPGA 不可能构造具有数千行 PE、并让整个 \(K\) 维一次性完全展开的阵列。因此：

$$
K_{\text{tile}}\ll K.
$$

完整结果需要合并多个 hidden tile：

$$
Y_{b,n} =
\sum_{t=0}^{\lceil K/K_{\text{tile}}\rceil-1}
p_{b,n}^{(t)}.
$$

例如：

$$
K=4096,
\qquad
K_{\text{tile}}=16.
$$

则 hidden/K 维需要被切成：

$$
\frac{4096}{16}=256
$$

轮。

阵列每轮只完成 16 个乘加项对应的局部归约，之后必须在 output buffer 中继续累加这 256 轮的结果。

因此，对 DFVG 数据流的以下判断是成立的：

> 由于它将一部分 \(K\) 维映射到阵列空间，而阵列无法覆盖完整 hidden size，所以必然需要对 \(K\) 维反复切分，并产生跨轮 partial-sum 累加。

---

## 5. DFVG 为什么仍然采用这种数据流

DFVG 的选择并不意味着 partial-sum 流动在所有矩阵乘法中都更优。它主要针对以下工作负载特征：

$$
\text{动态多 branch}
+
\text{同一权重被多个 branch 复用}.
$$

### 5.1 权重驻留并被多个 branch 连续复用

同一个本地权重：

$$
W_{k,n}
$$

可以连续处理多个 branch activation：

$$
X_{1,k},X_{2,k},\ldots,X_{B,k}.
$$

对应计算为：

$$
W_{k,n}X_{1,k},
\quad
W_{k,n}X_{2,k},
\quad
\ldots,
\quad
W_{k,n}X_{B,k}.
$$

加载一次 weight tile 后，可以让一组 speculative branches 依次进入阵列，摊薄权重读取成本。

### 5.2 branch 数可以沿时间变化

DFVG 的 token tree 是动态的，不同推测轮次中的有效 branch 数可能为：

$$
B=2,4,7,11,\ldots
$$

如果采用按最大 branch 数展开的 output-stationary 阵列：

$$
B_{\max}\times N_{\text{tile}},
$$

当实际 branch 数较少时，部分 PE 会空闲。

DFVG 将主要空间资源用于：

$$
K_{\text{tile}}\times N_{\text{tile}},
$$

并让 branch 沿时间进入阵列。这样，branch 数变化主要改变处理周期数，而不是直接改变有效 PE 的比例。

### 5.3 partial-sum 归约只使用邻近连接

partial sum 沿相邻 PE 传播：

$$
PE_0\rightarrow PE_1\rightarrow PE_2\rightarrow\cdots.
$$

这种结构具有以下实现优势：

- 连接规则；
- 通信距离短；
- 容易进行深流水；
- 不需要在阵列内部构造覆盖整个 \(K_{\text{tile}}\) 的高扇入组合加法树；
- 在大 FPGA 上更容易布线并满足较高时钟频率。

阵列填满后，可以流水化地连续接收新的 branch/token。

### 5.4 与“一 DSP 双 BF16 尾数乘法”自然匹配

DFVG 的 DSP packing 利用两个乘法共享同一权重：

$$
A_1W,
\qquad
A_2W.
$$

weight-stationary PE 内已经保存了：

$$
W.
$$

因此可以让两个 branch activation：

$$
A_1,\ A_2
$$

同时与该本地权重相乘，一次 DSP 整数尾数乘法产生两个独立的尾数乘积。

这并非只有 weight-stationary 才能实现，但本地共享权重使这种打包方式非常自然。

### 5.5 PE 内不必长期保存大量输出状态

DFVG 的 PE 主要保存：

- 本地权重；
- 当前流水级的 activation；
- 当前经过的 partial sum；
- 必要的流水寄存器。

长期的跨 hidden tile 累加集中在 output buffer 中，而不是让每个 PE 长期维护一个完整输出 ACC。

这降低了单个 PE 的输出状态复杂度，但代价是：

- output buffer 必须支持读—改—写或等价的跨轮累加；
- hidden tile 数量很大时，output buffer 的访问和累加开销会增加；
- partial sum 在阵列与 buffer 之间的移动也会产生额外能耗。

---

## 6. 两种阵列的数据流对比

| 对比项 | Output-Stationary：ACC 固定 | DFVG：Weight-Stationary + psum 流动 |
|---|---|---|
| PE 长期保存的数据 | 输出 ACC | 权重 |
| Activation | 在 PE 间传播或广播 | 水平传播 |
| Weight | 在 PE 间传播或广播 | 驻留在本地 weight buffer |
| Partial sum | 固定在 PE 内 | 垂直传播 |
| 主要空间映射 | \(B_{\text{tile}}\times N_{\text{tile}}\) | \(K_{\text{tile}}\times N_{\text{tile}}\) |
| \(K\)/hidden 维 | 主要沿时间完整展开 | 部分空间展开，其余切成多个 tile |
| \(B\)/branch 维 | 通常部分空间展开 | 更容易沿时间展开 |
| 大 hidden size | 更自然 | 需要多轮 hidden-tile 累加 |
| 动态 branch 数 | 可能导致为 branch 预留的 PE 空闲 | 更容易通过改变运行周期数适应 |
| 权重跨 branch 复用 | 依赖广播和可同时容纳的 branch 数 | 本地权重可被连续到达的 branch 复用 |
| 跨 \(K\) tile 的外部累加 | 通常不需要，或需求较少 | 需要在 output buffer 中完成 |
| PE 间归约通信 | 少，同一输出留在本地 | partial sum 逐 PE 传播 |
| 本地累加器资源 | 每个有效输出需要宽 ACC | 长期累加集中到 output buffer |
| 阵列内连接 | activation 与 weight 的分发可能较复杂 | activation 与 psum 的邻接传播较规则 |
| DSP 双 BF16 packing | 可以实现，但需组织共享权重 | 与本地共享权重自然匹配 |
| 通用性 | 对各种 \(K\) 更灵活 | 更针对动态多 branch 工作负载 |

---

## 7. 对 hidden 维切分问题的分析

### 7.1 对 DFVG 的批评为什么成立

DFVG 把有限的 PE 空间用于展开一部分 \(K\) 维：

$$
K_{\text{tile}}\times N_{\text{tile}}.
$$

当：

$$
K\gg K_{\text{tile}},
$$

就必须执行：

$$
T_K=\left\lceil\frac{K}{K_{\text{tile}}}\right\rceil
$$

轮计算，并将每轮的 partial sum 合并。

因此它牺牲了以下灵活性：

> 无法像 output-stationary 那样，让任意长度的 \(K\) 维直接在同一个 PE 的本地 ACC 中持续累加。

hidden size 越大，\(T_K\) 越大，以下代价越明显：

- output buffer 的累加次数增加；
- partial sum 的读写和传输增加；
- tile 启动与流水线填充开销增加；
- 如果 branch 数不足，权重驻留和流水化收益可能无法抵消切分成本。

### 7.2 Output-Stationary 也并非完全没有切分

output-stationary 一般仍然只能覆盖：

$$
B_{\text{tile}}\times N_{\text{tile}}.
$$

因此：

- 如果 \(B>B_{\text{tile}}\)，需要切分 batch/branch 维；
- 如果 \(N>N_{\text{tile}}\)，需要切分输出维；
- 本地 ACC 数量受到寄存器、DSP 和片上存储资源限制；
- 若权重无法高效广播，权重带宽仍可能成为瓶颈。

其优势并不是“不做任何 tiling”，而是：

> 对每个已映射到 PE 的输出元素，可以让完整 \(K\) 维沿时间累加，避免频繁输出 hidden-tile partial sums。

---

## 8. 什么时候 ACC 固定在 PE 中更好

当以下条件成立时，output-stationary 通常更有吸引力：

1. branch 数较小，而且最大值明确；
2. 阵列能够覆盖全部或大部分有效 branch；
3. hidden/K 维很大，例如 4096 或 8192；
4. 权重能够高效广播给不同 branch PE；
5. 每个 PE 有足够资源保存宽位 ACC；
6. 希望避免大量 hidden-tile partial sum；
7. branch 数变化不会导致严重的 PE 空闲；
8. output buffer 的反复累加带宽或能耗较高。

例如：

$$
B=8,
$$

阵列映射为：

$$
8\times N_{\text{tile}}.
$$

同一个权重：

$$
W_{k,n}
$$

广播到八个 branch PE，使其同时执行：

$$
ACC_{b,n}
\mathrel{+}=
X_{b,k}W_{k,n},
\qquad b=0,\ldots,7.
$$

随后让：

$$
K=4096
$$

沿时间流过。

这种映射能够同时获得：

- 权重跨 branch 复用；
- 本地 ACC；
- 不产生每个小 \(K\) tile 的外部 partial sum；
- 对大 hidden size 的自然支持。

如果 branch 数稳定且阵列可以容纳，这种设计可能比 DFVG 的数据流更灵活。

---

## 9. 什么时候 DFVG 的数据流更合适

DFVG 的 weight-stationary + partial-sum 流动更适合：

1. branch 数动态变化明显；
2. 不希望为最大 branch 数永久预留大量 PE；
3. 同一 weight tile 能够被许多 branch 连续复用；
4. 权重访问是主要瓶颈；
5. FPGA 布线和时钟频率比减少 hidden tiling 更重要；
6. 相邻 PE 的规则流水能够获得较高频率；
7. output buffer 足以高效承担跨 tile 累加；
8. 双 BF16 packing 能显著提高 DSP 有效吞吐；
9. branch 并行度足以摊薄阵列填充、hidden 切分和权重加载开销。

DFVG 的核心优化目标不是单独最小化 partial-sum 流动，而是联合利用：

$$
\boxed{
\text{权重驻留}
+
\text{动态多分支}
+
\text{跨 branch 权重复用}
+
\text{规则 systolic 流水}
+
\text{双 BF16 DSP packing}
}
$$

---

## 10. 最终判断

两种设计不存在脱离工作负载的绝对优劣。

### 10.1 Output-Stationary 的核心取舍

它将：

$$
B_{\text{tile}}\times N_{\text{tile}}
$$

映射到空间，把：

$$
K
$$

映射到时间。

因此对大 hidden size 更灵活，且同一输出的 partial sum 长时间保存在 PE 中；代价是需要较多本地 ACC 状态，并可能因动态 branch 数而产生 PE 空闲。

### 10.2 DFVG 数据流的核心取舍

它将：

$$
K_{\text{tile}}\times N_{\text{tile}}
$$

映射到空间，把动态 branch 更多地映射到时间。

因此能够让本地权重被不确定数量的 branch 连续复用，并获得规则的相邻 PE 流水；代价是大 hidden size 必须被切成很多轮，并在 output buffer 中合并 partial sums。

因此，对 DFVG 更准确的评价是：

> DFVG 的数据流是针对“动态多 branch、权重驻留和共享权重 DSP packing”定制的架构，而不是矩阵乘法最通用的数据流。它用 hidden/K 维切分和跨轮累加，换取动态 branch 适应性、权重跨 branch 复用以及更规则的 FPGA 流水与布线。

如果 branch 数较少且阵列能够覆盖全部 branch，同时权重广播能够高效实现，那么：

$$
\boxed{
ACC\text{ 固定在 PE}
+
K\text{ 维沿时间流动}
+
W\text{ 在 branch 间广播复用}
}
$$

很可能更灵活，并能够减少 output buffer 中的反复累加。

最终应通过下列参数共同判断：

$$
\boxed{
B,\ K,\ N,\ 
B_{\text{tile}},\ K_{\text{tile}},\ N_{\text{tile}},
\text{有效内存带宽},
\text{ACC 资源},
\text{branch 利用率},
\text{片上互连代价}
}
$$

而不能仅依据“二维脉动阵列”或“数据是否流动”判断哪一种实现更好。
