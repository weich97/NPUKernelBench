# aclnnGroupedMatmulSliceM

## 功能描述

### 算子功能
该Ascend C算子用于执行一种分组矩阵乘法。它根据一个分组列表 `groupList` 对二维输入张量 `a` 的行进行切片，然后将每个切片得到的子矩阵与三维输入张量 `b` 中对应的矩阵进行标准的矩阵乘法运算。该算子常用于处理批处理数据中各样本尺寸不一的场景，例如在自然语言处理中对不同长度的序列进行并行计算。

### 计算公式
假设输入张量 `a` 的维度为 $(M, K)$，输入张量 `b` 的维度为 $(\text{groupCount}, K, N)$，以及一个定义了分组边界的一维张量 `groupList`，其大小为 $\text{groupCount}$。

`groupList` 定义了如何将 `a` 按行切分成 $\text{groupCount}$ 个子矩阵 $A_0, A_1, \dots, A_{\text{groupCount}-1}$。设 `groupList` 为 $[g_0, g_1, \dots, g_{\text{groupCount}-1}]$，并定义 $g_{-1} = 0$，则第 $i$ 组（$i$ 从 0 到 $\text{groupCount}-1$）的行数 $m_i$ 为：

$$m_i = g_i - g_{i-1}$$

子矩阵 $A_i$ 是从 `a` 中抽取的、维度为 $(m_i, K)$ 的矩阵。同时，从 `b` 中取出第 $i$ 个矩阵 $B_i$，其维度为 $(K, N)$。

对于每一个组 $i$，计算其结果矩阵 $C_i = A_i \times B_i$，其维度为 $(m_i, N)$。$C_i$ 中的元素计算公式如下：

$$(C_i)_{pq} = \sum_{r=1}^{K} (A_i)_{pr} (B_i)_{rq}$$

其中，$p$ 的范围是从 $1$ 到 $m_i$，$q$ 的范围是从 $1$ 到 $N$。

最终，所有分组的计算结果 $C_0, C_1, \dots, C_{\text{groupCount}-1}$ 会被分别展平（Flatten）并沿主轴拼接（Concatenate），形成一个一维的输出张量 `c`。

$$c = \text{concat}(\text{flatten}(C_0), \text{flatten}(C_1), \dots, \text{flatten}(C_{\text{groupCount}-1}))$$

### 计算过程与类型转换
为了在执行大规模累加操作时保持较高的数值精度，并有效防止数据溢出，该算子在内部计算过程中采用了高精度累加的策略。具体流程如下：

1.  算子接收数据类型为 `float16` 的输入张量 `a` 和 `b`，以及 `int64` 类型的分组张量 `groupList`。
2.  根据 `groupList` 的值，将张量 `a` 逻辑上切分为多个子矩阵。
3.  对每个分组，执行一次矩阵乘法。在乘加计算时，内部的累加器（Accumulator）会使用 `float32` 类型。也就是说，`float16` 的乘积结果会先转换为 `float32`，然后再进行累加。
4.  所有分组的累加计算完成后，得到多个 `float32` 类型的结果矩阵。
5.  最后，将每个 `float32` 的结果矩阵转换回 `float16` 类型，展平并拼接成最终的一维输出张量。

## 接口定义

### 算子原型定义接口
#### Input
- a：Device侧的aclTensor，公式中的A，数据类型支持float16，维度为2维，数据格式支持ND。
- b：Device侧的aclTensor，公式中的B，数据类型支持float16，维度为3维，数据格式支持ND。
- groupList：Device侧的aclTensor，用于对输入`a`进行分组的依据，数据类型支持int64，维度为1维，数据格式支持ND。
#### Output
- c：Device侧的aclTensor，公式中的c，数据类型支持float16，维度为1维，数据格式支持ND。
#### Attr
- 无

## 约束与限制
  * 输入张量 `a` 和 `b` 的数据类型当前仅支持 `float16`。
  * 输入张量 `groupList` 的数据类型必须为 `int64`。
  * 输入张量 `a` 必须为2维，`b` 必须为3维，`groupList` 必须为1维。
  * `a` 的第二个维度（列数）必须与 `b` 的第二个维度（列数）相等。
  * `b` 的第一个维度（批次大小）必须与 `groupList` 的第一个维度（元素个数）相等。
  * `groupList` 必须是一个单调递增的序列，其最后一个元素的值必须等于 `a` 的第一个维度的大小（总行数）。
  * 所有输入张量的数据格式只支持ND。