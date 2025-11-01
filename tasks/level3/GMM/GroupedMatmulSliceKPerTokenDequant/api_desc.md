# aclnnGroupedMatmulSliceKPerTokenDequant

## 功能描述

### 算子功能
该Ascend C算子用于执行一种特殊的分组量化矩阵乘法。它接收两个`int8`类型的输入矩阵`a`和`b`，并根据`groupList`张量定义的边界，将它们的内积维度（K维度）分割成多个组。算子对每个组分别执行矩阵乘法，然后使用`scale`和`per_token_scale`两个量化尺度张量对结果进行反量化。

该算子常用于大规模语言模型（LLM）等场景中的量化推理，能有效降低内存占用和计算量。

### 计算公式
假设输入张量分别为：
- $A$ (对应输入 `a`): 维度为 $m \times k$，数据类型为 `int8`
- $B$ (对应输入 `b`): 维度为 $k \times n$，数据类型为 `int8`
- $S$ (对应输入 `scale`): 维度为 $\text{groupCount} \times n$，数据类型为 `bfloat16`
- $P$ (对应输入 `per_token_scale`): 维度为 $\text{groupCount} \times m$，数据类型为 `bfloat16`
- $G$ (对应输入 `groupList`): 维度为 $\text{groupCount}$，数据类型为 `int64`，定义了K维度的切分点。

我们定义 $G_{-1} = 0$。对于第 $i$ 个分组（$i$ 从 $0$ 到 $\text{groupCount}-1$），其在K维度上的起始索引为 $s_i = G_{i-1}$，结束索引为 $e_i = G_i$。该分组的K维度大小为 $k_i = e_i - s_i$。

算子将对每个分组执行以下计算：

1.  **切片 (Slicing)**:
    - 从 $A$ 中切出子矩阵 $A_i = A[:, s_i:e_i]$，其维度为 $m \times k_i$。
    - 从 $B$ 中切出子矩阵 $B_i = B[s_i:e_i, :]$，其维度为 $k_i \times n$。

2.  **矩阵乘法与反量化 (Matrix Multiplication and Dequantization)**:
    - 对每个分组计算结果矩阵 $C_i$（维度为 $m \times n$）。其任意元素 $(C_i)_{rc}$ 的计算公式如下：
    $$ (C_i)_{rc} = \left( \sum_{p=0}^{k_i-1} (A_i)_{r,p} \cdot (B_i)_{p,c} \right) \cdot (S_i)_c \cdot (P_i)_r $$
    其中，$r$ 和 $c$ 分别是行索引和列索引。$(S_i)_c$ 是 `scale` 张量第 $i$ 行第 $c$ 列的元素，$(P_i)_r$ 是 `per_token_scale` 张量第 $i$ 行第 $r$ 列的元素。

3.  **结果拼接 (Concatenation)**:
    - 将所有分组的结果矩阵 $C_0, C_1, \dots, C_{\text{groupCount}-1}$ 分别进行扁平化（Flatten）。
    - 将所有扁平化后的一维向量按顺序拼接，形成最终的输出张量 $C$。

### 计算过程与类型转换
为保证计算精度，算子内部采用高精度数据类型进行中间计算，具体流程如下：

1.  接收 `int8` 类型的输入 `a` 和 `b`，以及 `bfloat16` 类型的 `scale` 和 `per_token_scale`。
2.  对于每个分组，在执行矩阵乘法（即求和）之前，将切片后的子矩阵 $A_i$ 和 $B_i$ 从 `int8` 转换为 `int32`。
3.  矩阵乘法的累加过程在 `int32` 类型下完成。
4.  乘法结果（`int32`）与 `scale`、`per_token_scale`（`bfloat16`）相乘进行反量化。此步骤的计算会在 `float32` 精度下执行以保证数值稳定性。
5.  得到 `float32` 类型的分组结果后，将其转换为 `bfloat16`。
6.  所有分组的 `bfloat16` 结果被扁平化并拼接，形成最终的 `bfloat16` 输出张量。

## 接口定义

### 算子原型定义接口
#### Input
- a：Device侧的aclTensor，公式中的A。数据类型支持int8，维度支持2维，数据格式支持ND。
- b：Device侧的aclTensor，公式中的B。数据类型支持int8，维度支持2维，数据格式支持ND。
- scale：Device侧的aclTensor，公式中的S，用于反量化的尺度张量。数据类型支持bfloat16，维度支持2维，数据格式支持ND。
- per_token_scale：Device侧的aclTensor，公式中的P，用于反量化的逐token尺度张量。数据类型支持bfloat16，维度支持2维，数据格式支持ND。
- groupList：Device侧的aclTensor，公式中的G，定义K维度的累积切分点。数据类型支持int64，维度支持1维，数据格式支持ND。
#### Output
- c：Device侧的aclTensor，公式中的C，为所有分组结果扁平化后拼接而成的一维张量。数据类型支持bfloat16，维度支持1维，数据格式支持ND。
#### Attr
- 无

## 约束与限制
  * 输入张量 `a` 和 `b` 的数据类型必须为 `int8`。
  * 输入张量 `scale`、`per_token_scale` 以及输出张量 `c` 的数据类型必须为 `bfloat16`。
  * 输入张量 `groupList` 的数据类型必须为 `int64`。
  * 维度约束：
    * `a` 必须为二维张量，形状为 `(m, k)`。
    * `b` 必须为二维张量，形状为 `(k, n)`。
    * `scale` 必须为二维张量，形状为 `(groupCount, n)`。
    * `per_token_scale` 必须为二维张量，形状为 `(groupCount, m)`。
    * `groupList` 必须为一维张量，形状为 `(groupCount)`。
  * `a` 的第二维度（k）必须与 `b` 的第一维度（k）相等。
  * `groupList` 中的值必须是单调递增的，且其最后一个元素的值必须等于 `k`。
  * 所有输入张量的数据格式只支持ND。