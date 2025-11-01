# aclnnQuantMatmul

## 功能描述

### 算子功能
该Ascend C算子用于执行量化矩阵乘法。它接收两个 `int8` 类型的二维矩阵 `a` 和 `b`，以及两个 `float16` 类型的一维缩放因子张量 `scale` 和 `per_token_scale`，通过高精度累加和反量化过程，最终输出 `float16` 类型的矩阵。该算子广泛应用于量化神经网络（特别是大语言模型）的推理计算中，以实现性能加速和内存优化。

### 计算公式
假设输入张量分别为 $A$（维度为 $m \times k$）、$B$（维度为 $k \times n$）、缩放因子 $S$（维度为 $n$）和逐token缩放因子 $P$（维度为 $m$）。算子首先计算 $A$ 和 $B$ 的整数矩阵乘积，得到一个中间累加结果 $Acc$（维度为 $m \times n$）。

$$Acc_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$

随后，对累加结果进行反量化，与两个缩放因子相乘，得到最终输出张量 $D$（维度为 $m \times n$）。计算公式如下：

$$D_{ij} = Acc_{ij} \times S_j \times P_i$$

其中，$i$ 的范围是从 $1$ 到 $m$，$j$ 的范围是从 $1$ 到 $n$。$D_{ij}$ 表示输出矩阵 $D$ 中第 $i$ 行第 $j$ 列的元素值。

### 计算过程与类型转换
为了在整数乘加过程中避免溢出，并保证最终浮点结果的精度，算子内部采用了严格的类型转换和计算流程：

1.  算子接收两个 `int8` 类型的输入张量 `a` 和 `b`，以及两个 `float16` 类型的缩放张量 `scale` 和 `per_token_scale`。
2.  在执行矩阵乘法之前，输入张量 `a` 和 `b` 被提升为 `int32` 类型。
3.  使用 `int32` 类型执行乘加累积操作，得到 `int32` 类型的中间结果 `accumulator`。
4.  将 `int32` 类型的 `accumulator`、`float16` 类型的 `scale` 和 `per_token_scale` 均转换为 `float32` 类型。
5.  执行元素级乘法，将 `float32` 的累加结果与经过广播（Broadcasting）的两个缩放因子相乘。
6.  最后，将计算得到的 `float32` 结果张量转换回 `float16` 类型，作为最终输出。

## 接口定义

### 算子原型定义接口
#### Input
- a：Device侧的aclTensor，公式中的A，数据类型支持int8，维度支持2维，数据格式支持ND。
- b：Device侧的aclTensor，公式中的B，数据类型支持int8，维度支持2维，数据格式支持ND。
- scale：Device侧的aclTensor，公式中的S，数据类型支持float16，维度支持1维，数据格式支持ND。
- per_token_scale：Device侧的aclTensor，公式中的P，数据类型支持float16，维度支持1维，数据格式支持ND。
#### Output
- d：Device侧的aclTensor，公式中的D，数据类型支持float16，维度支持2维，数据格式支持ND。
#### Attr
- 无

## 约束与限制
  * 输入张量 `a` 和 `b` 的数据类型必须为 `int8`。
  * 输入张量 `scale` 和 `per_token_scale` 的数据类型必须为 `float16`。
  * 输入张量 `a` 和 `b` 必须为二维矩阵，`scale` 和 `per_token_scale` 必须为一维向量。
  * `a` 的第二个维度（列数）必须与 `b` 的第一个维度（行数）相等。
  * `scale` 张量的长度必须等于 `b` 的第二个维度（列数）。
  * `per_token_scale` 张量的长度必须等于 `a` 的第一个维度（行数）。
  * 所有输入张量的数据格式只支持ND。