# aclnnGemm

## 功能描述

### 算子功能
该Ascend C算子用于执行通用的矩阵乘法和加法运算（General Matrix Multiplication, GEMM），是BLAS（Basic Linear Algebra Subprograms）库中的核心操作之一。它计算 `alpha * (A * B) + beta * C` 的结果，其中 `A`、`B` 和 `C` 是输入矩阵，`alpha` 和 `beta` 是标量系数。此算子在深度学习中被广泛应用于实现全连接层、卷积层（通过im2col转换）以及注意力机制等关键模块。

### 计算公式
假设输入张量分别为 $A$（维度为 $m \times k$）、$B$（维度为 $k \times n$）和 $C$（维度为 $m \times n$），标量属性为 $\alpha$ 和 $\beta$。则输出张量 $D$（维度为 $m \times n$）的计算公式如下：

$$D = \alpha \cdot (A \cdot B) + \beta \cdot C$$

其元素级的计算公式可以表示为：

$$D_{ij} = \alpha \sum_{p=1}^{k} (A_{ip} \cdot B_{pj}) + \beta \cdot C_{ij}$$

其中，$i$ 的范围是从 $1$ 到 $m$，$j$ 的范围是从 $1$ 到 $n$。$D_{ij}$ 表示输出矩阵 $D$ 中第 $i$ 行第 $j$ 列的元素值。

### 计算过程与类型转换
该算子的计算流程严格遵循GEMM的定义：

1.  算子接收三个数据类型为 `float32` 的输入张量 `a`、`b`、`c`，以及两个 `float32` 类型的标量属性 `alpha` 和 `beta`。
2.  首先，执行 `alpha` 与矩阵 `a` 的标量乘法，或将 `alpha` 融入后续的矩阵乘法累加过程中。
3.  接着，计算 `a` 和 `b` 的矩阵乘法。
4.  然后，执行 `beta` 与矩阵 `c` 的标量乘法。
5.  最后，将步骤2和3的结果与步骤4的结果进行逐元素相加，得到最终的输出张量。
6.  整个计算过程中，所有张量的数据类型和中间累加过程均使用 `float32`，不涉及数据类型的提升或转换。最终输出的数据类型也为 `float32`。

## 接口定义

### 算子原型定义接口
#### Input
- a：Device侧的aclTensor，公式中的A，数据类型支持float32，维度支持2维，数据格式支持ND。
- b：Device侧的aclTensor，公式中的B，数据类型支持float32，维度支持2维，数据格式支持ND。
- c：Device侧的aclTensor，公式中的C，数据类型支持float32，维度支持2维，数据格式支持ND。
#### Output
- d：Device侧的aclTensor，公式中的D，数据类型为float32，维度为2维，数据格式支持ND。
#### Attr
- alpha：float类型的标量，公式中的α。
- beta：float类型的标量，公式中的β。

## 约束与限制
*   输入张量 `a`、`b` 和 `c` 的数据类型当前仅支持 `float32`。
*   输入张量 `a`、`b` 和 `c` 必须为二维矩阵。
*   输入张量的维度必须满足矩阵乘法和加法的要求：
    *   `a` 的第二个维度（列数）必须与 `b` 的第一个维度（行数）相等。
    *   `c` 的维度必须与 `a` 和 `b` 矩阵乘法结果的维度相同，即 `(m, n)`。
*   输入张量的数据格式只支持ND。
*   属性 `alpha` 和 `beta` 必须为 `float` 类型。