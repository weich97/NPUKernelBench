# aclnnGemv

## 功能描述

### 算子功能
该Ascend C算子用于执行通用的矩阵向量乘法（General Matrix-Vector Multiplication, GEMV）运算。它实现了将一个矩阵与一个向量相乘，并将结果与另一个经过缩放的向量相加的功能。此算子是基础线性代数子程序（BLAS）库中的核心操作之一，在科学计算和机器学习领域有广泛应用。

### 计算公式
假设输入矩阵为 $A$（维度为 $m \times n$），输入向量为 $x$（维度为 $n$）和 $y$（维度为 $m$），输入标量为 $\alpha$ 和 $\beta$，则输出向量 $out$（维度为 $m$）的计算公式如下：

$$out = \alpha \cdot (A \cdot x) + \beta \cdot y$$

具体到每个元素的计算方式为：

$$out_{i} = \alpha \sum_{j=1}^{n} (A_{ij} \cdot x_{j}) + \beta \cdot y_{i}$$

其中，$i$ 的范围是从 $1$ 到 $m$。$out_{i}$ 表示输出向量 $out$ 中的第 $i$ 个元素。

### 计算过程与类型转换
该算子的所有输入张量（`a`, `x`, `y`）、属性（`alpha`, `beta`）和输出张量（`out`）的数据类型均为 `float32`。整个计算过程完全在 `float32` 精度下进行，不涉及任何内部数据类型提升或转换。

## 接口定义

### 算子原型定义接口
#### Input
- a：Device侧的aclTensor，公式中的矩阵A，数据类型支持float32，维度支持2维，数据格式支持ND。
- x：Device侧的aclTensor，公式中的向量x，数据类型支持float32，维度支持1维，数据格式支持ND。
- y：Device侧的aclTensor，公式中的向量y，数据类型支持float32，维度支持1维，数据格式支持ND。
#### Output
- out：Device侧的aclTensor，计算结果，数据类型为float32，维度为1维，数据格式支持ND。
#### Attr
- alpha：必需的float类型属性，公式中的缩放系数 $\alpha$。
- beta：必需的float类型属性，公式中的缩放系数 $\beta$。

## 约束与限制
  * 所有输入张量 `a`、`x`、`y` 的数据类型当前仅支持 `float32`。
  * 输入张量 `a` 必须为二维矩阵。
  * 输入张量 `x` 和 `y` 必须为一维向量。
  * 输入矩阵 `a` 的第二个维度（列数）必须等于向量 `x` 的维度。
  * 输入矩阵 `a` 的第一个维度（行数）必须等于向量 `y` 的维度。
  * 所有输入张量的数据格式只支持ND。