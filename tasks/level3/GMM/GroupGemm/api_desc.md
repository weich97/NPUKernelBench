# aclnnGroupGemm

## 功能描述

### 算子功能
该Ascend C算子用于执行一组独立的通用矩阵乘法（GEMM）运算。它专为批量处理尺寸可变的矩阵的场景而设计，例如在Transformer模型中处理不同序列长度的批处理输入时，能够实现高效计算。该算子接收由各组矩阵数据拼接而成的一维扁平化张量、以及对应的标量乘数，并生成一个包含所有计算结果的、拼接后的一维扁平化输出张量。

### 计算公式
该算子执行 `groupCount` 次独立的计算。对于第 $i$ 组运算（$i$ 从 $1$ 到 `groupCount`），其计算逻辑如下：

-   设 $A_i$ 为该组的输入矩阵，维度为 $(m_i, k_i)$
-   设 $B_i$ 为该组的输入矩阵，维度为 $(k_i, n_i)$
-   设 $C_i$ 为该组的输入矩阵，维度为 $(m_i, n_i)$
-   设 $\alpha_i$ 和 $\beta_i$ 为该组对应的标量乘数。

该组的输出矩阵 $D_i$ 通过以下公式计算得出：

$$D_i = \alpha_i \cdot (A_i \cdot B_i) + \beta_i \cdot C_i$$

算子的最终输出 `d` 是将所有组的计算结果矩阵 $D_i$ 逐一扁平化后，再拼接成一个一维张量：

$$d = \text{concat}(\text{flatten}(D_1), \text{flatten}(D_2), ..., \text{flatten}(D_{\text{groupCount}}))$$

### 计算过程与类型转换
为了在计算过程中确保数值的稳定性和精度，该算子采用了混合精度计算策略，具体流程如下：

1.  算子接收数据类型为 `float16` 的输入矩阵张量 `a`、`b`、`c`，以及数据类型为 `float32` 的标量乘数张量 `alpha` 和 `beta`。
2.  在处理每一组运算时，算子会根据 `mList`、`kList` 和 `nList` 属性，从扁平化的输入张量中内部重构出对应的矩阵切片（$A_i, B_i, C_i$）。
3.  在执行矩阵乘法和加法之前，`float16` 类型的矩阵数据 $A_i, B_i, C_i$ 会被提升至 `float32` 类型。
4.  完整的GEMM计算公式 $D_i = \alpha_i \cdot (A_i \cdot B_i) + \beta_i \cdot C_i$ 全程在 `float32` 精度下执行。
5.  每组的计算完成后，得到的 `float32` 结果矩阵 $D_i$ 将被转换回 `float16` 类型。
6.  最终，所有 `float16` 类型的结果矩阵将被扁平化并拼接，形成最终的一维输出张量。

## 接口定义

### 算子原型定义接口
#### Input
- a：Device侧的aclTensor。一个一维张量，包含了所有 $A_i$ 矩阵扁平化后拼接的数据。数据类型支持float16，数据格式支持ND。
- b：Device侧的aclTensor。一个一维张量，包含了所有 $B_i$ 矩阵扁平化后拼接的数据。数据类型支持float16，数据格式支持ND。
- c：Device侧的aclTensor。一个一维张量，包含了所有 $C_i$ 矩阵扁平化后拼接的数据。数据类型支持float16，数据格式支持ND。
- alpha：Device侧的aclTensor。一个一维张量，包含了每组计算对应的标量乘数 $\alpha_i$。数据类型支持float32，数据格式支持ND。
- beta：Device侧的aclTensor。一个一维张量，包含了每组计算对应的标量乘数 $\beta_i$。数据类型支持float32，数据格式支持ND。
#### Output
- d：Device侧的aclTensor。一个一维张量，包含了所有结果矩阵 $D_i$ 扁平化后拼接的数据。数据类型为float16，数据格式为ND。
#### Attr
- mList：一个整数列表，用于指定每个矩阵组的 `m` 维度。
- kList：一个整数列表，用于指定每个矩阵组的 `k` 维度。
- nList：一个整数列表，用于指定每个矩阵组的 `n` 维度。

## 约束与限制
  * 输入张量 `a`、`b` 和 `c` 的数据类型必须为 `float16`。
  * 输入张量 `alpha` 和 `beta` 的数据类型必须为 `float32`。
  * 属性 `mList`、`kList` 和 `nList` 必须具有相同的长度，该长度定义了矩阵运算的组数（`groupCount`）。
  * `alpha` 和 `beta` 张量的长度必须等于 `groupCount`。
  * 输入张量 `a` 的总元素数量必须等于所有组的 `m_i * k_i` 之和。
  * 输入张量 `b` 的总元素数量必须等于所有组的 `k_i * n_i` 之和。
  * 输入张量 `c` 的总元素数量必须等于所有组的 `m_i * n_i` 之和。
  * 所有输入张量的数据格式仅支持ND。