# aclnnGroupedMatmulSliceMPerTokenDequant

## 功能描述

### 算子功能
该Ascend C算子用于执行一种特殊的分组量化矩阵乘法。它将输入矩阵 `a` 按照 `groupList` 定义的边界进行动态切片，然后将每个切片与 `b` 中对应的权重矩阵相乘。乘法结果会使用一个组级别（group-wise）的缩放因子 `scale` 和一个逐令牌（per-token）的缩放因子 `per_token_scale` 进行反量化。

该算子常见于大规模语言模型（LLM）的量化推理场景，能够高效处理动态输入长度（`token` 数量）的计算需求。

### 计算公式
假设有以下输入张量：
- $A$ (对应输入 `a`): 维度为 $m \times k$
- $B$ (对应输入 `b`): 维度为 $G \times k \times n$，其中 $G$ 为组数（`groupCount`）
- $S$ (对应输入 `scale`): 维度为 $G \times n$
- $P$ (对应输入 `per_token_scale`): 维度为 $m$
- $L$ (对应输入 `groupList`): 维度为 $G$，定义了分组的行边界

算子的计算可以分解为 $G$ 个独立的组计算。对于第 $g$ 组（$g$ 从 $0$ 到 $G-1$）：

1.  **确定切片边界**:
    - 起始行索引 $start_idx_g = L_{g-1}$ (当 $g>0$ 时)，且 $start_idx_0 = 0$。
    - 结束行索引 $end_idx_g = L_g$。

2.  **对输入A进行切片**:
    - 获取第 $g$ 组对应的子矩阵 $A_g = A[start_idx_g : end_idx_g, :]$。其维度为 $(end_idx_g - start_idx_g) \times k$。

3.  **执行矩阵乘法**:
    - 获取第 $g$ 组的权重矩阵 $B_g = B[g, :, :]$。
    - 计算中间结果 $M_g = A_g \times B_g$。其计算公式为：
      $$(M_g)_{ij} = \sum_{p=1}^{k} (A_g)_{ip} (B_g)_{pj}$$

4.  **反量化**:
    - 对中间结果 $M_g$ 进行反量化，得到该组的最终结果 $C_g$：
      $$(C_g)_{ij} = (M_g)_{ij} \times S_{gj} \times P_{start_idx_g + i}$$
      其中，$i$ 和 $j$ 分别是 $C_g$ 的行索引和列索引。

5.  **拼接结果**:
    - 将每一组的结果 $C_g$ 展平（Flatten），然后按组顺序拼接（Concatenate），形成最终的一维输出张量 $C$。

### 计算过程与类型转换
为了在保证计算精度的同时利用低比特整数运算的性能优势，算子内部执行了精确的类型转换流程：

1.  算子接收 `int8` 类型的输入张量 `a` 和 `b`，以及 `bfloat16` 类型的缩放因子 `scale` 和 `per_token_scale`。
2.  在执行矩阵乘法之前，`int8` 类型的子矩阵 $A_g$ 和 $B_g$ 会被提升为 `int32` 类型。
3.  矩阵乘法 $A_g \times B_g$ 在 `int32` 精度下完成，得到 `int32` 类型的中间结果 $M_g$。
4.  在反量化阶段，`int32` 的 $M_g$、`bfloat16` 的 $S_g$ 和 $P_g$ 切片均被转换为 `float32` 类型进行元素乘法运算。
5.  所有组的计算结果（`float32` 类型）在拼接后，最终被转换回 `bfloat16` 类型，作为算子的最终输出。

## 接口定义

### 算子原型定义接口
#### Input
- a：Device侧的aclTensor，公式中的A。数据类型支持int8，数据格式支持ND，维度必须为2维。
- b：Device侧的aclTensor，公式中的B。数据类型支持int8，数据格式支持ND，维度必须为3维。
- scale：Device侧的aclTensor，公式中的S。数据类型支持bfloat16，数据格式支持ND，维度必须为2维。
- per_token_scale：Device侧的aclTensor，公式中的P。数据类型支持bfloat16，数据格式支持ND，维度必须为1维。
- groupList：Device侧的aclTensor，公式中的L。数据类型支持int64，数据格式支持ND，维度必须为1维。
#### Output
- c：Device侧的aclTensor，公式中的C。数据类型为bfloat16，数据格式为ND，维度为1维。
#### Attr
- 无

## 约束与限制
  * 各输入张量的数据类型必须遵循接口定义：`a` 和 `b` 为 `int8`，`scale` 和 `per_token_scale` 为 `bfloat16`，`groupList` 为 `int64`。
  * 输入张量的维度必须固定：`a` 为2D，`b` 为3D，`scale` 为2D，`per_token_scale` 为1D，`groupList` 为1D。
  * 维度匹配约束：
      - `a.shape[1]` (k) 必须等于 `b.shape[1]` (k)。
      - `b.shape[0]` (groupCount) 必须等于 `scale.shape[0]` (groupCount) 和 `groupList.shape[0]` (groupCount)。
      - `b.shape[2]` (n) 必须等于 `scale.shape[1]` (n)。
      - `a.shape[0]` (m) 必须等于 `per_token_scale.shape[0]` (m)。
  * `groupList` 必须是单调非递减的，且其最后一个元素的值必须等于 `a` 的第一维大小（m）。
  * 所有输入张量的数据格式仅支持ND。