# aclnnInplaceFusedMatmulSoftmaxGrad

## 功能描述

### 算子功能
该Ascend C算子实现了`InplaceFusedMatmulSoftmaxGrad`的功能，在一个融合操作中计算`softmax`的梯度。该算子通常用在深度学习模型的反向传播阶段，特别是在包含`softmax`层的注意力机制或分类任务中。它接收`softmax`操作的输出、后续层反向传播的梯度以及一个`values`张量作为输入，并计算出相对于`softmax`输入的梯度。

### 计算公式
假设`softmax`的输出为`S`，后续层反向传播的梯度为`grad_output`，以及一个`values`张量，该算子的计算可以分为两步：

1.  首先，计算`grad_output`和`values`的转置的矩阵乘法，得到中间结果`grad_softmax`：
    
    $$\text{grad_softmax} = \text{grad_output} @ \text{values}^T$$
    
2.  然后，使用`grad_softmax`和`softmax`的输出`S`来计算最终的梯度：
    
    $$\text{output} = ( \text{grad_softmax} - \text{sum}(\text{S} \cdot \text{grad_softmax}, \text{axis}=-1, \text{keepdim=True})) \cdot \text{S}$$
    
其中 `·` 代表逐元素相乘。这个操作是inplace的，意味着计算结果会直接覆盖`softmax`输出张量`S`的内存空间。

### 计算过程与类型转换
算子在内部处理过程中，会根据输入的数据类型选择不同的计算路径和精度策略：

1.  **输入**：算子支持`float16`、`bfloat16`和`float32`三种数据类型的输入张量`softmaxOutput`、`gradOutput`和`values`。
2.  **中间计算**：
    * 在执行矩阵乘法（`grad_output @ values^T`）时，为了保证精度，累加过程会使用`float32`类型。
    * 后续的梯度计算也在高精度下进行，以避免数值不稳定。
3.  **输出**：计算得到的最终梯度结果会转换回与输入张量相同的数据类型，并写回到`softmaxOutput`张量所在的内存地址。

## 接口定义

### 算子原型定义接口
#### Input
- `softmaxOutput`：Device侧的aclTensor，即公式中的`S`。这是一个inplace操作的输入输出张量。数据类型支持`float16`、`bfloat16`、`float32`，数据格式为ND。
- `gradOutput`：Device侧的aclTensor，即公式中的`grad_output`。数据类型支持`float16`、`bfloat16`、`float32`，数据格式为ND。
- `values`：Device侧的aclTensor，即公式中的`values`。数据类型支持`float16`、`bfloat16`、`float32`，数据格式为ND。

#### Output
- `softmaxOutput`：Device侧的aclTensor，这是一个inplace算子，输出结果会直接写回到该张量。数据类型与输入`softmaxOutput`一致，数据格式为ND。

#### Attr
- 无

## 约束与限制
* 所有输入张量（`softmaxOutput`、`gradOutput`、`values`）的数据类型必须相同，且当前仅支持`float16`、`bfloat16`和`float32`。
* 输入张量的维度必须至少为2维。
* `gradOutput`的最后一个维度必须与`values`的最后一个维度相等。
* `softmaxOutput`、`gradOutput`和`values`的批处理维度（除最后两个维度外的所有维度）必须能够正确广播。
* 输入张量的数据格式只支持ND。

