# aclnnFastGeluGrad

## 功能描述

### 算子功能
‌`FastGeluGrad‌`是一种在`FastGelu`基础上进行升级的算子，`FastGeluGrad`算子在`FastGelu`的基础上增加了一个输入`dy`，使得其有两个输入和一个输出，不需要进行大幅度的数据搬迁。其主要目的是优化计算过程，减少类型转换的需求。

### 计算公式

$$
  z = \text{d}y \frac {\exp(-1.702|x|) + 1.702x\exp(-1.702|x|) + \exp(1.702(x-|x|))} {(\exp(-1.702|x|) + 1) ^ 2}
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.fast_gelu_grad()` 函数形式提供：


```python
def fast_gelu_grad(dy, x):
    """
    实现 FastGELU 激活函数的反向梯度计算。

    参数:
        dy (Tensor): 上游梯度，形状与 `x` 相同。
        x (Tensor): 输入张量，原始的前向输入，Device 侧张量。

    返回:
        Tensor: 输出张量，与 `x` 形状一致，表示 FastGELU 的梯度值。数据类型与输入一致。

    注意:
        - 支持 ND 格式张量；
        - 要求输入和梯度张量的数据类型一致；
        - 推荐用于训练阶段进行反向传播。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建上游梯度和输入
dy = torch.randn(4, 8, 2048, dtype=torch.float32)     # 与 x 相同形状的上游梯度
x = torch.randn(4, 8, 2048, dtype=torch.float32)      # 任意维度张量

# 使用 fast_gelu_grad 计算反向梯度
grad = kernel_gen_ops.fast_gelu_grad(dy, x)
```
## 约束与限制

- 张量数据格式支持ND。


