# aclnnMulSigmoidMulAddCustom

## 功能描述

### 算子功能
实现了数据mul->sigmoid->mul->add计算过程，返回结果的功能。

### 计算公式
$$\text{mul_res} = \text{a1} \cdot \text{a2}$$
  $$\text{sigmoid_res} = \frac{1}{1 + e^{-\text{mul_res}}}$$
  $$\text{mul_2_res} = \text{sigmoid_res} \cdot \text{a3}$$
  $$\text{add_result} = \text{mul_2_res} + \text{a4}$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.mul_sigmoid_mul_add_custom()` 函数形式提供：


```python
def mul_sigmoid_mul_add_custom(
    input: torch.Tensor,
    mulscalar1: torch.Tensor,
    mulscalar2: torch.Tensor,
    mulscalar3: torch.Tensor,
) -> torch.Tensor:
    """
    实现 mul_sigmoid_mul_add_custom 操作，对输入依次进行乘法、Sigmoid、乘法、加法运算。

    参数:
        input: 输入张量；
        mulscalar1: 输入张量，与 a1 可广播；
        mulscalar2: 输入张量，与中间结果可广播；
        mulscalar3: 输入张量，与中间结果可广播；

    返回:
        Tensor: 输出张量，形状与广播结果一致，表示经过 mul → sigmoid → mul → add 运算后的结果。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 构造输入
input = torch.rand(32, 32, dtype=torch.float32)

mulscalar1 = torch.randn(1, dtype=torch.float32)
mulscalar2 = torch.randn(1, dtype=torch.float32)
mulscalar3 = torch.randn(1, dtype=torch.float32)


output = kernel_gen_ops.mul_sigmoid_mul_add_custom(input, mulscalar1, mulscalar2, mulscalar3)

```
## 约束与限制

- 张量数据格式支持ND。


