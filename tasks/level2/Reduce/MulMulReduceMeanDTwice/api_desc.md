# aclnnMulMulReduceMeanDTwice

## 功能描述

### 算子功能
实现了数据相乘、第一次mean、平方差、第二次mean，返回计算结果的功能。

### 计算公式

$$
  \text{mul_res} = \text{mul0input0} \times \text{mul0input1} \times \text{mul1input0}
  $$
  $$
  \text{reduce_mean_0} = \frac{1}{N} \sum_{i=1}^{N} \text{mul_res}[i]
  $$
  $$
  \text{diff} = \text{mul_res} - \text{reduce_mean_0}
  $$
  $$
  \text{muld_res} = \text{diff} \times \text{diff}
  $$
  $$
  \text{x2} = \frac{1}{N} \sum_{i=1}^{N} \text{muld_res}[i]
  $$
  $$
  \text{reduce_mean_1} = \frac{\gamma}{\sqrt{\text{x2} + \text{addy}}}
  $$
  $$
  \text{output} = \beta - \text{reduce_mean_1} \times \text{reduce_mean_0} + \text{reduce_mean_1} \times \text{mul_res}
  $$

  $$
  \text{out} = \frac{1}{1 + e^{-(\text{output} \cdot \text{t}_1)}}
  $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.mul_mul_reduce_mean_d_twice()` 函数形式提供：


```python
def mul_mul_reduce_mean_d_twice(
    mul0_input0: torch.Tensor,
    mul0_input1: torch.Tensor,
    mul1_input0: torch.Tensor,
    addy: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """
    实现 mul_mul_reduce_mean_d_twice 操作，完成一系列乘法、归一化与缩放的运算。

    参数:
        mul0_input0: 输入张量，形状为 [B, D]，表示乘法第一个输入；
        mul0_input1: 输入张量，形状与 mul0_input0 相同，表示乘法第二个输入；
        mul1_input0: 缩放因子张量，通常为标量，可广播为输入形状；
        addy: 常数偏移张量，用于防止除零，形状可广播；
        gamma: 归一化缩放参数，形状为 [1, D]，可广播；
        beta: 偏移参数张量，形状为 [1, D]，可广播；

    返回:
        Tensor: 输出张量，经过归一化缩放和 Sigmoid 运算的结果，形状为 [B, D]。

    注意:
        - 所有张量支持广播机制；
        - 支持 ND 格式。
    """


```

## 使用案例

```python
import torch
import kernel_gen_ops  # 假设此模块已实现该操作

# 构造输入
mul0_input0 = torch.rand(90, 1024, dtype=torch.float16)
mul0_input1 = torch.rand(90, 1024, dtype=torch.float16)
mul1_input0 = torch.tensor(1.0, dtype=torch.float16)
addy = torch.tensor(1.0, dtype=torch.float16)
gamma = torch.rand(1, 1024, dtype=torch.float16)
beta = torch.rand(1, 1024, dtype=torch.float16)

# 调用运算
output = kernel_gen_ops.mul_mul_reduce_mean_d_twice(
    mul0_input0, mul0_input1, mul1_input0, addy, gamma, beta
)
```
## 约束与限制

- 张量数据格式支持ND。


