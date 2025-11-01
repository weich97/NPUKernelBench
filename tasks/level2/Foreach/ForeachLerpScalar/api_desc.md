# aclnnForeachLerpScalar

## 功能描述

### 算子功能
`aclnnForeachLerpScalar` 算子对两个输入张量列表 `x1` 和 `x2` 中的对应张量执行线性插值操作，并返回一个新的张量列表 `y`，其形状和大小与输入张量列表相同。插值系数由一个标量值 `weight` 提供。

### 计算公式
对于输入张量列表 $x1 = [x1_0, x1_1, ..., x1_{n-1}]$ 和 $x2 = [x2_0, x2_1, ..., x2_{n-1}]$，以及标量插值系数 ${\rm weight}$，输出张量列表 $y = [y_0, y_1, ..., y_{n-1}]$ 中的每个元素 $y_i$ 的计算公式为：

$$
y_i = x1_i + {\rm weight} \times (x2_i - x1_i) \quad (i=0, 1, ..., n-1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_lerp_scalar()` 函数形式提供：

```python
def foreach_lerp_scalar(x1: List[Tensor], x2: List[Tensor], weight: Tensor) -> List[Tensor]:
    """
    对输入张量列表 x1 和 x2 中的每个对应张量执行线性插值操作，并使用标量权重。

    参数:
        x1 (List[Tensor]): 第一个输入张量列表。所有张量的数据类型支持：torch.float16, torch.bfloat16, torch.float。
                           支持非连续 Tensor，shape 维度不超过 8 维，数据格式支持 ND。
        x2 (List[Tensor]): 第二个输入张量列表，必须与 x1 具有相同的长度和形状。
                           所有张量的数据类型支持：torch.float16, torch.bfloat16, torch.float。
                           支持非连续 Tensor，shape 维度不超过 8 维，数据格式支持 ND。
        weight (Tensor): 标量插值系数，必须是单个元素的 Tensor。数据类型支持：torch.float。

    返回:
        List[Tensor]: 输出张量列表，其中每个张量与 x1 和 x2 中对应的张量具有相同的形状和数据类型。
                      输出张量与输入张量具有相同的数据类型和数据格式。
    """

```
## 使用案例

```python

import torch
import kernel_gen_ops

# 构造输入张量列表
x1_list = [torch.randn(2, 4, dtype=torch.float), torch.randn(3, 5, dtype=torch.float)]
x2_list = [torch.randn(2, 4, dtype=torch.float), torch.randn(3, 5, dtype=torch.float)]
weight_scalar = torch.tensor(0.5, dtype=torch.float)

# 执行 ForeachLerpScalar 操作
y_list = kernel_gen_ops.foreach_lerp_scalar(x1_list, x2_list, weight_scalar)

```

## 约束与限制

* 输入张量列表 `x1` 和 `x2` **必须具有相同的长度**。
* `x1` 和 `x2` 中**对应位置的张量必须具有相同的形状、数据类型和设备**。
* `weight` **必须是一个标量**（单元素）张量。
* 支持**非连续 Tensor**。
* 支持的**最大维度为 8 维**。
* 输出张量与输入张量具有**相同的数据类型和数据格式**。