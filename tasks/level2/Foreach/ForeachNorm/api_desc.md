# aclnnForeachNorm

## 功能描述

### 算子功能
`aclnnForeachNorm` 返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行范数运算的结果。

### 计算公式

$$
x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
$$

$$
y = \left(\sum_{i=0}^{n-1}|x_i|^{p}\right)^{\frac{1}{{p}}}  (i=0,1,...n-1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_norm()` 函数形式提供：

```python
def foreach_norm(tensor_list, scalar):
    """
    对张量列表中的每个张量执行范数（norm）运算。
    
    参数:
        tensor_list (List[Tensor]): 输入张量列表，所有张量必须具有相同的数据类型。
        scalar: 范数的阶数
    
    返回:
        List[Tensor]: 一个新的张量列表，其中每个张量是对应输入张量的范数结果，通常为标量张量。
    
    注意:
        - 输入张量列表中的所有张量必须具有相同的数据类型
        - 张量 shape 维度不高于 8 维，数据格式支持 ND
        - 支持非连续的 Tensor
        - 输出张量与输入张量在数量上对应
    """

```

## 使用案例

```python
import torch
import random
import kernel_gen_ops

# 创建具有相同数据类型的张量列表
tensor_list = [
    torch.tensor([0.0, 1.0, 2.0], dtype=torch.float),
    torch.tensor([[-1.0, -2.0], [3.0, 4.0]], dtype=torch.float)
]

dtype_scale = random.choice([torch.int64, torch.float])
if dtype_scale == torch.int64:
    value = 2
else:
    value = 0.5
scalar = torch.tensor(value, dtype=dtype_scale)

# 计算每个张量的 L2 范数
result_list = kernel_gen_ops.foreach_norm(tensor_list, scalar)
```

## 约束与限制
无