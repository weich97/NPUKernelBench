# aclnnForeachAddScalarList

## 功能描述

### 算子功能
将指定的标量值加到张量列表中的每个张量中，返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行scalar相加运算的结果。

### 计算公式
  $$
  x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
  scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i=x_i+scalars_i (i=0,1,...n-1)
  $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_add_scalar_list()` 函数形式提供：

```python
def foreach_add_scalar_list(x, alpha):
    """
    对张量列表中的每个张量执行自然对数函数操作。
    
    参数:
        x (List[Tensor]): 输入Device侧张量列表，列表中所有张量必须具有相同的数据类型，数据格式为ND
        alpha (List[Tensor]): 输入Device侧张量列表，列表中所有张量必须具有相同的数据类型，数据格式为ND
        
    返回:
        List[Tensor]: 一个新的Device侧张量列表
    
    注意:
        - 输入张量列表中的所有张量必须具有相同的数据类型
        - 输出张量与输入张量具有相同的形状、数据类型和数据格式
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建具有相同数据类型的张量列表
tensor_list = [
    torch.tensor([[1.0], [2.0]], dtype=torch.float),
    torch.tensor([[4.0, 5.0], [6.0, 7.0]], dtype=torch.float)
]

# 使用 foreach_add_scalar_list 对张量列表中的每个张量应用 add 操作
result_list = kernel_gen_ops.foreach_add_scalar_list(x, alpha)
```

## 约束与限制
无
