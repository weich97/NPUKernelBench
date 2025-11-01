# aclnnForeachAddList

## 功能描述

### 算子功能
两个Tensor列表中的元素逐个相加，并返回一个新的Tensor列表。可以通过设置alpha参数来调整相加的系数。

### 计算公式

  $$
  x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}]\\
  y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
  $$

  $$
  y_i = {x1}_{i}+{x2}_{i}*{alpha} (i=0,1,...n-1)
  $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_add_list()` 函数形式提供：

```python
def foreach_add_list(x1, x2, alpha):
    """
    对张量列表中的每个张量执行自然对数函数操作。
    
    参数:
        x1 (List[Tensor]): 输入Device侧张量列表，列表中所有张量必须具有相同的数据类型，数据格式为ND
        x2 (List[Tensor]): 输入Device侧张量列表，列表中所有张量必须具有相同的数据类型，数据格式为ND
        alpha (List[Tensor]): 输入Device侧张量，数据格式为ND
        
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
    torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float),
    torch.tensor([[4.0, 5.0], [6.0, 7.0], [8.0]], dtype=torch.float)
]

# 使用 foreach_log 对张量列表中的每个张量应用 log 操作
result_list = kernel_gen_ops.foreach_add_list(x1, x2, alpha)
```

## 约束与限制
无
