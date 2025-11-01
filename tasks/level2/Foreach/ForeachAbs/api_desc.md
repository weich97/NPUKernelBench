# aclnnForeachAbs

## 功能描述

### 算子功能
`aclnnForeachAbs` 返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的张量的绝对值。

### 计算公式

对于输入张量列表 $x = [{x_0}, {x_1}, ... {x_{n-1}}]$，`aclnnForeachAbs` 操作生成输出张量列表 $y = [{y_0}, {y_1}, ... {y_{n-1}}]$。

对于每个张量 $x_i$，输出 $y_i$ 是通过取绝对值操作计算得到的：

$$y_i=|{x_i}| (i=0,1,...n-1)$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_abs()` 函数形式提供：

```python
def foreach_abs(tensor_list):
    """
    对张量列表中的每个张量执行取绝对值操作。
    
    参数:
        tensor_list (List[Tensor]): 输入Device侧张量列表，列表中所有张量必须具有相同的数据类型。
                                   张量shape维度不高于8维，数据格式支持ND。支持非连续的Tensor。
        
    返回:
        List[Tensor]: 一个新的Device侧张量列表，其中每个张量是对应输入张量的绝对值。
    
    注意:
        - 张量shape维度不高于8维，数据格式支持ND
        - 支持非连续的Tensor
        - 输出张量与输入张量具有相同的形状、数据类型和数据格式
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建具有相同数据类型的张量列表
tensor_list = [
    torch.tensor([-1.0, 2.0, -3.0], dtype=torch.float),
    torch.tensor([[-4.0, 5.0], [-6.0, 7.0]], dtype=torch.float)
]

# 使用 foreach_abs 对张量列表中的每个张量应用取绝对值操作
result_list = kernel_gen_ops.foreach_abs(tensor_list)
```

## 约束与限制

无