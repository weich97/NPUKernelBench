# aclnnForeachExpm1

## 功能描述

### 算子功能
`aclnnForeachExpm1` 返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行指数运算然后减1的结果。

### 计算公式

对于输入张量列表 $X = [X_0, X_1, ..., X_{n-1}]$，`aclnnForeachExpm1` 操作生成输出张量列表 $Y = [Y_0, Y_1, ..., Y_{n-1}]$。

对于每个张量 $X_i$，输出 $Y_i$ 是通过应用指数函数计算得到的：

$$Y_i = \exp(X_i) - 1.0 = e^{X_i} - 1.0$$

其中 $e$ 是自然对数的底数，约等于 2.71828。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_expm1()` 函数形式提供：

```python
def foreach_expm1(tensor_list):
    """
    对张量列表中的每个张量执行指数函数操作然后减1。
    
    参数:
        tensor_list (List[Tensor]): 输入Device侧张量列表，列表中所有张量必须具有相同的数据类型
        
    返回:
        List[Tensor]: 一个新的Device侧张量列表，其中每个张量是对应输入张量的指数函数结果
    
    注意:
        - 输入张量列表中的所有张量必须具有相同的数据类型
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
    torch.tensor([0.0, 1.0, 2.0], dtype=torch.float),
    torch.tensor([[-1.0, -2.0], [3.0, 4.0]], dtype=torch.float)
]

# 使用 foreach_expm1 对张量列表中的每个张量应用 exp 操作然后减1
result_list = kernel_gen_ops.foreach_expm1(tensor_list)
```

## 约束与限制
无