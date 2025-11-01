# aclnnForeachLog10

## 功能描述

### 算子功能
`aclnnForeachLog10` 操作对输入张量列表中的每个张量依次执行以10为底的对数函数变换，并返回一个新的张量列表，其中每个元素是对应输入张量的以10为底的对数函数变换结果。此操作保留张量的形状和维度信息，仅改变张量中的数值。

### 计算公式

对于输入张量列表 $X = [X_0, X_1, ..., X_{n-1}]$，`aclnnForeachLog10` 操作生成输出张量列表 $Y = [Y_0, Y_1, ..., Y_{n-1}]$。

对于每个张量 $X_i$，输出 $Y_i$ 是通过应用以10为底的对数函数计算得到的：

$$Y_i = \log_{10}(X_i)$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_log10()` 函数形式提供：

```python
def foreach_log10(tensor_list):
    """
    对张量列表中的每个张量执行以10为底的对数函数操作。
    
    参数:
        tensor_list (List[Tensor]): 输入Device侧张量列表，列表中所有张量必须具有相同的数据类型
        
    返回:
        List[Tensor]: 一个新的Device侧张量列表，其中每个张量是对应输入张量的以10为底的对数函数结果
    
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
    torch.tensor([1.0, 10.0, 100.0], dtype=torch.float),
    torch.tensor([[1000.0, 10000.0], [100000.0, 1000000.0]], dtype=torch.float)
]

# 使用 foreach_log10 对张量列表中的每个张量应用 log10 操作
result_list = kernel_gen_ops.foreach_log10(tensor_list)
```

## 约束与限制

无