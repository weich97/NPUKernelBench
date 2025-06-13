# aclnnForeachMulList

## 功能描述

### 算子功能
`aclnnForeachMulList` 返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入两个张量列表的每个张量进行相乘运算的结果。

### 计算公式

对于输入张量列表 $x1 = [x1_0, x1_1, ..., x1_{n-1}]$ 和 $x2 = [x2_0, x2_1, ..., x2_{n-1}]$，`aclnnForeachMulList` 操作生成输出张量列表 $y = [y_0, y_1, ..., y_{n-1}]$。

对于每个索引 $i$，输出张量 $y_i$ 通过以下计算得到：

$$y_i = x1_i * x2_i \quad (i=0,1,...n-1)$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_mul_list()` 函数形式提供：

```python
def foreach_mul_list(tensor_list1, tensor_list2):
    """
    对两个张量列表中的每对张量执行相乘操作。
    
    参数:
        tensor_list1 (List[Tensor]): 第一个输入Device侧张量列表。
                                    张量shape维度不高于8维，数据格式支持ND。支持非连续的Tensor。
        tensor_list2 (List[Tensor]): 第二个输入Device侧张量列表，列表中张量的数据类型、数据格式和shape
                                    必须与tensor_list1对应的张量相匹配。支持非连续的Tensor。
        
    返回:
        List[Tensor]: 一个新的Device侧张量列表，其中每个张量是对应两个输入张量相乘的结果。
        输出张量与输入张量具有相同的形状、数据类型和数据格式。
    
    注意:
        - 张量shape维度不高于8维，数据格式支持ND
        - 支持非连续的Tensor
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建第一个输入张量列表
tensor_list1 = [
    torch.tensor([1.0, 2.0, 3.0], dtype=torch.float),
    torch.tensor([[4.0, 5.0], [6.0, 7.0]], dtype=torch.float)
]

# 创建第二个输入张量列表
tensor_list2 = [
    torch.tensor([2.0, 3.0, 4.0], dtype=torch.float),
    torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float)
]

# 使用foreach_mul_list执行计算
result_list = kernel_gen_ops.foreach_mul_list(tensor_list1, tensor_list2)
```

## 约束与限制

无