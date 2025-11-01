# aclnnForeachMulScalarList

## 功能描述

### 算子功能
`aclnnForeachMulScalarList` 返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行scalar相乘运算的结果。

### 计算公式

对于输入张量列表 $x = [x_0, x_1, ..., x_{n-1}]$ 和标量列表 $scalars = [scalars_0, scalars_1, ..., scalars_{n-1}]$，`aclnnForeachMulScalarList` 操作生成输出张量列表 $y = [y_0, y_1, ..., y_{n-1}]$。

对于每个索引 $i$，输出张量 $y_i$ 通过以下计算得到：

$$y_i = x_i * scalars_i \quad (i=0,1,...n-1)$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_mul_scalar_list()` 函数形式提供：

```python
def foreach_mul_scalar_list(tensor_list, scalar_list):
    """
    对张量列表中的每个张量与对应的标量相乘。
    
    参数:
        tensor_list (List[Tensor]): 公式中的`x`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。数据格式支持ND，shape维度不高于8维。支持非连续的Tensor。
        scalar_list (List[Scalar]): 公式中的`scalars`，Host侧的aclScalarList，数据格式支持ND。支持非连续的Tensor。`scalars`的数据类型仅支持FLOAT和INT64，且与输入参数的数据类型具有一定对应关系：
    - 当入参`x`的数据类型为FLOAT、FLOAT16、BFLOAT16时，`scalars`的数据类型仅支持FLOAT。
    - 当入参`x`的数据类型为INT32时，`scalars`的数据类型仅支持INT64。
        
    返回:
        List[Tensor]: 公式中的`y`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32，数据格式支持ND，shape维度不高于8维。数据类型、数据格式和shape跟入参`x`的数据类型、数据格式和shape一致。支持非连续的Tensor。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量列表
tensor_list = [
    torch.tensor([1.0, 2.0, 3.0], dtype=torch.float),
    torch.tensor([[4.0, 5.0], [6.0, 7.0]], dtype=torch.float)
]

# 创建标量列表
scalar_list = [1.2, 2.2]

# 使用foreach_mul_scalar_list执行计算
result_list = kernel_gen_ops.foreach_mul_scalar_list(tensor_list, scalar_list)
```

## 约束与限制

无