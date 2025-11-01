# aclnnForeachDivScalarList

## 功能描述

### 算子功能
返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入的张量列表与标量列表逐元素相除的结果。

### 计算公式
$$
x = [x_0, x_1, ... x_{n-1}], scalars = [scalars_0, scalars_1, ... scalars_{n-1}]\\
y = [y_0, y_1, ... y_{n-1}]\\
$$

$$
y_i = \frac{x_i}{scalars_i} (i = 0, 1, ..., n - 1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_div_scalar_list()` 函数形式提供：

```python
def foreach_div_scalar_list(tensor_list, scalar_list):
    """
    实现自定义 ForeachDivScalarList 操作。
    
    参数:
        tensor_list (List[Tensor]): 公式中的x，Device侧的aclTensorList，表示进行除法运算的第一个输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16。数据格式支持ND，shape维度不高于8维。支持非连续的Tensor，不支持空Tensor。
        scalar_list (List[Tensor]): 公式中的scalars，Host侧的aclScalarList，表示进行除法运算的第二个输入标量列表。数据类型仅支持FLOAT，数据格式支持ND。支持非连续的Tensor，不支持空Tensor。
        
    返回:
        List[Tensor]: 公式中的y，Device侧的aclTensorList，表示进行除法运算的输出张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16 ，数据格式支持ND，shape维度不高于8维。数据类型、数据格式和shape跟入参x的数据类型、数据格式和shape一致。支持非连续的Tensor，不支持空Tensor。
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

# 创建标量张量列表
scalar_list = torch.tensor([1.2, 2.2], dtype=torch.float)

# 使用 foreach_div_scalar_list 执行计算
result = kernel_gen_ops.foreach_div_scalar_list(tensor_list, scalar_list)
```

## 约束与限制
无
