# aclnnForeachMinimumScalar

## 功能描述

### 算子功能
返回一个和输入张量列表同样形状大小的新张量列表，对张量列表 `x` 和标量值 `scalar` 执行逐元素比较，返回最小值的张量列表。

### 计算公式
$$
x = [x_0, x_1, ... x_{n-1}]\\
y = [y_0, y_1, ... y_{n-1}]\\
$$

$$
y_i = \min(x_i, scalar) (i = 0, 1, ..., n - 1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_minimum_scalar()` 函数形式提供：

```python
def foreach_minimum_scalar(tensor_list, scalar):
    """
    实现自定义 ForeachMinimumScalar 操作。
    
    参数:
        tensor_list (List[Tensor]): 公式中的`x`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16和INT32。数据格式支持ND，shape维度不高于8维。支持非连续的Tensor。
        scalar (Tensor): 公式中的`scalar`，Host侧的aclTensor，数据格式支持ND。支持非连续的Tensor。数据类型支持FLOAT、FLOAT16、INT32，且与入参`x`的数据类型具有一定对应关系：
    - 当`x`的数据类型为FLOAT、FLOAT16、INT32时，数据类型与`x`的数据类型保持一致。
    - 当`x`的数据类型为BFLOAT16时，数据类型支持FLOAT。
        
    返回:
        List[Tensor]: 公式中的`y`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16和INT32，数据格式支持ND，shape维度不高于8维。数据类型、数据格式和shape跟入参`x`的数据类型、数据格式和shape一致。支持非连续的Tensor。
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

# 创建标量张量
scalar = torch.tensor([2.0], dtype=torch.float)

# 使用 foreach_minimum_scalar 执行计算
result = kernel_gen_ops.foreach_minimum_scalar(tensor_list, scalar)
```

## 约束与限制
无
