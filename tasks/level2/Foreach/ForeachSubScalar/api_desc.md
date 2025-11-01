# aclnnForeachSubScalar

## 功能描述

### 算子功能
返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行scalar相减运算的结果。

### 计算公式
$$
x = [x_0, x_1, ... x_{n-1}]\\
y = [y_0, y_1, ... y_{n-1}]\\
$$

$$
y_i = x_i - scalar (i = 0, 1, ..., n - 1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_sub_scalar()` 函数形式提供：

```python
def foreach_sub_scalar(tensor_list, alpha):
    """
    实现自定义 ForeachSubScalar 操作。
    
    参数:
        tensor_list (List[Tensor]): 公式中的`x`，Device侧的aclTensorList，数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。数据格式支持ND，shape维度不高于8维。支持非连续的Tensor。
        alpha (Tensor): 公式中的`scalar`，Host侧的aclTensor，数据格式支持ND。支持非连续的Tensor。数据类型支持FLOAT、FLOAT16、INT32，且与入参`x`的数据类型具有一定对应关系：
    - 当`x`的数据类型为FLOAT、FLOAT16、INT32时，数据类型与`x`的数据类型保持一致。
    - 当`x`的数据类型为BFLOAT16时，数据类型支持FLOAT。
        
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

# 创建alpha张量
alpha = torch.tensor([1.2], dtype=torch.float)

# 使用 foreach_sub_scalar 执行计算
result = kernel_gen_ops.foreach_sub_scalar(tensor_list, alpha)
```

## 约束与限制
无
