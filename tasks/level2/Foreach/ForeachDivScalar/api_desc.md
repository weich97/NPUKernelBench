# aclnnForeachDivScalar

## 功能描述

### 算子功能
返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入的张量列表除以标量的结果。

### 计算公式
$$
x = [x_0, x_1, ... x_{n-1}]\\
y = [y_0, y_1, ... y_{n-1}]\\
$$

$$
y_i = \frac{x_i}{scalar} (i = 0, 1, ..., n - 1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_div_scalar()` 函数形式提供：

```python
def foreach_div_scalar(tensor_list, scalar):
    """
    实现自定义 ForeachDivScalar 操作。
    
    参数:
        tensor_list (List[Tensor]): 公式中的x，Device侧的aclTensorList，表示除法运算的第一个输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16。数据格式支持ND，shape维度不高于8维。支持非连续的Tensor，不支持空Tensor。
        scalar (Tensor): 公式中的scalar，Host侧的aclTensor，表示除法运算的第二个输入标量。数据格式支持ND。支持非连续的Tensor，不支持空Tensor。数据类型支持FLOAT、FLOAT16，且与入参x的数据类型具有一定对应关系：
                        - 当x的数据类型为FLOAT、FLOAT16时，数据类型与x的数据类型保持一致。
                        - 当x的数据类型为BFLOAT16时，数据类型支持FLOAT。
        
    返回:
        List[Tensor]: 公式中的y，Device侧的aclTensorList，表示张量x除以标量值scalar的输出张量列表。数据格式支持ND，数据类型支持FLOAT、FLOAT16、BFLOAT16，shape维度不高于8维。数据类型、数据格式和shape跟入参x的数据类型、数据格式和shape一致。支持非连续的Tensor，不支持空Tensor。
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

# 创建scalar张量
scalar = torch.tensor([1.2], dtype=torch.float)

# 使用 foreach_div_scalar 执行计算
result = kernel_gen_ops.foreach_div_scalar(tensor_list, scalar)
```

## 约束与限制
无
