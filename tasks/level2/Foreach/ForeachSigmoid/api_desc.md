# aclnnForeachSigmoid

## 功能描述

### 算子功能
`aclnnForeachSigmoid` 返回一个和输入张量列表同样形状大小的新张量列表，它的每一个张量是输入张量列表的每个张量进行Sigmoid函数运算的结果。

### 计算公式

$$
x = [{x_0}, {x_1}, ... {x_{n-1}}]\\
y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
$$ 

$$
y_i = \frac{1}{1+e^{-x_i}} (i=0,1,...n-1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_sigmoid()` 函数形式提供：

```python
def foreach_sigmoid(tensor_list):
    """
    对张量列表中的每个张量执行 Sigmoid 函数操作。
    
    参数:
        tensor_list (List[Tensor]): 输入张量列表，列表中所有张量必须具有相同的数据类型。
        
    返回:
        List[Tensor]: 一个新的张量列表，其中每个张量是对应输入张量的 Sigmoid 函数结果。
    
    注意:
        - 输入张量列表中的所有张量必须具有相同的数据类型
        - 张量 shape 维度不高于 8 维，数据格式支持 ND
        - 支持非连续的 Tensor
        - 输出张量与输入张量具有相同的形状、数据类型和数据格式
        - Sigmoid 函数定义为：sigmoid(x) = 1 / (1 + exp(-x))
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

# 使用 foreach_sigmoid 对张量列表中的每个张量应用 sigmoid 操作
result_list = kernel_gen_ops.foreach_sigmoid(tensor_list)
```

## 约束与限制
无