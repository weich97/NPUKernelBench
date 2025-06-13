# aclnnAddcdiv

## 功能描述

### 算子功能
执行 x1 除以 x2 的元素除法，将结果乘以标量 value 并将其添加到 input_data。

### 计算公式

$$
 y = input\_data+x1/x2×value
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.addcdiv()` 函数形式提供：


```python
def addcdiv(input_data, x1, x2, value=1.0):
    """
    实现自定义按元素乘加操作（element-wise multiply-add）。

    参数:
        input_data (Tensor): 基础张量，将在其上加上乘积项。数据格式支持ND。
        x1 (Tensor): 第一个参与乘法的张量。
        x2 (Tensor): 第二个参与乘法的张量。
        value (scalar): 缩放因子，用于乘积项的缩放

    返回:
        Tensor: 输出张量，其值为 input_data + value * x1 * x2。数据类型与输入一致。

    注意:
        - 所有输入张量必须可广播到相同形状；
        - 此操作是逐元素执行的；
        - 支持ND格式张量；
        - 通常用于梯度累计等场景。
    """


```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
input_data = torch.ones(2, 3, dtype=torch.float32)
x1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
x2 = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float32)
value = torch.tensor([0.5], dtype=torch.float32)

# 使用 addcdiv 执行计算
result = kernel_gen_ops.addcdiv(input_data, x1, x2, value)
```
## 约束与限制

- 张量数据格式支持ND。


