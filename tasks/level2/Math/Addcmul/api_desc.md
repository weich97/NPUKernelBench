# aclnnAddcmul

## 功能描述

### 算子功能
执行 x1 与 x2 的逐元素乘法，将结果乘以标量值value并与输入input_data做逐元素加法。

### 计算公式

$$
y = input\_data+value×x1×x2
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.addcmul()` 函数形式提供：


```python
def addcmul(input_data, x1, x2, value=1.0):
    """
    实现自定义按元素除加操作（element-wise divide-add）。

    参数:
        input_data (Tensor): 基础张量，将在其上加上除法项。数据格式支持ND。
        x1 (Tensor): 分子张量。
        x2 (Tensor): 分母张量（不能为0）。
        value (scalar or Tensor): 缩放因子，用于除法项的缩放。

    返回:
        Tensor: 输出张量，其值为 input_data + value * (x1 / x2)。数据类型与输入一致。

    注意:
        - 所有输入张量必须可广播到相同形状；
        - 此操作是逐元素执行的；
        - 支持ND格式张量；
        - 分母 x2 中不能包含零值；
        - 通常用于优化器实现等场景。
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

# 使用 addcmul 执行计算
result = kernel_gen_ops.addcmul(input_data, x1, x2, value)
```
## 约束与限制

- 张量数据格式支持ND。


