# aclnnCross

## 功能描述

### 算子功能
计算 `x1` 和 `x2` 在 dim 维度上的向量积（叉积）


## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.cross()` 函数形式提供：


```python
def cross(input1, input2, dim=-1):
    """
    实现自定义向量叉积操作（cross product）。

    参数:
        input1 (Tensor): 输入张量，表示第一个向量组，数据格式支持ND，最后一个维度长度必须为3。
        input2 (Tensor): 输入张量，表示第二个向量组，形状必须与 input 相同。
        dim (int, 可选): 指定在哪个维度上计算叉积，默认值为-65530（即最后一个维度）。
    
    返回:
        Tensor: 输出张量，表示对应位置上 input1 和 input2 的叉积，形状与输入相同，数据类型与 input1 相同。

    注意:
        - `input1` 和 `input2` 的形状必须相同；
        - 被指定的 dim 维度长度必须为3，否则将抛出错误；
        - 支持ND格式张量，在指定的 dim 维度上执行向量叉积；
        - 此操作适用于 3D 向量或在指定维度为3的批量计算场景。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量，最后一个维度为3
input1 = torch.randn(4, 8, 3, dtype=torch.float32)
input2 = torch.randn(4, 8, 3, dtype=torch.float32)

# 使用 cross 执行叉积计算
result = kernel_gen_ops.cross(input1, input2, dim=2)
```
## 约束与限制

- 张量数据格式支持ND。


