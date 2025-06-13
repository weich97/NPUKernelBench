# aclnnEye

## 功能描述

### 算子功能
创建一个二维矩阵 m×n，对角元素全为1，其它元素都为0。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.eye()` 函数形式提供：


```python
def eye(y, num_rows, num_columns=None, batch_shape=None, dtype=torch.float16, device=None):
    """
    实现自定义单位矩阵（或对角矩阵）生成操作。

    参数:
        n (int): 行数（必选）。生成的张量将有 n 行。
        m (int): 列数（必选）。生成的张量将有 m 列。
        batch_shape: 形状，广播到指定形状。
        dtype (torch.dtype, 可选): 返回张量的数据类型，默认 `torch.float32`。
        device (torch.device, 可选): 张量创建的设备，如 `'cpu'` 或 `'cuda'`。

    返回:
        Tensor: 输出张量，其对角线为 1，其余元素为 0。数据类型和设备根据参数指定。

    注意:
        - 本操作不依赖于输入张量，直接构造二维矩阵；
        - 支持浮点和整型数据类型；
        - 若 `m` 为 None，则返回 `n x n` 的方阵。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.rand(4, 8, 2048, dtype=torch.float32)  # 高维ND张量

# 使用 triu 执行计算
result = kernel_gen_ops.eye(x)
```
## 约束与限制

- 张量数据格式支持ND。


