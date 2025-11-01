# aclnnExpandV2

## 功能描述

### 算子功能
将输入张量x广播成指定shape的张量。该算子x参数仅支持int64输入。

### 计算公式

$$
  x = 
  \left[
  \begin{matrix}
  3
  \end{matrix}
  \right] \\
  shape = 
  \left[
  \begin{matrix}
  2,2
  \end{matrix}
  \right] \\
  y = 
  \left[
  \begin{matrix}
  3,3\\
  3,3
  \end{matrix}
  \right]
  $$


## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.expand_v2()` 函数形式提供：


```python
def expand_v2(x, shape):
    """
    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        shape : 目标扩展形状。该形状中的维度值必须与原始张量兼容：
            - 如果目标维度为 -1，则保持该维度与原始张量相同；
            - 如果原始张量的某个维度为1，则该维度可以被扩展为任意值；
            - 如果原始张量的维度不为1，则必须与目标维度相等。

    返回:
        Tensor: 输出张量，维度为 shape，数据类型与输入一致，数据格式支持ND。

    注意:
        - 该操作不会复制数据，返回的是原始张量的一个视图（view）；
        - 原始张量的维度必须与 shape 中非 -1 的部分兼容；
        - 扩展操作不支持在原始维度不为 1 的位置进行维度改变。
    """

```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量，shape 为 [3, 1]
x = torch.tensor([[1], [2], [3]], dtype=torch.int64)

# 目标扩展为 [3, 4]
result = kernel_gen_ops.expand_v2(x, [3, 4])
```
## 约束与限制

- 张量数据格式支持ND。


