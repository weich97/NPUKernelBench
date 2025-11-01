# aclnnSscal

## 功能描述

### 算子功能
对一个实数向量进行缩放，即将向量中的每个元素乘以一个实数alpha。

### 计算公式

$$
x = alpha * x
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.sscal()` 函数形式提供：


```python
def sscal(x, alpha, n, incx):
    """
    实现自定义缩放操作（sscal），将输入张量乘以一个标量。

    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持任意维度（ND）。
        alpha (float, 可选): 缩放因子，默认为1.0。
                    - 所有元素将乘以该值；
                    - 通常用于向量或张量的整体缩放操作；
        n (int): 要考虑的元素数量。
        incx (int): 访问 `x` 中元素时的步长（增量）。

    返回:
        Tensor: 输出张量，所有元素为原始张量乘以 alpha。数据类型与输入一致，数据格式支持ND。

    注意:
        - 输入张量可以为任意维度；
        - 所有元素将统一按标量 alpha 缩放；
        - 输出张量与输入张量形状相同。
    """

```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.ones(2048, dtype=torch.float32)  # 高维ND张量

# 使用sscal执行计算
result = kernel_gen_ops.sscal(x, diagonal=0)
```
## 约束与限制

- 张量数据格式支持ND。


