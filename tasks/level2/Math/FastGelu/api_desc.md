# aclnnFastGelu

## 功能描述

### 算子功能
快速高斯误差线性单元激活函数。

### 计算公式

$$
  y = \frac {x} {1 + \exp(-1.702 \left| x \right|)} \exp(0.851 (x - \left| x \right|))
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.fast_gelu()` 函数形式提供：


```python
def fast_gelu(x):
    """
    实现 FastGELU 激活函数。

    参数:
        x (Tensor): 输入张量，Device 侧的张量，数据格式支持ND。

    返回:
        Tensor: 输出张量，形状与输入一致，应用了 fast_gelu 激活函数。数据类型与输入一致。

    注意:
        - 支持 ND 格式张量；
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.randn(4, 8, 2048, dtype=torch.float32)  # 任意维度张量

# 使用 fast_gelu 激活函数
result = kernel_gen_ops.fast_gelu(x)
```
## 约束与限制

- 张量数据格式支持ND。


