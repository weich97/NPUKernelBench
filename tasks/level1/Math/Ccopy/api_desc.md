# aclnnCcopy

## 功能描述

### 算子功能
将复数向量x的值复制到另一个向量out中。 

### 计算公式
  $$
    out = x
  $$
  

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.ccopy()` 函数形式提供：


```python
def ccopy(x):
    """
    实现自定义张量拷贝操作（元素级深拷贝）。

    参数:
        x (Tensor): 输入张量，支持任意维度和数据类型。

    返回:
        Tensor: 与输入完全相同的新张量，但为不同的内存副本（深拷贝）。

    """

```

## 使用案例

```python
import torch
import kernel_gen_ops

real_part = torch.rand(32, 16, dtype=torch.float32)
imag_part = torch.rand(32, 16, dtype=torch.float32)
x = torch.complex(real_part, imag_part)

# 拷贝张量
result = kernel_gen_ops.ccopy(x)

```
## 约束与限制

- 张量数据格式支持ND。