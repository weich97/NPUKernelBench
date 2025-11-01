# aclnnMuls

## 功能描述

### 算子功能
将输入张量与标量张量相乘，返回一个和输入张量同样形状大小的新张量，它的每一个元素是输入张量元素与标量张量元素相乘的结果。

### 计算公式
$$
x = [x_0, x_1, ... x_{n-1}]\\
y = [y_0, y_1, ... y_{n-1}]\\
$$

$$
y_i = x_i * value (i = 0, 1, ..., n - 1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.muls()` 函数形式提供：

```python
def muls(x, value):
    """
    实现自定义 Muls 操作。
    
    参数:
        x (Tensor): Device侧的aclTensor，公式中的输入x，数据类型支持FLOAT32, FLOAT16，BF16，INT16,INT32, INT64, COMPLEX64，数据格式支持ND。
        value (Tensor): Device侧的attr，数据类型支持FLOAT，数据格式支持ND。
        
    返回:
        Tensor: Device侧的aclTensor，公式中的输出y，数据类型支持FLOAT32, FLOAT16，BF16，INT16,INT32, INT64, COMPLEX64，数据格式支持ND，输出维度和数据类型与x一致。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.randn(8, 2048, dtype=torch.float)

# 创建value张量
value = torch.tensor([1.2], dtype=torch.float)

# 使用 muls 执行计算
result = kernel_gen_ops.muls(x, value)
```
## 约束与限制

- 张量数据格式支持ND。


