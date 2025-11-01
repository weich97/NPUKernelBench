# Sin

## 功能描述

### 算子功能
返回一个和输入张量同样形状大小的新张量，它的每一个元素是输入张量对应元素的正弦值。

### 计算公式
$$
x = [x_0, x_1, ... x_{n-1}]\\
y = [y_0, y_1, ... y_{n-1}]\\
$$

$$
y_i = \sin(x_i) (i = 0, 1, ..., n - 1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.sin()` 函数形式提供：

```python
def sin(tensor):
    """
    实现自定义Sin操作。
    
    参数:
        tensor (Tensor): Device侧的aclTensor，公式中的输入x，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
    返回:
        Tensor: 公式中的输出y，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND，输出维度与x一致。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
tensor = torch.randn(8, 2048, dtype=torch.float)

# 使用 sin 执行计算
result = kernel_gen_ops.sin(tensor)
```
## 约束与限制

- 张量数据格式支持ND。


