# aclnnNeg

## 功能描述

### 算子功能
返回一个和输入张量同样形状大小的新张量，它的每一个元素是输入张量对应元素取负的结果。

### 计算公式
$$
x = [x_0, x_1, ... x_{n-1}]\\
y = [y_0, y_1, ... y_{n-1}]\\
$$

$$
y_i = -x_i (i = 0, 1, ..., n - 1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.neg()` 函数形式提供：

```python
def neg(tensor):
    """
    实现自定义 Neg 操作。
    
    参数:
        tensor (Tensor): 输入Device侧张量。
                         张量shape维度不高于8维，数据格式支持ND。支持非连续的Tensor。
        
    返回:
        Tensor: 一个新的Device侧张量，其中每个元素是输入张量对应元素取负的结果。
        输出张量与输入张量具有相同的形状、数据类型和数据格式。
    
    注意:
        - 张量shape维度不高于8维，数据格式支持ND
        - 支持非连续的Tensor
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
tensor = torch.randn(8, 2048, dtype=torch.float)

# 使用 neg 执行计算
result = kernel_gen_ops.neg(tensor)
```
## 约束与限制

- 张量数据格式支持ND。


