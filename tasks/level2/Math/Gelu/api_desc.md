# Gelu

## 功能描述

### 算子功能
该AscendC算子用于计算Gelu函数。

### 计算公式

  $$\text{GELU}(x) \approx \frac{x}{1 + \exp\left(-\sqrt{\frac{8}{\pi}} \left(x + 0.044715 \cdot x^3\right)\right)}$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.gelu()` 函数形式提供：

```python
def gelu(x):
    """
    实现GELU激活函数操作。
    
    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        
    返回:
        Tensor: 计算结果张量，数据类型与输入一致，数据格式支持ND。
    
    注意:
        - 张量数据格式支持ND
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.randn(8, 1024, dtype=torch.float32)

# 使用gelu执行计算
result = kernel_gen_ops.gelu(x)
```

## 约束与限制

- 输入输出张量数据格式支持ND。