# aclnnMatMulV2

## 功能描述

### 算子功能
实现了两个矩阵相乘，返回矩阵乘法结果的功能。

### 计算公式

  $$
  z = x ⋅ y
  $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.matmul()` 函数形式提供：

```python
def matmul(x, y):
    """
    实现自定义加法操作。
    
    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        y (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        
    返回:
        Tensor: 计算结果张量，数据类型与输入一致，数据格式支持ND。
    
    注意:
        张量数据格式支持ND
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.randn(8, 2048, dtype=torch.float16)
y = torch.randn(2048, 128, dtype=torch.float16)

# 使用add_custom执行计算
result = kernel_gen_ops.matmul(x, y)
```

## 约束与限制

- 输入输出张量数据格式支持ND。