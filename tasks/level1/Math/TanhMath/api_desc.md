# aclnnTanh

## 功能描述

### 算子功能
对输入张量的每个元素进行双曲正切函数计算，输出一个与输入形状相同的张量。`TanhMath` 将输入值映射到 (-1, 1) 区间，具有“S”形曲线特性，常用于神经网络中的激活函数。

### 计算公式
$$
y = \tanh(x) = \frac{\sinh(x)}{\cosh(x)} = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.tanh_math()` 函数形式提供：

```python
def tanh_math(x):
    """
    实现自定义 TanhMath 操作。
    
    参数:
        x (Tensor): Device侧的aclTensor，公式中的输入x，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND。
        
    返回:
        Tensor: Device侧的aclTensor，公式中的输出y，数据类型支持FLOAT16、BFLOAT16、FLOAT32，数据格式支持ND，输出维度与x一致。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.randn(8, 2048, dtype=torch.float)

# 使用 tanh_math 执行计算
result = kernel_gen_ops.tanh_math(x)
```
## 约束与限制

- 张量数据格式支持ND。


