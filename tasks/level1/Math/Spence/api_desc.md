# aclnnSpence

## 功能描述

### 算子功能
实现了Spence函数（二重对数函数，dilogarithm）计算。

### 计算公式

$$
  y = \text{Spence}(x) = -\int_{1}^{x} \frac{\ln(t)}{1-t} dt
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.spence()` 函数形式提供：


```python
def spence(x):
    """
    实现 Spence 函数（狄尔对数函数，Dilogarithm, 记作 Li₂(x)）。

    参数:
        x (Tensor): 输入张量，Device 侧的张量，数据格式支持 ND。
    
    返回:
        Tensor: 输出张量，每个元素为对应输入元素应用 Spence 函数的结果。数据类型与输入一致，形状保持不变。

    注意:
        - 支持高维 ND Tensor，按元素处理。
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.rand(2048, dtype=torch.float32)

result = kernel_gen_ops.spence(x)
```
## 约束与限制

- 张量数据格式支持ND。


