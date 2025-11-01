# aclnnSqrt

## 功能描述

### 算子功能
`aclnnRsqrt` 实现了对输入数据逐元素开平方，返回开平方结果的倒数。

### 计算公式

$$
z = 1/\sqrt{x}
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.rsqrt()` 函数形式提供：

```python
def rsqrt(x):
    """
    实现自定义开平方操作。
    
    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        
    返回:
        Tensor: 计算结果张量，数据类型与输入一致，数据格式支持ND。
    
    注意:
        - 张量数据格式支持ND
        - 输入x中所有元素必须大于等于0，否则结果为NaN
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.rand(8, 2048, dtype=torch.float)  # 使用正数以避免NaN

# 使用rsqrt_custom执行计算
result = kernel_gen_ops.rsqrt(x)
```


## 约束与限制

- 张量数据格式支持ND。


