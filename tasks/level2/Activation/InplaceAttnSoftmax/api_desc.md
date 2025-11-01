# aclnnInplaceAttnSoftmax

## 功能描述

### 算子功能
该InplaceAttnSoftmax算子提供等同torch softmax计算功能。InplaceAttnSoftmax算子的主要功能是缩放输入张量，将结果缩放在[0,1]范围内总和为1，并将结果原地写入输入张量。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.inplace_attn_softmax()` 函数形式提供：


```python
def inplace_attn_softmax(x):
    """
    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND。

    返回:
        Tensor: 输出张量，范围[0, 1], 数据类型与输入一致，数据格式支持ND。
    """
```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.randn(8, 1024, dtype=torch.float32)

# 使用gelu执行计算
result = kernel_gen_ops.inplace_attn_softmax(x)
```
## 约束与限制

- 张量数据格式支持ND。


