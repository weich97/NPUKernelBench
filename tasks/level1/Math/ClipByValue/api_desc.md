# aclnnClipByValue

## 功能描述

### 算子功能
用于将一个张量值剪切到指定的最小值和最大值：

y=min(max(x,clip_value_min),clip_value_max)

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.clip_by_value()` 函数形式提供：


```python
def clip_by_value(input, clip_value_min, clip_value_max):
    """
    实现自定义数值截断操作，将输入张量中的每个元素限制在指定的最小值和最大值之间。

    参数:
        input (Tensor): 输入张量，支持任意ND格式。
        clip_value_min (Tensor): shape为1，对于input中的每个元素，如果其值低于该下界则被截断为该值。
        clip_value_max (Tensor): 指定的上界。shape为1，对于input中的每个元素，如果其值高于该上界则被截断为该值。
    返回:
        Tensor: 裁剪后的张量，形状与输入张量一致，数据类型与输入相同。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

input = (torch.rand(8, 2048, dtype=torch.float32) * 9) + 1
clip_value_min = (torch.rand(1, dtype=torch.float32) * 2) + 1
clip_value_max = (torch.rand(1, dtype=torch.float32) * 6) + 4

result = kernel_gen_ops.clip_by_value(input, clip_value_min, clip_value_max)
```
## 约束与限制

- 张量数据格式支持ND。


