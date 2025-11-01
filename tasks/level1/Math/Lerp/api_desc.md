# aclnnLerp

## 功能描述

### 算子功能
对两个张量以`start`，`end`做线性插值，`weight`是权重值，`lerp`根据权重`weight`在`start`和`end`两个值之间进行插值，并将结果返回到输出张量。

### 计算公式

$$
y=start+weight*(end−start)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.lerp()` 函数形式提供：


```python
def lerp(start, end, weight) -> torch.Tensor:
    """
    实现自定义线性插值（lerp）操作。

    参数:
        start: 起始张量，Device 侧张量，数据格式支持 ND。
        end: 结束张量，形状与 start 相同，或可广播为 start 形状。
        weight: 插值因子，范围通常在 [0, 1]，形状与 start 相同，或可广播为 start 形状。

    返回:
        Tensor: 输出张量，线性插值得到的结果。数据类型与 start 相同，数据格式支持 ND。


    注意:
        - 支持广播操作；
        - 支持 ND 格式张量；
        - weight 可为标量或与 start/end 同形状的张量。
    """

```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
start = torch.randn(8. 2048, dtype=torch.float32)
end = torch.randn(8. 2048, dtype=torch.float32)
weight = torch.randn(8. 2048, dtype=torch.float32)

# 使用 lerp 执行插值
result = kernel_gen_ops.lerp(start, end, weight)
```
## 约束与限制

- 张量数据格式支持ND。


