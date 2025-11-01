# aclnnSwiGlu

## 功能描述

### 算子功能
`aclnnSwiGlu` 对输入张量按指定dim维度(如果没有指定，则默认为最后一维)进行拆分，然后对前一半执行 SiLU 激活函数，结果与后一半逐元素相乘，生成输出张量。

### 计算公式

对于输入张量 $x \in \mathbb{R}^{*, *, \dots, 2d}$，沿指定 dim 维度（如果没有指定，则默认为最后一维）均匀拆分为两个张量 $x_1$ 和 $x_2$，使得

$$
y = \mathrm{SiLU}(x_1) \odot x_2
$$

其中 $\mathrm{SiLU}(x) = \frac{x}{1 + e^{-x}}$，$\odot$ 表示逐元素相乘。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.swi_glu()` 函数形式提供：

```python
def swi_glu(x: Tensor, dim: int = -1) -> Tensor:
    """
    对输入张量执行 SwiGlu 操作。

    参数:
        x (Tensor): 输入 Device 侧张量，且shape必须在入参dim对应维度(如果指定，否则为最后一维)上可以整除2。支持的数据类型包括：
                    torch.float16、torch.bfloat16、torch.float。
                    支持非连续 Tensor，shape 维度不超过 8 维，数据格式支持 ND。

    注意:
        - 输入张量在dim对应维度(如果指定，否则为最后一维)必须可以均匀拆分为两部分
        - 输出张量的形状为输入张量在该维度减半后的形状
        - 支持非连续 Tensor
        - 支持的最大维度为 8 维
        - 输出张量与输入张量具有相同的数据类型和数据格式
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 构造输入张量，最后一维可被2整除
x = torch.randn(2, 4, dtype=torch.float)

# 执行 SwiGlu 操作
y = kernel_gen_ops.swi_glu(x, dim=-1)
```

## 约束与限制

无