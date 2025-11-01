# aclnnGeGluV2

## 功能描述

### 算子功能
`aclnnGeGluV2` 对输入张量的最后一维进行拆分，然后对前一半执行 GELU 激活函数，结果与后一半逐元素相乘，生成输出张量。

### 计算公式

对于输入张量 $x \in \mathbb{R}^{*, 2d}$，沿最后一维均匀拆分为两个张量 $x_1 \in \mathbb{R}^{*, d}$ 和 $x_2 \in \mathbb{R}^{*, d}$：

$$
y = \mathrm{GELU}(x_1) \odot x_2
$$

其中 $\mathrm{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2} \left( 1 + \mathrm{erf}\left( \frac{x}{\sqrt{2}} \right) \right)$，$\odot$ 表示逐元素相乘。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.ge_glu_v2()` 函数形式提供：

```python
def ge_glu_v2(x: Tensor) -> Tuple[Tensor, Tensor]:
    """
    对输入张量执行 GeGluV2 操作。

    参数:
        x (Tensor): 输入 Device 侧张量，最后一维长度必须为偶数。支持的数据类型包括：
                    torch.float16、torch.bfloat16、torch.float。
                    支持非连续 Tensor，shape 维度不超过 8 维，数据格式支持 ND。

    返回:
        Tuple[Tensor, Tensor]: 包含两个输出张量 (out, out_gelu)。
            - out (Tensor): GeLU 激活后的前半部分与后半部分逐元素相乘的结果，形状为 [..., d]，数据类型与输入相同。
            - out_gelu (Tensor): 对输入张量的前半部分执行 GELU 激活的结果，形状为 [..., d]，数据类型与输入相同。

    注意:
        - 输入张量最后一维必须可以均匀拆分为两部分
        - 输出张量的形状为输入张量的最后一维减半后的形状
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

# 执行 GeGluV2 操作
out = kernel_gen_ops.ge_glu_v2(x)

print("Shape of x:", x.shape)
print("Shape of out:", out.shape)
print("Shape of out_gelu:", out_gelu.shape)
```

### 约束与限制
无