# aclnnDequantSwigluQuant

## 功能描述

### 算子功能
`aclnnDequantSwigluQuant` 对输入张量 `x` 执行 Dequantization, SwiGLU 激活和 Quantization 操作。具体来说，它首先对输入进行反量化（如果提供了量化参数），然后将张量的最后一维拆分为两部分，对前半部分执行 SiLU 激活，结果与后半部分逐元素相乘。最后，对结果进行量化，并输出量化后的张量和量化比例因子。

### 计算公式

假设输入张量为 $x \in \mathbb{R}^{*, 2d}$。

1. **Dequantization (如果提供了 `quant_scale_optional` 和 `quant_offset_optional`)**:
   $$
   x_{dequant} = (x - \mathrm{offset}) \times \mathrm{scale}
   $$

2. **SwiGLU**:
   将 $x_{dequant}$ (或原始 $x$ 如果没有量化参数) 沿最后一维均匀拆分为两个张量 $x_1 \in \mathbb{R}^{*, d}$ 和 $x_2 \in \mathbb{R}^{*, d}$。
   $$
   y_{float} = \mathrm{SiLU}(x_1) \odot x_2
   $$
   其中 $\mathrm{SiLU}(x) = \frac{x}{1 + e^{-x}}$，$\odot$ 表示逐元素相乘。`activateLeft` 参数控制 SiLU 激活应用在哪一半 (默认为右半部分，但公式中是对左半部分激活，请注意 `activateLeft` 参数的行为)。

3. **Quantization (如果提供了 `quant_scale_optional` 和 `quant_offset_optional`)**:
   $$
   y_{int8} = \mathrm{Quantize}(y_{float}, \mathrm{quant\_scale}, \mathrm{quant\_offset})
   $$
   $$
   \mathrm{scale}_{out} = \mathrm{quant\_scale}
   $$

   如果没有提供量化参数，则输出 $y = y_{float}$ 且 $\mathrm{scale}_{out}$ 可能为单位值或根据具体实现而定。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.dequant_swiglu_quant()` 函数形式提供：

```python
def dequant_swiglu_quant(x: Tensor, weight_scale_optional: Optional[Tensor] = None,
                         activation_scale_optional: Optional[Tensor] = None,
                         bias_optional: Optional[Tensor] = None,
                         quant_scale_optional: Optional[Tensor] = None,
                         quant_offset_optional: Optional[Tensor] = None,
                         group_index_optional: Optional[Tensor] = None,
                         activate_left: bool = False,
                         quant_mode: str = "static") -> List[Tensor]:
    """
    对输入张量执行 Dequantization, SwiGLU 和 Quantization 操作。

    参数:
        x (Tensor): 输入 Device 侧张量，最后一维长度必须为偶数。支持的数据类型包括：
                    torch.float16、torch.bfloat16、torch.int32。
                    支持非连续 Tensor，shape 维度不超过 8 维，数据格式支持 ND。
        weight_scale_optional (Optional[Tensor]): 可选的权重缩放张量。
        activation_scale_optional (Optional[Tensor]): 可选的激活缩放张量。
        bias_optional (Optional[Tensor]): 可选的偏置张量。
        quant_scale_optional (Optional[Tensor]): 可选的量化比例因子张量。
        quant_offset_optional (Optional[Tensor]): 可选的量化偏移张量。
        group_index_optional (Optional[Tensor]): 可选的分组索引张量。
        activate_left (bool): 指示 SiLU 激活是否应用于左半部分 (默认为 False，可能应用于右半部分，请参考实际行为)。
        quant_mode (str): 量化模式，例如 "static", "per_tensor", "per_channel"。

    返回:
        List[Tensor]: 包含两个输出张量 (y, scale)。
            - y (Tensor): 量化后的输出张量，形状为 [..., d]，数据类型为 torch.int8 (如果执行量化)。
            - scale (Tensor): 输出的量化比例因子，形状可能为 [...] 或 [1] 等，取决于量化模式。

    注意:
        - 输入张量最后一维必须可以均匀拆分为两部分。
        - 输出张量 y 的数据类型取决于是否执行量化。
        - scale 张量的形状和值取决于量化模式。
        - 支持非连续 Tensor。
        - 支持的最大维度为 8 维。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 构造输入张量，最后一维可被2整除
x = torch.randn(2, 4, dtype=torch.float16)
scale = torch.tensor([0.1], dtype=torch.float32)
offset = torch.tensor([0], dtype=torch.float32)

# 执行 DequantSwigluQuant 操作
y, output_scale = kernel_gen_ops.dequant_swiglu_quant(x, quant_scale_optional=scale, quant_offset_optional=offset)

```

## 约束与限制

- **维度要求**：`x` 必须为 ≥2 维张量，且最后一维长度是 2 的倍数。  
- **量化参数形状**：  
  - `quantMode == static`：`quantScaleOptional` 与 `quantOffsetOptional` 为 1 维，长度为 1。  
  - `quantMode == dynamic`：二者为 1 维，长度为 `x` 最后一维的一半。