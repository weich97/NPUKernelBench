# aclnnGroupNormSwishGrad

## 功能描述
---
### 算子功能
计算 `GroupNormSwish` 算子相对于其输入 `x`、`gamma` 和 `beta` 的梯度。此算子用于反向传播过程。

### 计算公式
---
`aclnnGroupNormSwishGrad` 算子接收来自下游的梯度 `dy`，以及前向传播中计算得到的中间结果 `mean`、`rstd`、原始输入 `x`、`gamma` 和 `beta`。它计算以下梯度：

- $\frac{\partial L}{\partial x}$ (梯度 `dxOut`)
- $\frac{\partial L}{\partial \gamma}$ (梯度 `dgammaOutOptional`)
- $\frac{\partial L}{\partial \beta}$ (梯度 `dbetaOutOptional`)

梯度的计算涉及到组归一化和 Swish 激活函数的链式法则，具体推导遵循其数学定义：

- **Swish 激活函数的梯度 ($ \frac{\partial y_{\text{final}}}{\partial y_{\text{GroupNorm}}} $) :**
  设 $y_{\text{GroupNorm}}$ 为 GroupNorm 的输出，Swish 函数为 $f(z) = z \cdot \sigma(\text{scale} \cdot z)$。
  则 $ \frac{\partial f}{\partial z} = \sigma(\text{scale} \cdot z) + z \cdot \sigma(\text{scale} \cdot z) \cdot (1 - \sigma(\text{scale} \cdot z)) \cdot \text{scale} $

- **GroupNorm 的梯度 ($ \frac{\partial y_{\text{GroupNorm}}}{\partial x}, \frac{\partial y_{\text{GroupNorm}}}{\partial \gamma}, \frac{\partial y_{\text{GroupNorm}}}{\partial \beta} $) :**
  这部分梯度与标准 Group Normalization 的反向传播相同。
  * $\frac{\partial L}{\partial \beta} = \sum_{elements \ in \ normalized \ dim} \frac{\partial L}{\partial y_{\text{final}}} \cdot \frac{\partial y_{\text{final}}}{\partial y_{\text{GroupNorm}}}$
  * $\frac{\partial L}{\partial \gamma} = \sum_{elements \ in \ normalized \ dim} \left( \frac{\partial L}{\partial y_{\text{final}}} \cdot \frac{\partial y_{\text{final}}}{\partial y_{\text{GroupNorm}}} \right) \cdot \frac{x - \text{mean}(x)}{\sqrt{\text{Var}(x) + \epsilon}}$
  * $\frac{\partial L}{\partial x}$ 涉及更复杂的链式法则，包括通过均值和方差的梯度路径。

最终的梯度是各项链式法则的组合。

## 接口定义
---
### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.group_norm_swish_grad()` 函数形式提供：

```python
def group_norm_swish_grad(
    dy: torch.Tensor,
    mean: torch.Tensor,
    rstd: torch.Tensor,
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    num_groups: int,
    data_format: str,
    swish_scale: float,
    dgamma_is_require: bool,
    dbeta_is_require: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    计算 GroupNormSwish 算子相对于其输入 x, gamma, beta 的梯度。

    参数:
        dy (Tensor): 损失相对于 GroupNormSwish 输出的梯度。形状与 x 相同。
        mean (Tensor): GroupNorm 前向传播中计算的每组均值。形状通常为 (N, num_groups)。
        rstd (Tensor): GroupNorm 前向传播中计算的每组标准差的倒数。形状通常为 (N, num_groups)。
        x (Tensor): GroupNormSwish 的原始输入张量。
        gamma (Tensor): GroupNorm 的缩放参数。
        beta (Tensor): GroupNorm 的偏移参数。
        num_groups (int): 通道分组数。
        data_format (str): 输入数据格式，例如 "NCHW"。
        swish_scale (float): Swish 激活函数中的缩放因子。
        dgamma_is_require (bool): 指示是否需要计算 dgamma 的布尔标志。
        dbeta_is_require (bool): 指示是否需要计算 dbeta 的布尔标志。

    返回:
        Tuple[Tensor, Optional[Tensor], Optional[Tensor]]: 包含三个张量的元组：
            dxOut (Tensor): 损失相对于 x 的梯度，与 x 形状和数据类型相同。
            dgammaOutOptional (Optional[Tensor]): 损失相对于 gamma 的梯度。如果 dgamma_is_require 为 False，则为 None。
            dbetaOutOptional (Optional[Tensor]): 损失相对于 beta 的梯度。如果 dbeta_is_require 为 False，则为 None。
    """
```
## 使用案例

```python
import torch
import kernel_gen_ops

# 模拟 GroupNormSwishGrad 的输入
batch_size = 2
num_channels = 32
num_groups = 8
input_spatial_dims = (4, 4) # H, W
x_shape = (batch_size, num_channels) + input_spatial_dims

dtype = torch.float32
swish_scale = 1.0

dy = torch.randn(x_shape, dtype=dtype)
mean_rstd_shape = (batch_size, num_groups)
mean = torch.randn(mean_rstd_shape, dtype=dtype)
rstd = torch.randn(mean_rstd_shape, dtype=dtype) + 1e-3 # Ensure positive
x = torch.randn(x_shape, dtype=dtype)
gamma = torch.randn(num_channels, dtype=dtype)
beta = torch.randn(num_channels, dtype=dtype)

dgamma_is_require = True
dbeta_is_require = True

# 调用 GroupNormSwishGrad 算子
dx, dgamma, dbeta = kernel_gen_ops.group_norm_swish_grad(
    dy, mean, rstd, x, gamma, beta,
    num_groups, "NCHW", swish_scale,
    dgamma_is_require, dbeta_is_require
)

# 另一个案例：不需要 dgamma 和 dbeta
dx_only, dgamma_none, dbeta_none = kernel_gen_ops.group_norm_swish_grad(
    dy, mean, rstd, x, gamma, beta,
    num_groups, "NCHW", swish_scale,
    False, False # Set dgamma_is_require and dbeta_is_require to False
)
```

## 约束与限制
- **功能维度**
  * **数据格式**: Supports ND format.
  * **Data Types**: `dy`, `mean`, `rstd`, `x`, `gamma`, and `beta` must be **floating-point types** (float16, float32, bfloat16).
  * **Input Shape Consistency**:
    * `dy` and `x` must have the same **shape** and **data type**.
    * `mean` and `rstd` typically have a shape of `(N, num_groups)`, where `N` is the batch size.
    * `gamma` and `beta` must have a shape of `(C,)`, where `C` is the number of channels of `x` (`x.shape[1]`).
  * **Channels and Groups**: The number of channels of `x` (`x.shape[1]`) must be **divisible** by `num_groups`.
  * **`rstd` Value**: `rstd` should be a **positive** value.
  * **Data Format String**: `data_format` currently supports an empty string, indicating the default ND format.
  * **Output Shape and Type**:
    * `dxOut` will have the same shape and data type as `x`.
    * `dgammaOutOptional` will have the same shape and data type as `gamma` (if its calculation is required).
    * `dbetaOutOptional` will have the same shape and data type as `beta` (if its calculation is required).