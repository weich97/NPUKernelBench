# aclnnAddLayerNormGrad

## 功能描述
### 算子功能
计算 `AddLayerNorm` 算子中相加操作和层归一化操作的梯度。此算子用于反向传播过程，接收损失关于 `AddLayerNorm` 输出的梯度 `dy`，并计算损失关于其输入 `x1`、`x2`、`gamma` 和 `beta` 的梯度。

### 计算公式
`AddLayerNorm` 的前向计算为：
$$
x_{combined} = x1 + x2 (+ bias) \\
y = \frac{x_{combined} - \text{mean}(x_{combined})}{\sqrt{\text{Var}(x_{combined}) + \epsilon}} \cdot \gamma + \beta
$$
`aclnnAddLayerNormGrad` 旨在计算 `dy`（即 $\frac{\partial L}{\partial y}$）到 $\frac{\partial L}{\partial x1}$、$\frac{\partial L}{\partial x2}$、$\frac{\partial L}{\partial \gamma}$ 和 $\frac{\partial L}{\partial \beta}$ 的梯度。

核心梯度公式（简化表示）：
对于 `dy` 到 `dx` 的反向传播：
$$
\frac{\partial L}{\partial x_{combined}} = \frac{\partial L}{\partial y} \cdot \gamma \cdot \frac{1}{\sqrt{\text{Var}(x_{combined}) + \epsilon}} - \frac{1}{N} \sum (\dots)
$$
其中，$\sum (\dots)$ 包含与均值和方差相关的复杂项。
$N$ 为归一化维度的大小。

* $\frac{\partial L}{\partial \gamma} = \sum_{elements \ in \ normalized \ dim} \frac{\partial L}{\partial y} \cdot (x_{combined} - \text{mean}(x_{combined})) \cdot \frac{1}{\sqrt{\text{Var}(x_{combined}) + \epsilon}}$
* $\frac{\partial L}{\partial \beta} = \sum_{elements \ in \ normalized \ dim} \frac{\partial L}{\partial y}$
* $\frac{\partial L}{\partial x1} = \frac{\partial L}{\partial x_{combined}} (+ \frac{\partial L}{\partial \text{dsum}})$
* $\frac{\partial L}{\partial x2} = \frac{\partial L}{\partial x_{combined}} (+ \frac{\partial L}{\partial \text{dsum}})$

如果存在 `dsum` 输入，其梯度将直接加到 `dx1` 和 `dx2` 上。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.add_layer_norm_grad()` 函数形式提供：

```python
def add_layer_norm_grad(
    dy: torch.Tensor,
    x1: torch.Tensor,
    x2: torch.Tensor,
    rstd: torch.Tensor,
    mean: torch.Tensor,
    gamma: torch.Tensor,
    dsum: Optional[torch.Tensor]
) -> List[torch.Tensor]:
    """
    计算 AddLayerNorm 算子中输入 x1, x2, gamma, beta 的梯度。

    参数:
        dy (Tensor): 损失相对于 AddLayerNorm 输出的梯度。形状与 AddLayerNorm 输出相同。
        x1 (Tensor): AddLayerNorm 的第一个输入张量。
        x2 (Tensor): AddLayerNorm 的第二个输入张量，与 x1 形状和数据类型相同。
        rstd (Tensor): AddLayerNorm 前向传播中计算的 (方差 + epsilon) 的平方根倒数，
                       形状为批次维度 + 归一化维度为1的张量 (e.g., (B, 1, 1) 或 (B, H, 1))。
        mean (Tensor): AddLayerNorm 前向传播中计算的均值，
                       形状为批次维度 + 归一化维度为1的张量 (e.g., (B, 1, 1) 或 (B, H, 1))。
        gamma (Tensor): LayerNorm的缩放参数张量，形状与归一化维度相同 (e.g., (C,) 或 (H, W))。
        dsum (Optional[Tensor]): 一个可选的梯度张量，用于与 dx 合并。
                                 如果存在，其形状与 x1/x2 相同。

    返回:
        List[Tensor]: 包含三个张量的List：
            dx (Tensor): 损失相对于 x1 和 x2 的梯度，与 x1 形状和数据类型相同。
            dgamma (Tensor): 损失相对于 gamma 的梯度，与 gamma 形状和数据类型相同。
            dbeta (Tensor): 损失相对于 beta 的梯度，与 gamma 形状和数据类型相同。
    """
```
## 使用案例

```python
import torch
import kernel_gen_ops

# 模拟 AddLayerNormGrad 的输入
batch_size = 2
input_dim = 16
normalized_dim = 16 # Last dimension for LayerNorm

dy = torch.randn(batch_size, input_dim, dtype=torch.float32)
x1 = torch.randn(batch_size, input_dim, dtype=torch.float32)
x2 = torch.randn(batch_size, input_dim, dtype=torch.float32)

# rstd 和 mean 的形状取决于归一化维度，对于 (B, N) 格式，它们是 (B, 1)
# 对于 (B, H, W) 且归一化W，它们是 (B, H, 1)
mean_rstd_shape = list(dy.shape[:-1]) + [1] * len([normalized_dim]) if isinstance(normalized_dim, int) else dy.shape[:-len(normalized_dim)] + [1] * len(normalized_dim)
rstd = torch.rand(mean_rstd_shape, dtype=torch.float32) + 1e-3
mean = torch.rand(mean_rstd_shape, dtype=torch.float32)

gamma = torch.randn(normalized_dim, dtype=torch.float32) # Gamma shape (N,)

dsum_present = True
dsum = torch.randn_like(dy) if dsum_present else None

# 调用 AddLayerNormGrad 算子
dx, dgamma, dbeta = kernel_gen_ops.add_layer_norm_grad(
    dy, x1, x2, rstd, mean, gamma, dsum
)

```

## 约束与限制
- **功能维度**
  * 数据格式支持：ND.
  * `dy`、`x1`、`x2`、`rstd`、`mean`、`gamma`、`dsum` 必须是**浮点类型** (float16, float32, bfloat16).
  * `dy`、`x1`、`x2`、`dsum` 必须具有相同的**形状**和**数据类型**.
  * `rstd` 和 `mean` 的形状应与 `x1` 的**批次维度**相同，且**归一化维度**为1 (例如，如果 `x1` 形状是 `(B, C, H, W)`，归一化在 `H, W` 维度，则 `rstd` 和 `mean` 形状为 `(B, C, 1, 1)`).
  * `gamma` 的形状必须与 `x1` 的**归一化维度**一致.
  * 输出 `dx` 的形状和数据类型与 `x1` 相同.
  * 输出 `dgamma` 和 `dbeta` 的形状和数据类型与 `gamma` 相同.