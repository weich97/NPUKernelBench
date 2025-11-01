# aclnnGroupNormSwish

## 功能描述
---
### 算子功能
对输入张量 `x` 执行组归一化（Group Normalization），并根据 `activateSwish` 参数选择是否接着应用 Swish 激活函数。它返回归一化后的输出、每组的均值和标准差的倒数。

### 计算公式
---
- **GroupNorm:**
  记 $E[x]$ 代表输入 $x$ 在每个组内的均值，$Var[x]$ 代表每个组内的方差。
  $$
  E[x]_g = \frac{1}{|S_g|} \sum_{i \in S_g} x_i \\
  Var[x]_g = \frac{1}{|S_g|} \sum_{i \in S_g} (x_i - E[x]_g)^2
  $$
  其中 $S_g$ 代表属于组 $g$ 的元素集合。
  则组归一化后的输出 $yOut_g$、均值 $meanOut_g$ 和标准差的倒数 $rstdOut_g$ 为：
  $$
  \left\{
  \begin{array} {rcl}
  yOut_g & = & \frac{x_g - E[x]_g}{\sqrt{Var[x]_g + \text{eps}}} \cdot \gamma_g + \beta_g \\
  meanOut & = & E[x] \\
  rstdOut & = & \frac{1}{\sqrt{Var[x] + \text{eps}}} \\
  \end{array}
  \right.
  $$
  其中 $\gamma_g$ 和 $\beta_g$ 是对应组的缩放和偏移参数。

- **Swish (SiLU):**
  当 `activateSwish` 为 `True` 时，Swish 激活函数会被应用到 GroupNorm 的输出 `yOut` 上。
  $$
  yOut_{\text{final}} = yOut \cdot \frac{1}{1 + e^{-\text{swishScale} \cdot yOut}}
  $$

## 接口定义
---
### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.group_norm_swish()` 函数形式提供：

```python
def group_norm_swish(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    num_groups: int,
    data_format_optional: str,
    eps: float,
    activate_swish: bool,
    swish_scale: float
) -> List[torch.Tensor]:
    """
    对输入张量 x 执行组归一化，并可选地应用 Swish 激活函数。

    参数:
        x (Tensor): 输入张量。通常为 (N, C, H, W) 或 (N, C)。
        gamma (Tensor): 缩放参数张量，形状为 (C,)，用于 GroupNorm。
        beta (Tensor): 偏移参数张量，形状为 (C,)，用于 GroupNorm。
        num_groups (int): 要将通道 C 划分为的组数。C 必须能被 num_groups 整除。
        data_format_optional (str): 输入数据格式，例如 "NCHW" 或 "NHWC"。通常为空字符串表示默认 ND 格式。
        eps (float): 添加到方差中的小值，以确保数值稳定性。
        activate_swish (bool): 是否在 GroupNorm 后应用 Swish 激活函数。
        swish_scale (float): Swish 激活函数中的缩放因子。

    返回:
        List[Tensor]: 包含三个张量的List：
            yOut (Tensor): GroupNorm 后的输出，如果 activate_swish 为 True 则为 Swish 激活后的输出。
                           形状与 x 相同。
            meanOut (Tensor): 每组的均值。形状通常为 (N, num_groups)。
            rstdOut (Tensor): 每组标准差的倒数。形状通常为 (N, num_groups)。
    """
```
## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量和参数
x_shape = [4, 32, 10, 10]
num_channels = x_shape[1]
num_groups = 8
eps = 1e-5
activate_swish = True
swish_scale = 1.0

x = torch.randn(x_shape, dtype=torch.float32)
gamma = torch.ones(num_channels, dtype=torch.float32)
beta = torch.zeros(num_channels, dtype=torch.float32)

# 调用 group_norm_swish 算子
y_out, mean_out, rstd_out = kernel_gen_ops.group_norm_swish(
    x, gamma, beta, num_groups, "", eps, activate_swish, swish_scale
)

# 另一个案例：不带 Swish
activate_swish_no = False
y_out_no_swish, mean_out_no_swish, rstd_out_no_swish = kernel_gen_ops.group_norm_swish(
    x, gamma, beta, num_groups, "", eps, activate_swish_no, swish_scale
)
```

## 约束与限制
- **功能维度**
  * **数据格式**：支持 ND 格式。
  * **数据类型**：`x`、`gamma`、`beta` 必须是**浮点类型** (float16, float32, bfloat16)。
  * **通道与分组**：`x` 的通道数 (`x.shape[1]`) 必须能被 `num_groups` **整除**。
  * **参数形状**：`gamma` 和 `beta` 的形状必须为 `(C,)`，其中 `C` 是 `x` 的通道数 (`x.shape[1]`)。
  * **Epsilon 值**：`eps` 必须是**正数**。
  * **数据格式字符串**：`data_format_optional` 当前支持空字符串，表示默认的 ND 格式。
  * **输出形状与类型**：
    * `yOut` 的形状和数据类型与 `x` 相同。
    * `meanOut` 和 `rstdOut` 的形状通常为 `(N, num_groups)`，数据类型与 `x` 相同。