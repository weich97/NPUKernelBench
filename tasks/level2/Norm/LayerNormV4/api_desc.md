# aclnnLayerNormV4

## 功能描述
### 算子功能
对指定层进行均值为0、标准差为1的归一化计算。此算子实现了经典的 Layer Normalization。

### 计算公式
$$
out = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + eps}} \cdot w + b
$$

其中：
* $\mathrm{E}[x]$ 表示输入张量 $x$ 在归一化维度上的均值。
* $\mathrm{Var}[x]$ 表示输入张量 $x$ 在归一化维度上的方差。
* $eps$ 是一个添加到方差上的小值，用于数值稳定性，防止除以零。
* $w$ 是可选的缩放参数 (weight)。
* $b$ 是可选的偏移参数 (bias)。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.layer_norm_v4()` 函数形式提供：

```python
def layer_norm_v4(input_tensor, normalized_shape, weight, bias, eps):
    """
    对输入张量执行 Layer Normalization V4 操作。

    参数:
        input_tensor (Tensor): 输入Device侧张量。
        normalized_shape (Tuple[int]): 一个表示要归一化的维度的列表或元组。
                                      例如，如果 input_tensor 的形状是 (N, C, H, W)，
                                      normalized_shape=(C, H, W) 将对整个特征维度进行归一化。
        weight (Optional[Tensor]): 可选的缩放参数张量，形状与 normalized_shape 相同。
        bias (Optional[Tensor]): 可选的偏移参数张量，形状与 normalized_shape 相同。
        eps (double): 添加到方差中的小值，用于数值稳定性。

    返回:
        List[Tensor]: 包含三个张量的列表：
                      - 第一个张量 (`out`) 是归一化后的输出张量。
                      - 第二个张量 (`meanOutOptional`) 是计算过程中的均值张量。
                      - 第三个张量 (`rstdOutOptional`) 是计算过程中的标准差倒数张量。
    """
```
## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量和参数
input_shape = [128, 256]
normalized_shape = [256] # Example: normalizing the last dimension
dtype = torch.float32
eps = 1e-5

input_tensor = torch.randn(input_shape, dtype=dtype)
weight = torch.randn(normalized_shape, dtype=dtype)
bias = torch.randn(normalized_shape, dtype=dtype)

# 使用 layer_norm_v4 执行操作
output, mean_val, rstd_val = kernel_gen_ops.layer_norm_v4(input_tensor, normalized_shape, weight, bias, eps)

# Example without weight and bias
output_no_affine, _, _ = kernel_gen_ops.layer_norm_v4(input_tensor, normalized_shape, None, None, eps)
```
## 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * 张量形状维度不高于**8维**。
  * `input` 和 `out` 的形状和数据类型必须一致。
  * `normalized_shape` 必须是 `input` 张量的一个或多个**连续尾部维度**, 长度小于等于输入input的长度，不支持为空。
  * `weightOptional` 和 `biasOptional` 必须是与 `normalized_shape` 形状相同的 **1D 张量**，或者为 `None`。
  * `meanOutOptional` 和 `rstdOutOptional` 的形状应与 `input` 匹配，但在归一化维度上为 **1**。
  * `input`、`weightOptional`、`biasOptional` 的数据类型支持 **FLOAT16**、**FLOAT**、**BFLOAT16** (实际支持取决于 NPU 硬件)。
  * `eps` 必须为**浮点数**，且通常取值如 1e-5 或 1e-6。