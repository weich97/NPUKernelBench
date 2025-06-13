# aclnnDeepNorm

## 功能描述
### 算子功能
DeepNorm 算子用于深度归一化操作，它将两个输入张量 `x` 和 `gx` 进行加权求和，然后对结果进行层归一化（Layer Normalization）操作。

### 计算公式
$$x_{sum} = x \cdot \alpha + gx$$

$$mean = Mean(x_{sum})$$

$$variance = Mean((x_{sum} - mean)^2)$$

$$rstd = {{1}\over\sqrt {variance + eps}}$$

$$y = \gamma \cdot (x_{sum} - mean) \cdot rstd + \beta$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.deep_norm()` 函数形式提供：

```python
def deep_norm(x, gx, beta, gamma, alpha, epsilon):
    """
    对输入张量执行 DeepNorm 操作。

    参数:
        x (Tensor): 第一个输入Device侧张量。
        gx (Tensor): 第二个输入Device侧张量，与 `x` 具有相同的形状和数据类型。
        beta (Tensor): 偏移参数张量，用于层归一化。形状与归一化维度一致。
        gamma (Tensor): 缩放参数张量，用于层归一化。形状与归一化维度一致。
        alpha (double): 用于 `x` 加权求和的标量系数。
        epsilon (double): 公式中的输入 `eps`，添加到方差中以确保数值稳定。

    返回:
        List[Tensor]: 包含三个张量的列表：
                      - 第一个张量 (`meanOut`) 是计算过程中的均值。
                      - 第二个张量 (`rstdOut`) 是计算过程中的标准差的倒数。
                      - 第三个张量 (`yOut`) 是 DeepNorm 的最终输出。
    """
    import torch
    import kernel_gen_ops

    # 创建输入张量和参数
    input_shape = [3, 1, 4]
    normalized_shape = [4] # Assuming normalization over the last dimension
    dtype = torch.float32
    alpha = 0.3
    epsilon = 1e-6

    x = torch.randn(input_shape, dtype=dtype)
    gx = torch.randn(input_shape, dtype=dtype)
    beta = torch.randn(normalized_shape, dtype=dtype)
    gamma = torch.randn(normalized_shape, dtype=dtype)

    # 使用 deep_norm 执行操作
    mean_out, rstd_out, y_out = kernel_gen_ops.deep_norm(x, gx, beta, gamma, alpha, epsilon)

    print("Mean Output shape:", mean_out.shape)
    print("RSTD Output shape:", rstd_out.shape)
    print("DeepNorm Output (y_out) shape:", y_out.shape)

## 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * 张量形状维度不高于**8维**。
  * 'x`、`gx`、`yOut` 的形状和数据类型必须一致。
  * 'beta` 和 `gamma` 的形状必须与**归一化维度**一致。
  * 'meanOut` 和 `rstdOut` 的形状应与 `x` 的形状匹配，但在归一化维度上为 **1**。
  * 'x`、`gx`、`beta`、`gamma` 的数据类型支持 **FLOAT16**、**FLOAT**、**BFLOAT16**、**INT32**、**INT64**、**DOUBLE**、**INT8** (实际支持取决于 NPU 硬件)。
  * 'alpha` 和 `epsilon` 必须为**浮点数**。