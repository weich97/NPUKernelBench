# aclnnDeepNormGrad

## 功能描述
### 算子功能
DeepNorm 算子的反向计算，用于计算对输入张量 `x`、`gx` 以及参数 `beta` 和 `gamma` 的梯度。

### 计算公式
假设 $D$ 是归一化维度的大小，且归一化发生在张量的末尾维度。

$$
d_{gx_i} = tmpone_i \cdot rstd + \frac{2}{D} \cdot d_{var} \cdot tmptwo_i + {\frac{1}{D}} \cdot d_{mean}
$$

$$
d_{x_i} = alpha \cdot d_{gx_i}
$$

$$
d_{beta} = \sum_{i=1}^{N} d_{y_i}
$$

$$
d_{gamma} = \sum_{i=1}^{N} d_{y_i} \cdot rstd \cdot {tmptwo}_i
$$

其中：
$$
tmpone_i = d_{y_i} \cdot gamma
$$

$$
tmptwo_i = alpha \cdot x_i + {gx}_i - mean
$$

$$
d_{var} = \sum_{i=1}^{N} (-0.5) \cdot {tmpone}_i \cdot {tmptwo}_i \cdot {rstd}^3
$$

$$
d_{mean} = \sum_{i=1}^{N} (-1) \cdot {tmpone}_i \cdot rstd
$$
上述求和 $\sum$ 发生在归一化维度上。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.deep_norm_grad()` 函数形式提供：

```python
def deep_norm_grad(dy, x, gx, gamma, mean, rstd, alpha):
    """
    计算 DeepNorm 算子的反向传播梯度。

    参数:
        dy (Tensor): DeepNorm 前向输出 `y` 的梯度张量。
        x (Tensor): DeepNorm 前向输入张量 `x`。
        gx (Tensor): DeepNorm 前向输入张量 `gx`。
        gamma (Tensor): DeepNorm 前向的缩放参数张量。
        mean (Tensor): DeepNorm 前向计算得到的均值张量。
        rstd (Tensor): DeepNorm 前向计算得到的标准差的倒数张量。
        alpha (double): DeepNorm 前向中 `x` 的加权系数。

    返回:
        List[Tensor]: 包含四个张量的列表：
                      - 第一个张量 (`dxOut`) 是对 `x` 的梯度。
                      - 第二个张量 (`dgxOut`) 是对 `gx` 的梯度。
                      - 第三个张量 (`dbetaOut`) 是对 `beta` 的梯度。
                      - 第四个张量 (`dgammaOut`) 是对 `gamma` 的梯度。
    """
```
## 使用案例

```python
import torch
import kernel_gen_ops

# 假设输入和参数
input_shape = [3, 1, 4]
normalized_shape = [4]
dtype = torch.float32
alpha = 0.3
epsilon = 1e-6

# 模拟 DeepNorm 前向传播的输入和中间输出
dy = torch.randn(input_shape, dtype=dtype)
x = torch.randn(input_shape, dtype=dtype)
gx = torch.randn(input_shape, dtype=dtype)
beta = torch.randn(normalized_shape, dtype=dtype) # beta is not input to grad, but its shape is needed
gamma = torch.randn(normalized_shape, dtype=dtype)

# Calculate mean and rstd from a simulated DeepNorm forward pass
x_add = x * alpha + gx
reduction_dims = tuple(range(x_add.dim() - len(normalized_shape), x_add.dim()))
mean = x_add.mean(dim=reduction_dims, keepdim=True)
variance = (x_add - mean).pow(2).mean(dim=reduction_dims, keepdim=True)
rstd = torch.rsqrt(variance + epsilon)

# 使用 deep_norm_grad 计算反向梯度
dx, dgx, dbeta, dgamma = kernel_gen_ops.deep_norm_grad(dy, x, gx, gamma, mean, rstd, alpha)

```

## 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * 张量形状维度不高于**8维**。
  * `dy`、`x`、`gx`、`dxOut`、`dgxOut` 的形状和数据类型必须一致。
  * `gamma`、`dbetaOut`、`dgammaOut` 的形状和数据类型必须一致。
  * `mean` 和 `rstd` 的形状应与 `x` 匹配，但在归一化维度上为 **1**。
  * `dy`、`x`、`gx`、`gamma`的数据类型支持 **FLOAT16**、**FLOAT**、**BFLOAT16**。
  * `rstd`、`mean`的数据类型支持**FLOAT32**。