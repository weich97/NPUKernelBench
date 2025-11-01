# aclnnRmsNormGrad

## 功能描述
### 算子功能
计算 RMSNorm 算子的反向传播梯度，包括对输入张量 `x` 的梯度 `dx` 和对缩放参数 `gamma` 的梯度 `dgamma`。

### 计算公式
RMSNorm 的前向计算公式为：
$$y = {{x}\over\sqrt {Mean(x^2)+eps}} * \gamma$$
其中 $rstd = {{1}\over\sqrt {Mean(x^2)+eps}}$，则 $y = x * rstd * \gamma$

反向传播的梯度计算：
**对 $\gamma$ 的梯度 ($d\gamma$)：**
$$
d\gamma = \sum_{j \in \mathrm{reduction\_dims}} (dy \cdot x \cdot rstd)_j
$$
其中 $\text{reduction\_dims}$ 是除了 `gamma` 维度外的所有维度。

**对 $x$ 的梯度 ($dx$)：**
$$
dx = dy \cdot \gamma \cdot rstd
     - \left(
         \sum_{j \in \text{reduction\_dims}} (dy \cdot \gamma \cdot x \cdot rstd^3)_j
       \right)
     \cdot \frac{x}{N}
$$
其中 $N$ 是进行均方根归一化的维度大小（通常是最后一维或 `normalized_shape` 的乘积）。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.rms_norm_grad()` 函数形式提供：

```python
def rms_norm_grad(dy, x, rstd, gamma):
    """
    计算 RMSNorm 算子的反向传播梯度。

    参数:
        dy (Tensor): 输出 `y` 的梯度张量，与 `x` 具有相同的形状和数据类型。
        x (Tensor): RMSNorm 前向传播的输入张量。
        rstd (Tensor): RMSNorm 前向传播计算得到的均方根的倒数（Reciprocal Standard Deviation）张量。
                       通常形状为 `x` 的形状，但在被归一化的维度上为 1。
        gamma (Tensor): RMSNorm 前向传播的缩放参数张量。

    返回:
        List[Tensor]: 包含两个张量的列表，第一个是 `dx`（对 `x` 的梯度），第二个是 `dgamma`（对 `gamma` 的梯度）。
    """
```
## 使用案例
```python
import torch
import kernel_gen_ops

# 假设输入和参数
input_shape = [128, 256]
gamma_shape = [256]
dtype = torch.float32
epsilon = 1e-6

# 模拟 RMSNorm 前向传播的输入和输出
x = torch.randn(input_shape, dtype=dtype)
gamma = torch.randn(gamma_shape, dtype=dtype)
# 计算 rstd (通常是 RMSNorm 前向的中间输出)
rstd = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
rstd = rstd.view(x.shape[:-1] + (1,)).to(dtype=dtype) # Adjust rstd shape if needed

# 梯度输出 dy
dy = torch.randn(input_shape, dtype=dtype)

# 使用 rms_norm_grad 计算反向梯度
dx, dgamma = kernel_gen_ops.rms_norm_grad(dy, x, rstd, gamma)

```

## 约束与限制
- **功能维度**
  * 数据格式支持：ND。
  * 张量形状维度不高于8维。
  * 'dy`、`x`、`dx` 的形状、数据类型和数据格式必须一致。
  * 'rstd` 的形状应与 `x` 的形状匹配，但在归一化维度上为 1。
  * 'gamma` 和 `dgamma` 的形状、数据类型和数据格式必须一致。
  * 'dy`, `x`, `rstd`, `gamma` 的数据类型支持 FLOAT16、FLOAT、BFLOAT16、INT32、INT64、DOUBLE、INT8 (实际支持取决于 NPU 硬件)。
  * 'epsilon` 必须与前向传播 RMSNorm 中使用的值保持一致。