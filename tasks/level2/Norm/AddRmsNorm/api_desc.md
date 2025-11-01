# aclnnAddRmsNorm

## 功能描述
### 算子功能
对两个输入张量 `x1` 和 `x2` 执行逐元素相加操作，然后对相加结果进行均方根归一化 (RMS Norm) 操作。

### 计算公式
$$x_{sum} = x1 + x2$$

$$y = {{x_{sum}}\over\sqrt {Mean(x_{sum}^2)+eps}} * \gamma$$

其中 $rstd = {{1}\over\sqrt {Mean(x_{sum}^2)+eps}}$，则 $y = x_{sum} * rstd * \gamma$。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.add_rms_norm()` 函数形式提供：

```python
def add_rms_norm(x1, x2, gamma, epsilon):
    """
    对两个输入张量执行逐元素相加，然后进行均方根归一化（RMS Norm）操作。

    参数:
        x1 (Tensor): 第一个输入Device侧张量。
        x2 (Tensor): 第二个输入Device侧张量，与 `x1` 具有相同的形状和数据类型。
        gamma (Tensor): 缩放参数张量，与 `x1` 需要归一化的维度的维度值相同，用于 RMS 归一化。
        epsilon (double): 公式中的输入 `eps`，添加到分母中的值，以确保数值稳定。

    返回:
        List[Tensor]: 包含三个张量的列表：
                      - 第一个张量 (`yOut`) 是 `x1` 和 `x2` 相加后进行 RMS 归一化的结果。
                      - 第二个张量 (`rstdOut`) 是计算过程中的均方根的倒数。
                      - 第三个张量 (`xOut`) 是 `x1` 和 `x2` 相加的原始结果。
    """
```
## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量和参数
input_shape = [128, 256]
gamma_shape = [256] # Assuming last dim is normalized
dtype = torch.float32
epsilon = 1e-6

x1 = torch.randn(input_shape, dtype=dtype)
x2 = torch.randn(input_shape, dtype=dtype)
gamma = torch.randn(gamma_shape, dtype=dtype)

# 使用 add_rms_norm 执行操作
y_out, rstd_out, x_out = kernel_gen_ops.add_rms_norm(x1, x2, gamma, epsilon)

```

## 约束与限制
- **功能维度**
  * 数据格式支持：ND。
  * 张量形状维度不高于8维。
  * `x1`、`x2`、`yOut`、`xOut` 的形状、数据类型和数据格式必须一致。
  * `gamma` 的形状必须与归一化维度一致。
  * `rstdOut` 的形状应与 `x1` 的形状匹配，shape与x前几维保持一致，前几维表示不需要norm的维度。
  * `x1`、`x2`、`gamma` 的数据类型支持 **FLOAT16**、**FLOAT**、**BFLOAT16**。
  * `epsilon` 必须为浮点数，且通常取值如 1e-5 或 1e-6。