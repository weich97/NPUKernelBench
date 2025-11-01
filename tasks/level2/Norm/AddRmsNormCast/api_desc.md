# aclnnAddRmsNormCast

## 功能描述
### 算子功能
是将均方根归一化后的结果与输入数据相加，并涉及数据类型的转换。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.add_rms_norm_cast()` 函数形式提供：

```python
def add_rms_norm_cast(x1, x2, gamma, epsilon):
    """
    对输入张量执行均方根归一化（RMS Norm）操作，并对部分数据进行数据类型转换。

    参数:
        x1 (Tensor): 第一个输入Device侧张量。
        x2 (Tensor): 第二个输入Device侧张量，与 `x1` 具有相同的形状和数据类型。
        gamma (Tensor): 缩放参数张量，与 `x1` 需要归一化的维度的维度值相同，用于 RMS 归一化。
        epsilon (double): 公式中的输入 `eps`，添加到分母中的值，以确保数值稳定。

    返回:
        List[Tensor]: 包含三个张量的列表：
                      - 第一个张量 (`y1`) 是 `x1` 和 `x2` 相加后进行 RMS 归一化的结果，并转换成float。
                      - 第二个张量 (`y2`) 是 `x1` 和 `x2` 相加后进行 RMS 归一化的结果，数据类型和x1相同。
                      - 第三个张量 (`rstd`) 是计算过程中的均方根的倒数。
                      - 第四个张量 (`x`) 是 `x1` 和 `x2` 相加后的结果，数据类型和x1相同。
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
epsilon = 1e-5

x1 = torch.randn(input_shape, dtype=dtype)
x2 = torch.randn(input_shape, dtype=dtype)
gamma = torch.randn(gamma_shape, dtype=dtype)

# 使用 add_rms_norm 执行操作
y1, y2, rstd, x = kernel_gen_ops.add_rms_norm_cast(x1, x2, gamma, epsilon)
```

- **功能维度**
  * 数据格式支持：ND。
  * 张量形状维度不高于8维。