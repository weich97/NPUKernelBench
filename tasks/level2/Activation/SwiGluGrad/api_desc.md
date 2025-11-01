# aclnnSwiGLUGrad

## 功能描述

### 算子功能
`aclnnSwiGLUGrad` 是前向算子 `aclnnSwiGLU` 的反向操作。该算子接收前向输出的梯度 `yGrad` 和前向输入张量 `x`，计算输入 `x` 对应的梯度张量 `grad_input`。

### 计算公式

设前向输入张量为 $x$，沿指定 dim 维度均匀拆分为两个张量 $x_1$ 和 $x_2$，使得

$$
y = \mathrm{SiLU}(x_1) \odot x_2
$$

反向传播中，已知 $y$ 的梯度 $yGrad$，目标是计算输入 $x$ 的梯度：

- $\mathrm{grad}_{x_2} = yGrad \odot \mathrm{SiLU}(x_1)$
- $\mathrm{grad}_{x_1} = yGrad \odot x_2 \odot \mathrm{SiLU}'(x_1)$，其中 $\mathrm{SiLU}'(x) = \sigma(x) (1 + x (1 - \sigma(x)))$, $\sigma(x)$ 为 Sigmoid 函数

最终拼接 $\mathrm{grad}_{x_1}$ 和 $\mathrm{grad}_{x_2}$，生成输出张量 $grad\_input$。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.swi_glu_grad()` 函数形式提供：

```python
def swi_glu_grad(y_grad: Tensor, x: Tensor, dim: int = -1) -> Tensor:
    """
    计算 SwiGLU 前向操作的输入张量 x 的梯度。

    参数:
        y_grad (Tensor): 前向输出对应的梯度张量，形状为 [..., d]，数据类型为 torch.float16、torch.bfloat16 或 torch.float。
        x (Tensor): 前向输入张量，在 dim 维度的长度必须为偶数, 数据类型与 y_grad 相同。
        dim (int, 可选): 切分维度，取值范围 [-x.dim(), x.dim()-1]，默认 -1。

    返回:
        Tensor: 输入张量 x 对应的梯度张量, 数据类型与 x 相同。

    注意:
        - x 在指定维度dim上必须可以均匀拆分为两部分
        - y_grad 的形状必须与前向输出一致
        - 输出张量的形状为 x 的形状
        - 支持非连续 Tensor
        - 支持的最大维度为 8 维
        - 输出张量与输入张量具有相同的数据类型和数据格式
    """

```

## 使用案例

```python
import torch
import kernel_gen_ops

# 构造前向输入张量 x，最后一维长度为偶数
x = torch.randn(2, 4, dtype=torch.float)

# 构造与前向输出形状匹配的梯度张量 y_grad
y_grad = torch.randn(2, 2, dtype=torch.float)

# 执行 SwiGLUGrad 操作
grad_input = kernel_gen_ops.swi_glu_grad(y_grad, x, dim = -1)
```

## 约束与限制

无