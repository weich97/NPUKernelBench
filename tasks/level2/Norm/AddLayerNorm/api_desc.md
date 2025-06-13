# aclnnAddLayerNorm

## 功能描述
### 算子功能
对两个输入张量执行逐元素相加后进行层归一化（LayerNorm）操作。

### 计算公式
  $$
  x = x1 + x2 + bias
  $$

  $$
  y = {{x-\bar{x}}\over\sqrt {Var(x)+eps}} * \gamma + \beta
  $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.add_layer_norm()` 函数形式提供：

```python
def add_layer_norm(x1, x2, bias, gamma, beta, epsilon, additional_out):
    """
    对两个输入张量执行逐元素相加后进行层归一化（LayerNorm）操作。

    参数:
        x1 (Tensor): 第一个输入Device侧张量。
        x2 (Tensor): 第二个输入Device侧张量，与 x1 具有相同的形状和数据类型。
        bias (Tensor): 偏置张量，shape可以和gamma/beta或是和x1/x2一致。
        gamma (Tensor): 缩放参数张量，与x1需要norm的维度的维度值相同，用于层归一化。
        beta (Tensor): 偏移参数张量，与x1需要norm的维度的维度值相同，用于层归一化。
        epsilon (double): 公式中的输入eps，添加到分母中的值，以确保数值稳定，仅支持取值1e-5。
        additional_out (bool): 表示是否开启x=x1+x2+bias的输出。

    返回:
        Tensor: 一个新的Device侧张量，表示 x1 和 x2 逐元素相加后进行层归一化的结果。
    """
```
## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量和参数
x1 = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float)
x2 = torch.tensor([[0.5, 1.5], [-1.0, 2.0]], dtype=torch.float)
bias = torch.tensor([0.1, -0.1], dtype=torch.float)
gamma = torch.tensor([1.0, 1.0], dtype=torch.float)
beta = torch.tensor([0.0, 0.0], dtype=torch.float)
epsilon = 1e-5
additional_out = True

# 使用 add_layer_norm 对两个输入张量执行逐元素相加后进行层归一化（LayerNorm）操作
y = kernel_gen_ops.add_layer_norm(x1, x2, bias, gamma, beta, epsilon, additional_out)
```


## 约束与限制
- **功能维度**
  * 数据格式支持：ND。
  * eps默认只支持1e-5。
  * 张量形状维度不高于8维，数据格式支持ND。
  * 输出张量 y 与输入张量 x1、x2 具有相同的形状、数据类型和数据格式。
  * gamma、beta 的形状必须与归一化维度一致。
