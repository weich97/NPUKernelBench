# aclnnPreLayerNorm

## 功能描述

### 算子功能
`PreLayerNorm`是`Add`和`LayerNorm`的融合算子，`Add`算子的输出作为`LayerNorm`算子的输入，进行归一化处理

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.pre_layer_norm()` 函数形式提供：


```python
def pre_layer_norm(
    x: torch.Tensor,
    y: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    """
    实现 pre_layer_norm 操作，对 x + y 的结果进行 LayerNorm 归一化。

    参数:
        x: 输入张量；
        y: 输入张量，形状与 x 相同；
        gamma: 缩放因子张量，形状为 [D] 或可广播为最后一维；
        beta: 偏移因子张量，形状为 [D] 或可广播为最后一维；
        epsilon: 防止除零的小常数，通常为标量张量或 float。

    返回:
        Tensor: 输出张量，形状与输入相同，表示加法后经过 LayerNorm 的结果。
    """

```

## 使用案例

```python
import torch
import kernel_gen_ops  # 假设已定义 pre_layer_norm 算子

# 构造输入
shape = [32, 4, 1024]
x = torch.randn(shape, dtype=torch.float32)
y = torch.randn(shape, dtype=torch.float32)
gamma = torch.randn(shape[-1], dtype=torch.float32)
beta = torch.randn(1024[-1], dtype=torch.float32)
epsilon = torch.empty(1).uniform_(1e-7, 1e-5).item()

# 执行 Pre-LayerNorm 运算
output = kernel_gen_ops.pre_layer_norm(x, y, gamma, beta, epsilon)

```
## 约束与限制

- 张量数据格式支持ND。


