# aclnnSwish

## 功能描述

### 算子功能
该算子实现Swish激活函数计算功能。Swish激活函数定义为输入 $x$ 与 $Sigmoid(s*x)$ 函数值的乘积。该函数结合线性与非线性特性，实现更平滑的梯度传播，其平滑曲线允许部分负值输入保留，能够在深层网络中优化训练过程并提高模型性能。通过调整参数 $s$ 取值，Swish可以在线性函数和ReLU函数之间平滑过渡：当 $s\to\infty$时，Swish趋近于ReLU；当 $s=0$时，Swish成为简单的缩放线性函数。Swish 函数具有非单调特性，其梯度在不同区间表现不同，将其应用在训练深度神经网络中能够有效避免梯度消失问题，并且在优化过程中有良好的自适应性，有助于模型更好地拟合数据，在许多深度学习任务中发挥重要作用。

### 计算公式

$$
y=x\cdot\mathrm{sigmoid}\left(s\cdot x\right)=x\cdot\frac{1}{1+e^{-s\cdot x}}
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.swish()` 函数形式提供：


```python
def swish(x, s=1.0):
    """
    实现 Swish 激活函数操作。

    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        s (float, 可选): 控制 Swish 激活函数形状的缩放参数，默认为1.0。
                         - 当 s → ∞ 时，Swish 趋近于 ReLU；
                         - 当 s = 0 时，Swish 为线性缩放函数。

    返回:
        Tensor: 输出张量，为输入张量经过 Swish 激活函数处理后的结果。
                数据类型与输入一致，数据格式保持不变。

    注意:
        - Swish 激活函数定义为:  `Swish(x) = x * sigmoid(s * x)`；
        - 该函数结合线性与非线性特性，有助于缓解梯度消失，提高模型表达能力；
        - 支持任意形状输入张量，逐元素应用 Swish 运算；
        - 推荐在深层神经网络中使用以提升性能。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.randn(4, 8, 2048, dtype=torch.float32)  # 任意高维张量

# 使用 swish 执行激活操作，默认 s=1.0
result = kernel_gen_ops.swish(x)

# 使用 swish 并自定义参数 s=0.5
result_custom = kernel_gen_ops.swish(x, s=0.5)
```
## 约束与限制

- 张量数据格式支持ND。


