# aclnnIsFinite

## 功能描述
### 算子功能
判断输入张量中每个元素是否为有限数（即不是无穷大也不是非数值NaN）。

### 计算公式
对于输入张量 $x$ 中的每个元素 $x_i$，输出张量 $y$ 的对应元素 $y_i$ 计算如下：
$$
y_i = \begin{cases}
1 & \text{if } x_i \text{ is finite} \\
0 & \text{otherwise}
\end{cases}
$$
其中，有限数是指不为 $+\infty$、$-\infty$ 或 NaN 的任何浮点数。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.is_finite()` 函数形式提供：

```python
def is_finite(x):
    """
    判断输入张量中每个元素是否为有限数。

    参数:
        x (Tensor): 输入Device侧张量。

    返回:
        Tensor: 一个数值张量（0.0 或 1.0），形状与输入 `x` 相同，表示每个元素是否为有限数。
    """
    import torch
    import kernel_gen_ops

    # 创建输入张量
    x_tensor = torch.tensor([1.0, float('inf'), -float('inf'), float('nan'), 0.0, 5.0], dtype=torch.float32)

    # 使用 is_finite 执行操作
    is_finite_output = kernel_gen_ops.is_finite(x_tensor)

    print("Input tensor:", x_tensor)
    print("IsFinite output:", is_finite_output)
    # Expected output: tensor([1., 0., 0., 0., 1., 1.]) if dtype is float, or [ True, False, False, False,  True,  True] if bool

