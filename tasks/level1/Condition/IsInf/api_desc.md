# aclnnIsInf

## 功能描述
### 算子功能
判断输入张量中每个元素是否为正无穷大 (`+inf`) 或负无穷大 (`-inf`)。

### 计算公式
对于输入张量 $x$ 中的每个元素 $x_i$，输出张量 $y$ 的对应元素 $y_i$ 计算如下：
$$
y_i = \begin{cases}
1 & \text{if } x_i = +\infty \text{ or } x_i = -\infty \\
0 & \text{otherwise}
\end{cases}
$$
其中，`1` 通常表示 `True`，`0` 表示 `False`。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.is_inf()` 函数形式提供：

```python
def is_inf(x):
    """
    判断输入张量中每个元素是否为无穷大（正无穷或负无穷）。

    参数:
        x (Tensor): 输入Device侧张量。

    返回:
        Tensor: 一个布尔张量，形状与输入 `x` 相同，表示每个元素是否为无穷大。
    """
    import torch
    import kernel_gen_ops

    # 创建输入张量
    x_tensor = torch.tensor([1.0, float('inf'), -float('inf'), float('nan'), 0.0, 5.0], dtype=torch.float32)

    # 使用 is_inf 执行操作
    is_inf_output = kernel_gen_ops.is_inf(x_tensor)

    print("Input tensor:", x_tensor)
    print("IsInf output:", is_inf_output)
    # Expected output: tensor([False,  True,  True, False, False, False])

## 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * `x` 和 `y`（输出）的形状必须一致。
  * `y`（输出）的数据类型为 **BOOL**。
  * `x` 的数据类型支持 **FLOAT16**、**FLOAT**、**BFLOAT16**。