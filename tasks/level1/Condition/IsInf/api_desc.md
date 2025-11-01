# aclnnIsInf

## 功能描述
### 算子功能
判断输入张量中每个元素是否为正无穷大 (`+inf`) 或负无穷大 (`-inf`)。

### 计算公式
对于输入张量 $x$ 中的每个元素 $x_i$，输出张量 $y$ 的对应元素 $y_i$ 计算如下：
$$
y_i = \begin{cases}
True & \text{if } x_i = +\infty \text{ or } x_i = -\infty \\
False & \text{otherwise}
\end{cases}
$$

### 实现原理提示
利用 **IEEE 754** 浮点数标准中无穷大值的特定**二进制表示**。这种基于位运算的方法可以被硬件高效执行，性能更优。
根据 IEEE 754 标准，任何精度的浮点无穷大值都遵循一个通用特征：其**阶码（Exponent）的所有位都为1，同时尾数（Mantissa）的所有位都为0**。

因此，硬件判断逻辑通常遵循以下步骤：

1.  **提取关键位**：通过按位与（Bitwise AND）操作和一个**用于屏蔽符号位的掩码**（即该掩码的符号位为0，其余所有位均为1）来去除符号位的影响。这使得 `+inf` 和 `-inf` 可以被统一处理。
2.  **模式匹配**：将提取出的二进制位与当前数据类型的标准无穷大模式进行比较。
3.  **生成结果**：如果二者完全匹配，则判定该数值为无穷大，输出 `True`；否则输出 `False`。

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
        Tensor(bool): 布尔张量，形状与输入相同。
                      - 如果元素为无穷大，返回 True；
                      - 否则返回 False。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x_tensor = torch.tensor([1.0, float('inf'), -float('inf'), float('nan'), 0.0, 5.0], dtype=torch.float32)

# 使用 is_inf 执行操作
is_inf_output = kernel_gen_ops.is_inf(x_tensor)
```

## 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * `x` 和 `y`（输出）的形状必须一致。
  * `y`（输出）的数据类型为 **BOOL**。
  * `x` 的数据类型支持 **`torch.float16`**、**`torch.float`**、**`torch.bfloat16`**。