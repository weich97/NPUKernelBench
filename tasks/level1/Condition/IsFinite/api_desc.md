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

### 实现原理提示
利用 **IEEE 754** 浮点数标准中**非有限数（NaN / Inf）**的特定**二进制表示**。此方法完全基于高效的位运算和整数算术，以实现最优的硬件执行性能。
根据 IEEE 754 标准，非有限数（无穷大或NaN）具有一个共同特征：其**阶码（Exponent）的所有位均为1**。有限数的阶码则至少有一位为0。
因此，硬件判断逻辑遵循以下步骤：

1.  **屏蔽并提取关键位**：通过按位与（Bitwise AND）操作和一个预设的**掩码（MASK）**（其值等于对应浮点类型的无穷大二进制表示，如 `0x7C00` for FP16），来分离出符号位、阶码和部分尾数。此步骤统一了对正负数的处理。
2.  **算术比较**：将上一步提取出的二进制位作为整数，减去（`Sub`）该掩码值。
    * 对于**有限数**，其阶码小于全1模式，因此相减结果必为**负数**。
    * 对于**非有限数**（Inf/NaN），其阶码为全1模式，相减结果为**零或正数**。
3.  **结果布尔化**：通过一系列整数运算（如 `Maxs`, `Muls`），将负数（有限）转换为`1`（True），将零或正数（非有限）转换为`0`（False），最终生成布尔类型的判断结果。

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
        Tensor(bool): 布尔张量，形状与输入相同。
                      - 如果元素为有限数，返回 True；
                      - 否则返回 False。
    """

```
## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x_tensor = torch.tensor([1.0, float('inf'), -float('inf'), float('nan'), 0.0, 5.0], dtype=torch.float32)

# 使用 is_finite 执行操作
is_finite_output = kernel_gen_ops.is_finite(x_tensor)
```

## 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * `x` 和 `y`（输出）的形状必须一致。
  * `y`（输出）的数据类型为 **BOOL**。
  * `x` 的数据类型支持 **`torch.float16`**、**`torch.float`**、**`torch.bfloat16`**。