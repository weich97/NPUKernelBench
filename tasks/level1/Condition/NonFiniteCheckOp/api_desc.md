# aclnnNonFiniteCheckOp

## 功能描述
### 算子功能
检查输入张量中是否存在任何非有限（non-finite）的元素。非有限数包括正无穷大 (`+inf`)、负无穷大 (`-inf`) 和非数值 (`NaN`)。

### 计算公式
算子遍历输入张量 $x$ 中的所有元素 $x_i$。
如果张量中存在任何一个元素 $x_k$ 使得 $x_k$ 不满足有限数的条件（即 $x_k = \pm \infty$ 或 $x_k = \text{NaN}$），则输出结果为 `1.0`。
否则，如果所有元素都是有限数，则输出结果为 `0.0`。
$$
\text{out} = \begin{cases}
1.0 & \text{if any } x_i \text{ is } \pm\infty \text{ or } \text{NaN} \\
0.0 & \text{otherwise}
\end{cases}
$$
输出 `out` 是一个标量张量。

### 实现原理
采用一种**两阶段高效检测策略**，结合了硬件并行归约与 **IEEE 754** 浮点数标准的位操作，以判断张量中是否存在非有限数（无穷大 `Inf` 或非数值 `NaN`）。
根据 IEEE 754 标准，所有非有限数（`Inf` 和 `NaN`）共享一个通用特征：其**阶码（Exponent）的所有位都为1**。该算法利用此特性进行高效判断。
硬件判断逻辑遵循以下步骤：
1.  **高效候选筛选**：不直接遍历所有元素。首先，利用硬件归约指令（`ReduceMax`/`ReduceMin`）对一大块数据进行并行计算，瞬间找出其中的最大值和最小值。因为任何非有限数（`+inf`, `-inf` 或 `NaN`）都必然会体现在这两个极值中，从而将检查范围从上万个元素缩小到仅两个。

2.  **提取阶码位**：对筛选出的极值进行位操作。
    * 通过**按位与（Bitwise AND）**操作和一个掩码（如 `0x7FFFFFFF`）来屏蔽符号位，统一处理正负数。
    * 通过**按位右移（Bitwise Right Shift）**操作，精准地分离出阶码部分。

3.  **模式匹配与输出**：将提取出的阶码位与一个所有位全为1的模式（`float`为 `0xFF`，`half`为 `0x1F`）进行比较。如果完全匹配，则判定该值为非有限数，立即将最终结果置为 `True` 并可提前终止后续检查；否则判定为有限数。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.non_finite_check_op()` 函数形式提供：

```python
def non_finite_check_op(x):
    """
    检查输入张量中是否存在任何非有限（+/-Inf 或 NaN）的元素。

    参数:
        x (Tensor): 输入Device侧张量。

    返回:
        Tensor: 一个标量张量（0.0 或 1.0），表示输入张量是否包含非有限数。
    """

```
## 使用案例

```python
import torch
import kernel_gen_ops

# 创建包含非有限值的输入张量
x_tensor_non_finite = torch.tensor([1.0, float('inf'), 0.0, float('nan')], dtype=torch.float32)
# 创建只包含有限值的输入张量
x_tensor_finite = torch.tensor([1.0, -2.0, 3.0, 0.0], dtype=torch.float32)

# 使用 non_finite_check_op 执行操作
output_non_finite = kernel_gen_ops.non_finite_check_op(x_tensor_non_finite)
output_finite = kernel_gen_ops.non_finite_check_op(x_tensor_finite)

```

## 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * `x` 和 `y`（输出）的形状必须一致。
  * `y`（输出）张量是一个**标量**，数据类型为 **`torch.float`**。
  * `x` 的数据类型支持 **`torch.float16`**、**`torch.float`**、**`torch.bfloat16`**。