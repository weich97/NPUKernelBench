# aclnnForeachRoundOffNumber

## 功能描述
### 算子功能
对输入张量列表中的每个张量执行四舍五入操作，并根据指定的舍入模式 `roundMode` 对数值进行取整。返回一个与输入张量列表形状相同的新张量列表。

### 计算公式
$$
x = [{x_0}, {x_1}, \ldots, {x_{n-1}}] \\
y = [{y_0}, {y_1}, \ldots, {y_{n-1}}] \\
$$
对于列表中的每个张量 $x_i$，其对应的输出张量 $y_i$ 计算为：
$$
y_i = \text{round}(x_i, \text{roundMode}) \quad (i=0,1,\ldots,n-1)
$$
其中 `roundMode` 的取值及对应的舍入策略如下：
* **`roundMode = 1`**: 对输入进行**四舍六入五成双**舍入操作（即向最接近的偶数舍入，例如 `2.5 -> 2.0`, `3.5 -> 4.0`）。
* **`roundMode = 2`**: 对输入进行**向负无穷舍入**取整操作（即 `floor`）。
* **`roundMode = 3`**: 对输入进行**向正无穷舍入**取整操作（即 `ceil`）。
* **`roundMode = 4`**: 对输入进行**四舍五入舍入**操作（即 `round half up`，向远离零的方向舍入，例如 `2.5 -> 3.0`, `-2.5 -> -3.0`）。
* **`roundMode = 5`**: 对输入进行**向零舍入**操作（即 `truncation`）。
* **`roundMode = 6`**: 对输入进行**最近邻奇数舍入**操作。
* **`roundMode` 为其他值时**: 如果存在精度损失，进行四舍六入五成双舍入操作；不涉及精度损失时则不进行舍入操作。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_round_off_number()` 函数形式提供：

```python
def foreach_round_off_number(x, round_mode):
    """
    对输入张量列表中的每个张量执行四舍五入到指定的 round_mode 小数位数运算。

    参数:
        x (List[Tensor]): 进行舍入运算的输入张量列表。
                          Device侧的 TensorList，数据类型支持 FLOAT、FLOAT16、BFLOAT16。
        round_mode (int): 舍入模式的整数标量。
                          Host侧的 int8 标量，取值 1-6 对应不同舍入策略，其他值有默认行为。

    返回:
        List[Tensor]: 一个与输入张量列表形状和数据类型相同的输出张量列表。
    """

```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量列表
x_list_float32 = [
    torch.tensor([2.5, 3.5, -2.5, -3.5, 2.0, 3.0, 2.1, -2.1], dtype=torch.float32),
    torch.tensor([0.7, -0.3, 10.5, -10.5], dtype=torch.float32)
]
x_list_float16 = [
    torch.tensor([1.4, 1.6, 1.5, -1.5], dtype=torch.float16)
]

# 示例 1: round_mode = 1 (四舍六入五成双)
out1 = kernel_gen_ops.foreach_round_off_number(x_list_float32, 1)

# 示例 2: round_mode = 2 (向负无穷舍入 - floor)
out2 = kernel_gen_ops.foreach_round_off_number(x_list_float32, 2)

# 示例 3: round_mode = 3 (向正无穷舍入 - ceil)
out3 = kernel_gen_ops.foreach_round_off_number(x_list_float32, 3)

# 示例 4: round_mode = 4 (四舍五入 - round half up)
out4 = kernel_gen_ops.foreach_round_off_number(x_list_float32, 4)

# 示例 5: round_mode = 5 (向零舍入 - truncate)
out5 = kernel_gen_ops.foreach_round_off_number(x_list_float32, 5)

# 示例 6: round_mode = 6 (最近邻奇数舍入) - behavior might need specific NPU-side validation
out6 = kernel_gen_ops.foreach_round_off_number(x_list_float32, 6)

```
## 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * `x` 和 `out` 均表示张量列表，它们的**长度、每个张量的形状、数据类型和数据格式必须一致**。
  * `x` 中的张量数据类型支持 **FLOAT**、**FLOAT16**、**BFLOAT16**。
  * `x` 和 `out` 中的张量形状维度不高于**8维**。
  * `x` 和 `out` 支持**非连续**的 `Tensor`，**不支持空** `Tensor`。
  * `roundMode` 是一个**标量**，数据类型仅支持 **INT8**。