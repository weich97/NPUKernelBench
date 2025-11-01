# aclnnIcamin

## 功能描述
### 算子功能
计算给定**复数张量（向量）**中**实部和虚部绝对值之和最小**的元素的**一维索引**。这是一个类似于 BLAS `ICAMINs` 的操作。

### 计算公式
对于输入复数张量 $x$，考虑从第一个元素开始，每隔 `incx` 个元素取一个，共取 `n` 个元素。
设这些被考虑的元素为 $x'_0, x'_1, \ldots, x'_{n-1}$。
算子返回这些元素中 $|Re(x'_i)| + |Im(x'_i)|$ 最小的那一个的**1-based 索引**。
$$
\text{index} = \underset{0 \le i < n}{\operatorname{argmin}} (|Re(x'_{i})| + |Im(x'_{i})|) + 1
$$
其中 $Re(x'_i)$ 和 $Im(x'_i)$ 分别是复数 $x'_i$ 的实部和虚部。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.icamin()` 函数形式提供：

```python
def icamin(x, n, incx):
    """
    计算给定复数张量（向量）中实部和虚部绝对值之和最小的元素的一维索引。

    参数:
        x (Tensor): 输入Device侧的复数张量（通常为一维向量）。
        n (int): 要考虑的元素数量。
        incx (int): 访问 `x` 中元素时的步长（增量）。

    返回:
        Tensor: 一个标量张量，包含绝对值最小的元素的1-based索引，数据类型为 int32。
    """
```
## 使用案例

```python
import torch
import kernel_gen_ops
import numpy as np # For complex number generation

# 创建输入张量
x_np = np.array([1.0 + 2.0j, -5.0 + 1.0j, 3.0 + 3.0j, 8.0 - 0.5j, -2.0 + 0.1j, 10.0 + 0.0j], dtype=np.complex64)
x_tensor = torch.from_numpy(x_np)

n_val = 6 # Consider all 6 elements
incx_val = 1 # Step of 1

# 使用 icamin 执行操作
result_idx = kernel_gen_ops.icamin(x_tensor, n_val, incx_val)
```

## 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * `x` 张量通常为**一维向量**。
  * `x` 的数据类型支持 **COMPLEX64**、**COMPLEX128**。
  * `n` 必须是 **INT64** 类型，表示要考虑的元素数量，且 `n` 必须是非负数。
  * `incx` 必须是 **INT64** 类型，表示访问 `x` 中元素时的步长，且 `incx` 必须大于 0。目前支持数值为1。
  * `out`（输出）张量是一个**标量**，数据类型为 **INT32**。
  * `x` 的实际长度必须足以容纳 `n` 个元素按 `incx` 步长访问（即 `(n - 1) * incx + 1` 必须小于等于 `x` 的总长度）。