# aclnnIsamax

## 功能描述
### 算子功能
计算给定张量（向量）中绝对值最大的元素的**一维索引**。这是一个类似于 BLAS `ISAMAX` 的操作。

### 计算公式
对于输入张量 $x$，考虑从第一个元素开始，每隔 `incx` 个元素取一个，共取 `n` 个元素。
设这些被考虑的元素为 $x'_0, x'_1, \ldots, x'_{n-1}$。
算子返回这些元素中绝对值最大的那一个的**1-based 索引**。
$$
\text{index} = \underset{0 \le i < n}{\operatorname{argmax}} (|x'_{i}|) + 1
$$
其中 $x'_i$ 是从原始张量 $x$ 中按 `incx` 步长提取的第 $i$ 个元素。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.isamax()` 函数形式提供：

```python
def isamax(x, n, incx):
    """
    计算给定张量（向量）中绝对值最大的元素的一维索引。

    参数:
        x (Tensor): 输入Device侧张量（通常为一维向量）。
        n (int): 要考虑的元素数量。
        incx (int): 访问 `x` 中元素时的步长（增量）。

    返回:
        Tensor: 一个标量张量，包含绝对值最大元素的1-based索引，数据类型为 int32。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x_tensor = torch.tensor([1.0, -5.0, 3.0, 8.0, -2.0, 10.0], dtype=torch.float32)
n_val = 6 # Consider all 6 elements
incx_val = 1 # Step of 1

# 使用 isamax 执行操作
result_idx = kernel_gen_ops.isamax(x_tensor, n_val, incx_val)
```

### 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * `x` 张量通常为**一维向量**。
  * `x` 的数据类型支持 **FLOAT**。
  * `n` 必须是 **INT64** 类型，表示要考虑的元素数量，且 `n` 必须是非负数。
  * `incx` 必须是 **INT64** 类型，表示访问 `x` 中元素时的步长，且 `incx` 必须大于 0。目前支持数值为1。
  * `out`（输出）张量是一个**标量**，数据类型为 **INT32**。
  * `x` 的实际长度必须足以容纳 `n` 个元素按 `incx` 步长访问（即 `(n - 1) * incx + 1` 必须小于等于 `x` 的总长度）。