# aclnnSasum

## 功能描述
### 算子功能
计算给定实数向量中所有元素的**绝对值之和**。这是一个类似于 BLAS `SASUM` 的操作。

### 计算公式
对于输入张量 $x$，考虑从第一个元素开始，每隔 `incx` 个元素取一个，共取 `n` 个元素。
设这些被考虑的元素为 $x'_0, x'_1, \ldots, x'_{n-1}$。
算子返回这些元素绝对值的和：
$$
\text{out} = \sum_{i=0}^{n-1} |x'_i|
$$
其中 $x'_i$ 是从原始张量 $x$ 中按 `incx` 步长提取的第 $i$ 个元素。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.sasum()` 函数形式提供：

```python
def sasum(x, n, incx):
    """
    计算给定实数向量中所有元素的绝对值之和。

    参数:
        x (Tensor): 输入Device侧张量。
        n (int): 要考虑的元素数量。
        incx (int): 访问 `x` 中元素时的步长（增量）。

    返回:
        Tensor: 一个标量张量，包含计算出的绝对值之和，数据类型与输入 `x` 相同。
    """
```
## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x_tensor = torch.tensor([1.0, -2.0, 3.0, -4.0, 5.0], dtype=torch.float32)
n_val = 5 # Consider all 5 elements
incx_val = 1 # Step of 1

# 使用 sasum 执行操作
sum_abs_value = kernel_gen_ops.sasum(x_tensor, n_val, incx_val)

```

### 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * `x` 张量shape维度不高于3维。
  * `x` 的数据类型支持 **FLOAT**。
  * `n` 必须是 **INT64** 类型，表示要考虑的元素数量，且 `n` 必须是非负数。
  * `incx` 必须是 **INT64** 类型，表示访问 `x` 中元素时的步长，且 `incx` 必须大于 0。当前支持数值为1。
  * `out`（输出）张量是一个**标量**，数据类型与 `x` 相同。
  * `x` 的实际长度必须足以容纳 `n` 个元素按 `incx` 步长访问（即 `(n - 1) * incx + 1` 必须小于等于 `x` 的总长度）。