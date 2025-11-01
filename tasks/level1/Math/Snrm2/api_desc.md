# aclnnSnrm2

## 功能描述
### 算子功能
`aclnnSnrm2` 计算实数向量的欧式范数。

### 计算公式
对于输入张量 $x$，考虑从第一个元素开始，每隔 `incx` 个元素取一个，共取 `n` 个元素。
设这些被考虑的元素为 $x'_0, x'_1, \ldots, x'_{n-1}$。
算子返回这些元素的欧式范数：
$$
 out = \sqrt{\sum_{i=0}^{n-1} |x'_i|^2}
$$
其中 $x'_i$ 是从原始张量 $x$ 中按 `incx` 步长提取的第 $i$ 个元素。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.snrm2()` 函数形式提供：

```python
def snrm2(x,  n, incx):
    """
    计算实数向量的欧式范数。
    
    参数:
        x (Tensor): 输入Device侧张量。
        n (int): 要考虑的元素数量。
        incx (int): 访问 `x` 中元素时的步长（增量）。

    返回:
        Tensor: 一个标量张量，包含计算出的欧式范数，数据类型与输入 `x` 相同。
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
sum_abs_value = kernel_gen_ops.snrm2(x_tensor, n_val, incx_val)

```


## 约束与限制

- 张量数据格式支持ND。
- incx目前支持数值为1，数据类型支持INT64。
- x支持FLOAT32。


