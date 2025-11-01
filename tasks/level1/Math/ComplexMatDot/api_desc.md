# ComplexMatDot

## 功能描述

### 算子功能
返回一个和输入复数矩阵同样形状大小的新复数矩阵，它的每一个元素是输入的两个复数矩阵对应位置元素的逐点乘结果。

### 计算公式
$$
{out_{ij}} = {matx_{ij}} * {maty_{ij}}
$$
其中，下标 $i$、$j$ 表示第 $i$ 行、第 $j$ 列。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.complex_mat_dot()` 函数形式提供：

```python
def complex_mat_dot(matx, maty, m, n):
    """
    实现自定义 ComplexMatDot 操作。
    
    参数:
        matx (Tensor): 公式中的matx，Device侧的aclTensor，Atlas A2 训练系列产品/Atlas 800I A2推理产品数据类型支持COMPLEX64，数据格式支持ND。不支持非连续的Tensor，不支持空Tensor。
        maty (Tensor): 公式中的maty，Device侧的aclTensor，Atlas A2 训练系列产品/Atlas 800I A2推理产品数据类型支持COMPLEX64，数据格式支持ND。不支持非连续的Tensor，不支持空Tensor。
        m (int): 表示矩阵行数，数据类型支持INT64。
        n (int): 表示矩阵列数，数据类型支持INT64。
        
    返回:
        Tensor: 表示计算结果，公式中的out，Device侧的aclTensor，Atlas A2 训练系列产品/Atlas 800I A2推理产品数据类型支持COMPLEX64，数据格式支持ND。不支持非连续的Tensor，不支持空Tensor。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建第一个输入复数矩阵
matx = torch.randn(5, 5, dtype=torch.complex64)

# 创建第二个输入复数矩阵
maty = torch.randn(5, 5, dtype=torch.complex64)

# 矩阵行数
m = 5

# 矩阵列数
n = 5

# 使用 complex_mat_dot 执行计算
result = kernel_gen_ops.complex_mat_dot(matx, maty, m, n)
```
## 约束与限制

- 张量数据格式支持ND。


