# aclnnTril

## 功能描述

### 算子功能
`aclnnTril` 实现算子功能：用于提取张量的下三角部分。返回一个张量`out`，包含输入矩阵(2D张量)的下三角部分，`out`其余部分被设为0。这里所说的下三角部分为矩阵指定对角线`diagonal`之上的元素。参数`diagonal`控制对角线：默认值是`0`，表示主对角线。如果 `diagonal > 0`，表示主对角线之上的对角线；如果 `diagonal < 0`，表示主对角线之下的对角线。。

### 计算公式

$$
y = tril(x, diagonal)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.tril()` 函数形式提供：


```python
def tril(x, diagonal=0):
    """
    实现自定义下三角操作。
    
    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND，维度必须大于等于2。
        diagonal (int, 可选): 对角线偏移，默认为0。
                    - 0: 表示主对角线；
                    - >0: 表示在主对角线上方的第k条对角线；
                    - <0: 表示在主对角线下方的第k条对角线。
    
    返回:
        Tensor: 输出张量，为输入张量的下三角形式。数据类型与输入一致，数据格式支持ND。
    
    注意:
        - 输入张量必须是至少2维；
        - 支持ND格式张量，在最后两个维度上执行下三角操作；
        - 上三角区域的元素将被置为0。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.rand(4, 8, 2048, dtype=torch.float32)  # 高维ND张量

# 使用tril执行计算
result = kernel_gen_ops.tril(x, diagonal=0)
```
## 约束与限制

- 张量数据格式支持ND。


