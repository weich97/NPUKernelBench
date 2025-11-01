# aclnnUnalignAdd

## 功能描述

### 算子功能
`Unalign_add`实现了两个数据相加，返回相加结果的功能(非对齐)。

### 计算公式

$$
z = x + y
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.unalign_add()` 函数形式提供：

```python
def unalign_add(x, y):
    """
    实现自定义非对齐加法操作（unalign_add）。

    参数:
        x (Tensor): 输入张量1，Device侧张量，数据格式支持 ND。
        y (Tensor): 输入张量2，Device侧张量，数据格式支持 ND。

    返回:
        Tensor: 输出张量，为 x 与 y 的非2整数次幂对齐。

    注意:
        - 输入张量必须至少为1维；
        - 通常用于处理输入数据非2整数次幂对齐的情况；
        - 输出张量 shape 与广播后的 shape 一致；
        - 支持 float16数据类型。
    """

```

## 使用案例

```python
import torch
import kernel_gen_ops


# 创建两个张量
x = torch.randn(8, 2048, dtype=torch.float32)
y = torch.randn(8, 2048, dtype=torch.float32)    

# 执行非对齐加法操作
result = kernel_gen_ops.unalign_add(x, y)
```
## 约束与限制

- 张量数据格式支持ND。


