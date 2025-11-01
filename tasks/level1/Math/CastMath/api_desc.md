# Cast

## 功能描述

### 算子功能
将输入张量从源数据类型转换为目标数据类型。

### 计算公式
假设输入张量为 $x$，目标数据类型由 `dst_type` 标识，输出张量为 $y$，则：
$$
y = \text{Cast}(x, \text{dst_type})
$$
其中 $\text{Cast}$ 表示将 $x$ 转换为 `dst_type` 所对应的目标数据类型的操作。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.cast()` 函数形式提供：

```python
def cast(x, dst_type):
    """
    实现自定义 Cast 操作。
    
    参数:
        x (Tensor): Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、INT32、INT8、UINT8、BOOL、INT64、BFLOAT16、INT16，数据格式支持ND。
        dst_type (int): 目标数据类型，支持数据类型为INT64。
        
    返回:
        Tensor:Device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、INT32、INT8、UINT8、BOOL、INT64、BFLOAT16、INT16，数据格式支持ND。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float)

# 目标数据类型标识
dst_type = 2  # 假设对应 torch.int8

# 使用 cast 执行计算
result = kernel_gen_ops.cast(x, dst_type)
```
## 约束与限制

- 张量数据格式支持ND。


