# aclnnFill

## 功能描述

### 算子功能
创建一个形状由输入 `dims` 指定的张量，并用标量值 `value` 填充所有元素。该算子常用于初始化张量为特定值。

### 计算公式
$$
y = \text{fill}(dims, value)
$$
其中 $y$ 是输出张量，$dims$ 是指定的形状，$value$ 是用于填充的标量值。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.fill()` 函数形式提供：

```python
def fill(dims, value):
    """
    实现自定义 Fill 操作。
    
    参数:
        dims (List[int]): Device侧的aclIntArray，公式中的输入dims，数据类型支持 INT64。
        value (Tensor): Device侧的aclScalar，公式中的输入value，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT8、INT32、INT64、BOOL。
        
    返回:
        Tensor: Device侧的aclTensor，公式中的输出y，数据类型支持FLOAT16、BFLOAT16、FLOAT32、INT8、INT32、INT64、BOOL，数据格式支持ND。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 指定输出张量的形状
dims = [3, 3]

# 创建用于填充的标量值
value = torch.tensor([2.0], dtype=torch.float)

# 使用 fill 执行计算
result = kernel_gen_ops.fill(dims, value)
```
## 约束与限制

- 张量数据格式支持ND。


