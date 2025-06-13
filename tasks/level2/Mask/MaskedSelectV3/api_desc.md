# MaskSelectV3

## 功能描述

### 算子功能
根据一个布尔掩码张量（mask）中的值选择输入张量（input_tensor）中的元素作为输出，形成一个新的一维张量。

### 计算公式
从输入张量 `input_tensor` 中选择 `mask` 为 `True` 位置的元素组成新的一维张量。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.mask_select_v3()` 函数形式提供：

```python
def mask_select_v3(input_tensor, mask):
    """
    实现MaskSelectV3操作。
    
    参数:
        input_tensor (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        mask (Tensor): 布尔掩码张量，Device侧的张量，数据格式支持ND。
        
    返回:
        Tensor: 计算结果张量，为一维张量，数据类型与输入一致，数据格式支持ND。
    
    注意:
        张量数据格式支持ND
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
input_tensor = torch.randn(8, 2048, dtype=torch.float16)
mask = torch.randint(0, 2, (8, 2048), dtype=torch.bool)

# 使用mask_select_v3执行计算
result = kernel_gen_ops.mask_select_v3(input_tensor, mask)
```

## 约束与限制

无
    