# aclnnForeachErf

## 功能描述
---
### 算子功能
`aclnnForeachErf` 操作对输入张量列表中的每个张量依次执行**误差函数 (Error Function) 变换**，并返回一个新的张量列表，其中每个元素是对应输入张量的误差函数变换结果。此操作保留张量的形状和维度信息，仅改变张量中的数值。

### 计算公式
---

对于输入张量列表 $X = [X_0, X_1, ..., X_{n-1}]$，`aclnnForeachErf` 操作生成输出张量列表 $Y = [Y_0, Y_1, ..., Y_{n-1}]$。

对于每个张量 $X_i$，输出 $Y_i$ 是通过应用误差函数计算得到的：

$$y_i = \text{erf}({x_i}) = \frac{2}{\sqrt{\pi}} \int_{0}^{{x_i}} e^{-t^2} dt \quad (i=0,1,...n-1)$$

## 接口定义
---
### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_erf()` 函数形式提供：

```python
def foreach_erf(tensor_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    对张量列表中的每个张量执行误差函数操作。
    
    参数:
        tensor_list (List[torch.Tensor]): 输入Device侧张量列表，列表中所有张量必须具有相同的数据类型。
        
    返回:
        List[torch.Tensor]: 一个新的Device侧张量列表，其中每个张量是对应输入张量的误差函数结果。
    
    注意:
        - 输入张量列表中的所有张量必须具有相同的数据类型。
        - 张量 shape 维度不高于8维，数据格式支持 ND。
        - 支持非连续的 Tensor。
        - 输出张量与输入张量具有相同的形状、数据类型和数据格式。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建具有相同数据类型的张量列表
tensor_list = [
    torch.tensor([0.0, 0.5, 1.0], dtype=torch.float),
    torch.tensor([[-1.0, -0.5], [0.5, 1.5]], dtype=torch.float)
]

# 使用 foreach_erf 对张量列表中的每个张量应用 erf 操作
result_list = kernel_gen_ops.foreach_erf(tensor_list)

# 打印结果进行验证
for i, original_tensor in enumerate(tensor_list):
    erf_tensor = result_list[i]
```

## 约束与限制
---
- **参数说明**:
  - `x` (aclTensorList*, 计算输入)：公式中的`x`，Device侧的aclTensorList，表示进行误差函数运算的输入张量列表。**数据类型**支持FLOAT、FLOAT16、BFLOAT16。**shape维度**不高于8维，**数据格式**支持ND。支持**非连续的Tensor**，**不支持空Tensor**。shape与出参`out`的shape一致。
  - `out` (aclTensorList*，计算输出)：公式中的`y`，Device侧的aclTensorList，表示进行误差函数运算的输出张量列表。**数据类型**支持FLOAT、FLOAT16、BFLOAT16。**shape维度**不高于8维，**数据格式**支持ND。支持**非连续的Tensor**，**不支持空Tensor**。**数据类型、数据格式和shape**与入参`x`的数据类型、数据格式和shape一致。
