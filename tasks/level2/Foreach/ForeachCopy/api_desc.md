# aclnnForeachCopy

## 功能描述
---
### 算子功能
`aclnnForeachCopy` 操作对输入张量列表中的每个张量执行**逐元素拷贝操作**，并返回一个新的张量列表，其中每个元素是对应输入张量的精确副本。此操作保留张量的形状、数据类型和维度信息。

### 计算公式
---

对于输入张量列表 $X = [X_0, X_1, ..., X_{n-1}]$，`aclnnForeachCopy` 操作生成输出张量列表 $Y = [Y_0, Y_1, ..., Y_{n-1}]$。

对于每个输入张量 $X_i$，对应的输出 $Y_i$ 是通过简单的逐元素拷贝得到的：

$$Y_i = X_i$$

具体来说，对于 $X_i$ 中的每个元素 $x$，对应的输出元素 $y$ 拷贝为：

$$y = x$$

## 接口定义
---
### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_copy()` 函数形式提供：

```python
def foreach_copy(tensor_list: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    对张量列表中的每个张量执行逐元素拷贝操作。
    
    参数:
        tensor_list (List[torch.Tensor]): 输入张量列表，列表中所有张量必须具有相同的数据类型。
        
    返回:
        List[torch.Tensor]: 一个新的张量列表，其中每个张量是对应输入张量的副本。
    
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
    torch.tensor([0.0, 1.0, 2.0], dtype=torch.float),
    torch.tensor([[-1.0, -2.0], [3.0, 4.0]], dtype=torch.float),
    torch.randn(3, 4, 5, dtype=torch.float16) # Example with float16 and more dims
]

# 使用 foreach_copy 对张量列表中的每个张量应用拷贝操作
result_list = kernel_gen_ops.foreach_copy(tensor_list)

# 验证拷贝是否成功
for i, original_tensor in enumerate(tensor_list):
    copied_tensor = result_list[i]
```

## 约束与限制
---
- **功能维度**
  * **数据格式**：支持 ND 格式。
  * **数据类型**：输入 `tensor_list` 中的所有张量必须具有**相同的数据类型**（例如 float16、float32、bfloat16、int8、unit8、int6、unit16、int32、unit32、int64、double、bool）。
  * **张量维度**：张量形状不得超过 8 维。
  * **数据连续性**：支持非连续张量。
  * **输出特性**：输出张量与其对应的输入张量具有相同的形状、数据类型和数据格式。
  * **列表长度**：输入 `tensor_list` 可包含一个或多个张量。