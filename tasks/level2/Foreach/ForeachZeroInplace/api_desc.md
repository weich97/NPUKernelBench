# aclnnForeachZero

## 功能描述

### 算子功能
`aclnnForeachZero` 对输入张量列表中的每个张量 **原地** 置 0，覆盖原有数据。调用完成后，原张量列表中的每个元素值全部变成 0。

### 计算公式

输入张量列表  
$$
x = [x_0, x_1, \dots, x_{n-1}]
$$  

对每个张量 $x_i$ 执行：  
$$
x_i = 0,\quad i = 0,1,\dots,n-1
$$

## 接口定义

### Python 接口
通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_zero_()` 函数形式提供（末尾带下划线表示原地修改）：

```python
def foreach_zero_(tensor_list):
    """
    将张量列表中的每个张量原地置 0。

    参数:
        tensor_list (List[Tensor]): 输入 Device 侧张量列表，列表中所有张量必须具有相同的数据类型。

    返回:
        None: 原地修改，无返回值。

    注意:
        - 输入张量列表中的所有张量必须具有相同的数据类型。
        - 张量 shape 维度不高于 8 维，数据格式支持 ND。
        - 支持非连续的 Tensor。
        - 不支持空 Tensor。
        - 操作完成后，原张量中的数据被覆盖为 0。
    """

import torch
import kernel_gen_ops

# 创建具有相同数据类型的张量列表
tensor_list = [
    torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16, device='npu'),
    torch.tensor([[4, 5], [6, 7]], dtype=torch.float16, device='npu')
]

# 原地将所有张量置 0
kernel_gen_ops.foreach_zero_(tensor_list)

## 约束与限制

- **设备要求**  
  仅支持 Device 侧张量，不支持 Host 侧张量。

- **数据类型**  
  仅支持以下数据类型：  
  - FLOAT  
  - FLOAT16  
  - BFLOAT16  
  - INT16  
  - INT32  

- **数据格式**  
  仅支持 ND 格式。

- **维度限制**  
  张量 shape 维度不得高于 8 维。

- **连续性**  
  支持非连续 Tensor。

- **空 Tensor 限制**  
  不支持空 Tensor（即 tensor_list 不能为空）。
