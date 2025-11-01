# aclnnForeachPowScalar

## 支持的数据类型  
- `x` 张量列表元素  
  - `torch.float16`  
  - `torch.float32`  
  - `torch.bfloat16`  
  - `torch.int32`  

- `scalar`（指数）  
  - 当 `x` 为 `float16 / float32 / int32` 时：与 `x` 同 dtype  
  - 当 `x` 为 `bfloat16` 时：仅支持 `float32`

## 功能描述

### 算子功能
`aclnnForeachPowScalar` 对输入张量列表中的每个张量逐元素做幂运算，返回一个新的张量列表，其中每个张量等于对应输入张量的 `scalar` 次方。

### 计算公式

输入张量列表  
$$
x = [x_0, x_1, \dots, x_{n-1}]
$$

输出张量列表  
$$
y = [y_0, y_1, \dots, y_{n-1}]
$$

对每个张量  
$$
y_i = x_i^{\text{scalar}}, \quad i = 0, 1, \dots, n-1
$$

- `x`：Device 侧 `aclTensorList`，底数  
- `scalar`：Host 侧标量 `aclTensor`，指数  
- `y`：Device 侧 `aclTensorList`，结果  

## 接口定义

### Python 接口
```python
def foreach_pow_scalar(x: List[torch.Tensor], scalar: torch.Tensor) -> List[torch.Tensor]:
    """
    对输入张量列表中的每个张量执行标量幂运算。

    参数:
        x (List[Tensor]): 输入 Device 侧张量列表，所有张量须同 dtype。
                          支持 float16、float32、bfloat16、int32；支持 ND；维度 ≤ 8；支持非连续。
        scalar (Tensor):  Host 侧 0-D 张量（标量），指数。
                          - 若 x 为 float16/float32/int32 → 与 x 同 dtype  
                          - 若 x 为 bfloat16 → 仅支持 float32

    返回:
        List[Tensor]: 新张量列表，形状及 dtype 与输入列表对应张量一致，数据格式支持 ND，维度 ≤ 8，支持非连续。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 示例 1：float32 底数，float32 指数
x_list = [
    torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32, device='npu'),
    torch.tensor([1.5, 2.5], dtype=torch.float32, device='npu')
]
scalar = torch.tensor(3.0, dtype=torch.float32)   # Host 侧标量

y_list = kernel_gen_ops.foreach_pow_scalar(x_list, scalar)

# 示例 2：bfloat16 底数，float32 指数
x_bf16 = [torch.ones(2, 2, dtype=torch.bfloat16, device='npu')]
scalar_f32 = torch.tensor(2.0, dtype=torch.float32)
y_bf16 = kernel_gen_ops.foreach_pow_scalar(x_bf16, scalar_f32)
```

## 约束与限制
无