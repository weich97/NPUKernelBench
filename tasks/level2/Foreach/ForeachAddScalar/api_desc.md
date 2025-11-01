# aclnnForeachAddScalar

## 支持的数据类型  
| 张量        | 数据类型                              |
|-------------|----------------------------------------|
| `x`         | `float16`, `float32`, `bfloat16`, `int32` |
| `scalar`    | - 与 `x` 同 dtype（`float16/float32/int32`）<br>- 当 `x` 为 `bfloat16` 时，`scalar` 可为 `float32` |

## 功能描述

### 算子功能
将 Host 侧标量 `scalar` 加到 Device 侧张量列表 `x` 中的 **每个元素**，返回新的张量列表；原列表保持不变。

### 计算公式
$$
x = [x_0, x_1, \dots, x_{n-1}]
$$

$$
y_i = x_i + \text{scalar},\quad i = 0,1,\dots,n-1
$$

## 接口定义

### Python 接口
```python
def foreach_add_scalar(
    x: List[torch.Tensor],
    scalar: torch.Tensor
) -> List[torch.Tensor]:
    """
    逐元素加标量。

    参数
    ----
    x : List[torch.Tensor]
        Device 侧张量列表；所有张量 dtype 相同，支持 float16/float32/bfloat16/int32；
        shape 维度 ≤ 8，数据格式 ND，支持非连续。
    scalar : torch.Tensor
        Host 侧 **0-D 张量**（标量）；dtype 与 x 的对应关系见“支持的数据类型”表。

    返回
    ----
    List[torch.Tensor]
        新张量列表，dtype、shape、数据格式均与输入列表对应张量相同。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 示例：float32 输入
x_list = [
    torch.randn(2, 3, dtype=torch.float32, device='npu'),
    torch.randn(4, dtype=torch.float32, device='npu')
]
scalar = torch.tensor(2.5, dtype=torch.float32)   # Host 侧标量

y_list = kernel_gen_ops.foreach_add_scalar(x_list, scalar)
```

## 约束与限制
无
