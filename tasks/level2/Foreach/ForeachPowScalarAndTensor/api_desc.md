# aclnnForeachPowScalarAndTensor

## 支持的数据类型
- **指数张量列表** `x` 元素：  
  ‑ `torch.float16`  
  ‑ `torch.float32`  
  ‑ `torch.bfloat16`  
  ‑ `torch.int32`

- **底数标量** `scalar`：  
  ‑ 当 `x.dtype ∈ {float16, float32, bfloat16}` 时，必须为 **torch.float32**  
  ‑ 当 `x.dtype == torch.int32` 时，必须为 **torch.int64**

支持的数据格式：**ND**；张量维度 **2–8**；支持非连续 Tensor；不支持空 Tensor。

---

## 功能描述

### 算子功能
`aclnnForeachPowScalarAndTensor` 将 **Host 侧标量 `scalar`** 作为底数，对 **Device 侧张量列表 `x`** 中的每一个张量进行逐元素幂运算，返回一个新的张量列表 `y`，其形状、数据类型、数据格式与输入完全一致。

### 计算公式
给定输入张量列表  
$$
X = [x_0, x_1, \dots, x_{n-1}]
$$  
以及 Host 侧标量底数 $s$，输出张量列表  
$$
Y = [y_0, y_1, \dots, y_{n-1}]
$$  
逐元素计算：

$$
y_{i,\,j} = s^{\,x_{i,\,j}} \quad \forall i \in [0,n-1], \forall j \in \text{shape}(x_i)
$$

---

## 接口定义

### Python 封装
通过 PyBind11 暴露为：

```python
def foreach_pow_scalar_and_tensor(
        x: List[torch.Tensor],
        scalar: torch.Tensor,
) -> List[torch.Tensor]:
    """
    将 Host 侧标量 scalar 作为底数，对 Device 侧张量列表 x 逐元素进行幂运算。

    参数
    ----
    x : List[torch.Tensor]
        指数张量列表，Device 侧，dtype ∈ {float16, float32, bfloat16, int32}，
        shape 维度 2-8，支持 ND 格式。
    scalar : torch.Tensor
        Host 侧 **0-D 标量** 底数，dtype 与 x 的 dtype 按上文规则匹配。

    返回
    ----
    List[torch.Tensor]
        新张量列表，形状、dtype、格式与输入完全一致。

    """

```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建具有相同数据类型的张量列表
tensor_list = [
    torch.tensor([0.0, 1.0, 2.0], dtype=torch.float),
    torch.tensor([[-1.0, -2.0], [3.0, 4.0]], dtype=torch.float)
]
scalar = torch.tensor(0.25, dtype=torch.float)

# 使用 foreach_reciprocal 对张量列表中的每个张量应用 reciprocal 操作
result_list = kernel_gen_ops.foreach_pow_scalar_and_tensor(scalar, tensor_list)
```

## 约束与限制
无