# aclnnForeachAddcdivScalar

## 功能描述

### 算子功能
对多个张量进行逐元素加、乘、除操作，返回一个和输入张量列表同样形状大小的新张量列表，$x2_{i}$和$x3_{i}$进行逐元素相除，并将结果乘以scalar，再与$x1_{i}$相加。

### 计算公式
$$
x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}], x3 = [{x3_0}, {x3_1}, ... {x3_{n-1}}]\\  
y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
$$

$$
y_i = {x1}_{i}+ \frac{{x2}_{i}}{{x3}_{i}}\times{scalar} (i=0,1,...n-1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_addcdiv_scalar()` 函数形式提供：

```python
def foreach_addcdiv_scalar(tensor_list1, tensor_list2, tensor_list3, scalar):
    """
    实现自定义 ForeachAddcdivScalar 操作。
    
    参数:
        tensor_list1 (List[Tensor]): 公式中的x1，Device侧的aclTensorList，表示混合运算中加法的第一个输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。shape与入参x2、x3和出参out的shape一致。支持非连续的Tensor，不支持空Tensor。该参数中所有tensor的数据类型保存一致。
        tensor_list2 (List[Tensor]): 公式中的x2，Device侧的aclTensorList，表示混合运算中除法的第一个输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟x1入参一致。支持非连续的Tensor，不支持空Tensor。该参数中所有tensor的数据类型保存一致。
        tensor_list3 (List[Tensor]): 公式中的x3，Device侧的aclTensorList，表示混合运算中除法的第二个输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟x1入参一致。支持非连续的Tensor，不支持空Tensor。该参数中所有tensor的数据类型保存一致。
        scalar (Tensor): 公式中的scalar，Host侧的aclTensor，表示混合运算中乘法的第二个输入标量。数据格式支持ND。支持非连续的Tensor，不支持空Tensor。支持维度不高于8维。元素个数为1。数据类型支持FLOAT、FLOAT16，且与入参x1的数据类型具有一定对应关系：(1)当x1的数据类型为FLOAT、FLOAT16时，数据类型与x1的数据类型保持一致。(2)当x1的数据类型为BFLOAT16时，数据类型支持FLOAT。
        
    返回:
        List[Tensor]: 公式中的y，Device侧的aclTensorList，表示混合运算的输出张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟x1入参一致。支持非连续的Tensor，不支持空Tensor。该参数中所有tensor的数据类型保存一致。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建第一个输入张量列表
tensor_list1 = [
    torch.tensor([1.0, 2.0, 3.0], dtype=torch.float),
    torch.tensor([[4.0, 5.0], [6.0, 7.0]], dtype=torch.float)
]

# 创建第二个输入张量列表
tensor_list2 = [
    torch.tensor([2.0, 3.0, 4.0], dtype=torch.float),
    torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float)
]

# 创建第三个输入张量列表
tensor_list3 = [
    torch.tensor([1.0, 1.0, 1.0], dtype=torch.float),
    torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float)
]

# 创建scalar张量
scalar = torch.tensor([1.2], dtype=torch.float)

# 使用 foreach_addcdiv_scalar 执行计算
result = kernel_gen_ops.foreach_addcdiv_scalar(tensor_list1, tensor_list2, tensor_list3, scalar)
```

## 约束与限制
无
