# aclnnForeachAddcmulList

## 功能描述

### 算子功能
返回一个和输入张量列表同样形状大小的新张量列表，对张量列表`x2`和张量列表`x3`执行逐元素乘法，将结果乘以张量`scalars`后将结果与张量列表`x1`执行逐元素加法。

### 计算公式
$$
x1 = [{x1_0}, {x1_1}, ... {x1_{n-1}}], x2 = [{x2_0}, {x2_1}, ... {x2_{n-1}}], x3 = [{x3_0}, {x3_1}, ... {x3_{n-1}}]\\
scalars = [{scalars_0}, {scalars_1}, ... {scalars_{n-1}}]\\
y = [{y_0}, {y_1}, ... {y_{n-1}}]\\
$$

$$
{\rm y}_i = x1_{i} + {\rm scalars}_i × x2_{i} × x3_{i} (i=0,1,...n-1)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.foreach_addcmul_scalar_list()` 函数形式提供：

```python
def foreach_addcmul_scalar_list(tensor_list1, tensor_list2, tensor_list3, scalars):
    """
    实现自定义 ForeachAddcmulScalarList 操作。
    
    参数:
        tensor_list1 (List[Tensor]): 公式中的x1，Device侧的aclTensorList，表示进行混合运算中加法的第一个输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。shape维度不高于8维，数据格式支持ND。shape与入参x2、x3、scalars和出参out的shape一致。支持非连续的Tensor，不支持空Tensor。该参数中所有tensor的数据类型保存一致。
        tensor_list2 (List[Tensor]): 公式中的x2，Device侧的aclTensorList，表示进行混合运算中乘法的第二个输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟入参x1一致。支持非连续的Tensor，不支持空Tensor。该参数中所有tensor的数据类型保存一致。
        tensor_list3 (List[Tensor]): 公式中的x3，Device侧的aclTensorList，表示进行混合运算中乘法的第三个输入张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟入参x1一致。支持非连续的Tensor，不支持空Tensor。该参数中所有tensor的数据类型保存一致。
        scalars (Tensor): 公式中的scalars，Host侧的aclTensor。表示进行混合运算中乘法的第一个输入标量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。支持维度不高于8维。数据格式支持ND。数据类型和数据格式跟入参x1一致。支持非连续的Tensor，不支持空Tensor。
        
    返回:
        List[Tensor]: 公式中的y，Device侧的aclTensorList，表示进行混合运算的输出张量列表。数据类型支持FLOAT、FLOAT16、BFLOAT16、INT32。shape维度不高于8维，数据格式支持ND。数据类型、数据格式和shape跟入参x1一致。支持非连续的Tensor，不支持空Tensor。该参数中所有tensor的数据类型保存一致。
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
    torch.tensor([3.0, 4.0, 5.0], dtype=torch.float),
    torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float)
]

# 创建scalars张量
scalars = torch.tensor([1.2, 1.0], dtype=torch.float)

# 使用 foreach_addcmul_scalar_list 执行计算
result = kernel_gen_ops.foreach_addcmul_scalar_list(tensor_list1, tensor_list2, tensor_list3, scalars)
```

## 约束与限制
无
