# aclnnTopKV3

## 功能描述

### 算子功能
返回输入Tensor在指定维度上的k个极值及索引。

### 计算公式
此算子为获取极值及索引操作，无特定数学公式。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.top_kv3()` 函数形式提供：

```python
def top_kv3(self_tensor, k, dim, largest, sorted):
    """
    实现TopKV3算子功能。
    
    参数:
        self_tensor (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        k (int): 计算维度上输出的极值个数，取值范围为[0, self_tensor.size(dim)]。
        dim (int): 计算维度，取值范围为[-self_tensor.dim(), self_tensor.dim())。
        largest (bool): 布尔型，True表示计算维度上的结果应由大到小输出，False表示计算维度上的结果由小到大输出。
        sorted (bool): 布尔型，True表示输出结果排序，False表示输出结果不排序。
        
    返回:
        Tuple[Tensor, Tensor]: 计算结果张量，第一个张量为极值，第二个张量为索引，数据类型与输入一致，数据格式支持ND。
    
    注意:
        张量数据格式支持ND
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
self_tensor = torch.randn(8, 2048, dtype=torch.float16)
k = 2
dim = 1
largest = True
sorted_val = True

# 使用top_kv3执行计算
values, indices = kernel_gen_ops.top_kv3(self_tensor, k, dim, largest, sorted_val)
```

## 约束与限制

无。
    