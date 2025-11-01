# aclnnGatherV3

## 功能描述

### 算子功能：
从输入Tensor的指定维度dim，按index中的下标序号提取元素，保存到out Tensor中。
例如，对于输入张量 $self=\begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}$ 和索引张量 index=[1, 0]，
self.index_select(0, index)的结果： $y=\begin{bmatrix}4 & 5 & 6 \\ 1 & 2 & 3\end{bmatrix}$;

x.index_select(1, index)的结果： $y=\begin{bmatrix}2 & 1\\ 5 & 4\\8 & 7\end{bmatrix}$;

### 具体计算过程如下:
以三维张量为例, shape为(3,2,2)的张量 **self** =$\begin{bmatrix}[[1,&2],&[3,&4]], \\ [[5,&6],&[7,&8]], \\ [[9,&10],&[11,&12]]\end{bmatrix}$   **index**=[1, 0], self张量dim=0、1、2对应的下标分别是$l、m、n$,  index是一维

dim为0, index_select(0, index)：   I=index[i];  &nbsp;&nbsp;   out$[i][m][n]$ = self$[I][m][n]$

dim为1, index_select(1, index)：   J=index[j];  &nbsp;&nbsp;&nbsp;    out$[l][j][n]$ = self$[l][J][n]$

dim为2, index_select(2, index)：   K=index[k]; &nbsp;  out$[l][m][k]$ = self$[l][m][K]$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.gather_v3()` 函数形式提供：

```python
def gather_v3(self_tensor, indices, axis):
    """
    实现自定义GatherV3操作。
    
    参数:
        self_tensor (Tensor): 输入张量，Device侧的张量，数据格式支持ND，数据类型支持float32, float16, bfloat16, int64, int16, int32, int8, uint64, uint16, uint32, uint8, bool。
        indices (Tensor): 索引张量，Device侧的张量，数据格式支持ND，数据类型支持int32, int64。
        axis (Tensor): 轴张量，Device侧的张量，数据格式支持ND，数据类型支持int64。
        
    返回:
        Tensor: 计算结果张量，数据类型与self_tensor一致，数据格式支持ND。
    
    注意:
        张量数据格式支持ND
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
self_tensor = torch.randn(4, 2, dtype=torch.float16)
indices = torch.tensor([1, 0], dtype=torch.int32)
axis = torch.tensor([1], dtype=torch.int64)

# 使用gather_v3执行计算
result = kernel_gen_ops.gather_v3(self_tensor, indices, axis)
```

## 约束与限制

- 输入输出张量数据格式支持 ND。
    