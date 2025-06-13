# aclnnArange

## 功能描述

### 算子功能
从start起始到end结束按照step的间隔取值，并返回大小为 $\frac{end-start}{step}+1$的1维张量。其中，步长step是张量中相邻两个值的间隔。

### 计算公式

$$
\text{out}_{i+1} = \text{out}_i + \text{step}
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.arange()` 函数形式提供：

```python
def arange(start, end, step=1):
    """
    实现自定义arange操作，用于生成一个范围内的序列张量。
    
    参数:
        start (float 或 int): 序列的起始值（包含）。
        end (float 或 int): 序列的结束值（不包含）。
        step (float 或 int, 可选): 步长，默认为1。不能为0。
        
    返回:
        Tensor: 包含从start到end（不包括end），以step为步长的1维张量。数据类型为float32，数据格式支持ND。
    
    注意:
        - step 不能为0；
        - 若 (end - start)/step 不是整数，则会截断；
        - 支持正向与反向生成序列（step可为负）；
        - 输出为一维张量。
    """
```
## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量
x = torch.rand(8, 2048, dtype=torch.float)  # 使用正数以避免NaN

# 使用arange执行计算
result = kernel_gen_ops.arange(start=0, end=10, step=1)
```


## 约束与限制

- 张量数据格式支持ND。


