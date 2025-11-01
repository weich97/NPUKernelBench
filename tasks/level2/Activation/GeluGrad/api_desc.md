# GeluGrad

## 功能描述

### 算子功能
该AscendC算子用于计算Gelu函数的梯度。

### 计算公式

  - **Gelu函数**
    $$
    y=\frac{x}{\exp((c_{0}x^{2}+c_{1})x)+1}
    $$
    其中，$c_{0}=-0.0713548162726002527220,c_{1}=-1.595769121605730711759$

  - **对于Atlas A2 训练系列产品:**
    $$
    px=\exp((x^{2}\times c_{0}+c_{1})\times x)
    $$
    $$
    res0=(x^{2}\times c_{2}+c_{3})\times x
    $$
    $$
    t=\frac{1}{px+1}
    $$
    $$
    z=(px\times res0\times t^{2}+t)\times dy
    $$
    其中，$c_{0}=-0.0713548162726002527220,c_{1}=-1.595769121605730711759,c_{2}=0.2140644488178007,c_{3}=1.595769121605730711759$

  - **对于Atlas 200I/500 A2推理产品：**
    $$
    g1=\frac{1}{\exp((x^{2}\times c_{0}+c_{1})\times x)+1}
    $$
    $$
    g2=x^{2}\times c_{2}+c_{3}
    $$
    $$
    z=((((g1-1)\times x)\times g2+1)\times g1)\times dy
    $$
    其中，$c_{0}=-0.0713548162726002527220,c_{1}=-1.5957691216057308,c_{2}=-0.21406444881780074632901625683959062,c_{3}=-1.5957691216057307117597842397375274738$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.gelu_grad()` 函数形式提供：

```python
def gelu_grad(dy, x, y):
    """
    实现GELU激活函数的梯度计算操作。
    
    参数:
        dy (Tensor): Device侧的张量，公式中的输入dy。
        x (Tensor): Device侧的张量，公式中的输入x。
        y (Tensor): Device侧的张量，公式中的输入y。
        
    返回:
        Tensor: 计算结果张量，公式中的输出z。
    
    注意:
        - 张量数据格式支持ND。
        - 输出张量的维度与输入张量x一致。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops # 假设的模块名称，请根据实际情况修改

# 创建输入张量
# 假设 dy, x, y 是先前操作中得到的Device侧张量
dy = torch.randn(256, 512, dtype=torch.float32) 
x = torch.randn(256, 512, dtype=torch.float32)
y = torch.nn.functional.gelu(x)

# 使用gelugrad执行计算
result = kernel_gen_ops.gelu_grad(dy, x, y)
```

## 约束与限制

- 输入输出张量数据格式支持ND。