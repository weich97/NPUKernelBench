# aclnnSelectReduceMaxDSubExpReduceSumDRealDiv

## 功能描述

### 算子功能
实现了数据select + reduce + max + sub + exp + reduce_sum_d + div，返回结果的功能。

### 计算公式
$$\text{input1_sel} = \text{input1} \cdot \text{sel} $$
$$\text{input2_sel} = \text{input2} \cdot (\neg \text{sel}) $$
$$\text{add_res} = \text{input1_sel} + \text{input2_sel} $$
$$\text{max_res} = \max(\text{add_res}, \text{axis}=-1) $$
$$\text{sub_res} = \text{add_res} - \text{max_res} $$
$$\text{exp_res} = e^{\text{sub_res}} $$
$$\text{sum_res} = \sum(\text{exp_res}, \text{axis}=-1) $$
$$\text{result} = \frac{\text{exp_res}}{\text{sum_res}} $$
$$\text{output}_1 = \frac{1}{1 + e^{-(\text{sub_res} \cdot 1)}} $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.select_reduce_max_d_sub_exp_reduce_sum_d_real_div()` 函数形式提供：


```python
def select_reduce_max_d_sub_exp_reduce_sum_d_real_div(sel, input1, input2):
    """
    实现融合选择 + softmax + sigmoid 操作的算子。

    该函数根据布尔掩码 `sel` 在 `input1` 和 `input2` 之间进行元素选择，然后在最后一维上执行
    数值稳定的 softmax 和 sigmoid 操作，返回两个输出。

    参数:
        sel (Tensor): bool 类型张量，用于选择 `input1` 和 `input2` 的元素。
                      当 sel 为 True 时选择 input1，否则选择 input2。
                      形状必须与 input1/input2 相同。
        input1 (Tensor): 第一个输入张量。形状必须与 `input2` 相同。
        input2 (Tensor): 第二个输入张量。形状必须与 `input1` 相同。
        

    返回:
        Tuple[Tensor, Tensor]:
            - softmax_result: 在选择结果上按最后一维计算得到的 softmax。
            - sigmoid_result: 对中心化后的选择结果执行 sigmoid 激活。

    注意:
        - 所有输入张量的形状必须一致；
        - 仅在最后一维（axis=-1）上执行 softmax/sigmoid 操作；
        - 本函数实现了数值稳定版本的 softmax，即减去最大值后再执行 exp；
        - 可用于注意力机制、分类预测中的选择性归一化处理。
    """


```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建输入张量（例如：Transformer中的注意力 logits）
input1 = torch.randn(2, 4, 8, 16, dtype=torch.float32)
input2 = torch.randn(2, 4, 8, 16, dtype=torch.float32)
sel = torch.randint(0, 2, (2, 4, 8, 16), dtype=torch.bool)  # 随机布尔 mask

# 执行计算
softmax_result, sigmoid_result = kernel_gen_ops.select_reduce_max_d_sub_exp_reduce_sum_d_real_div(sel, input1, input2)

```
## 约束与限制

- 张量数据格式支持ND。


