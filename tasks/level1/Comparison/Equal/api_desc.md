# aclnnEqual

## Functional Description

### Operator Function
The `Equal` operator performs element-wise equality comparison between two input tensors. For each pair of corresponding elements, it returns whether the two values should be treated as equal under the task's numerical rule. Equality comparison is a fundamental operation in tensor programs and is commonly used in validation, masking, and control-flow construction.

### Formula
For floating-point inputs, the implementation may use tolerance-aware arithmetic to materialize equality as a tensor value.

For `half` and `bfloat16_t` inputs:

$$
y_{compute} =
\begin{cases}
MIN\_ACCURACY\_FP16, & \text{if } |x_1 - x_2| > MIN\_ACCURACY\_FP16 \\
|x_1 - x_2|, & \text{if } |x_1 - x_2| < MIN\_ACCURACY\_FP16
\end{cases}
$$

$$
y = 1 - (y_{compute} \times MAX\_MUL\_FP16)^2
$$

For `float` inputs:

$$
y_{compute} =
\begin{cases}
MIN\_ACCURACY\_FP32, & \text{if } |x_1 - x_2| > MIN\_ACCURACY\_FP32 \\
|x_1 - x_2|, & \text{if } |x_1 - x_2| < MIN\_ACCURACY\_FP32
\end{cases}
$$

$$
y = 1 - (y_{compute} \times MAX\_MUL\_FP16)^3
$$

For integer inputs:

$$
y =
\begin{cases}
0, & \text{if } |x_1 - x_2| > 1 \\
1 - |x_1 - x_2|, & \text{if } |x_1 - x_2| < 1
\end{cases}
$$

## Interface Definition

### Python Interface
The operation is exposed through a PyBind11 wrapper as `kernel_gen_ops.equal()`:

```python
def equal(input1, input2):
    """
    Perform custom element-wise tensor equality comparison.

    Args:
        input1 (Tensor): First input tensor in ND format.
        input2 (Tensor): Second input tensor. Shape and data type must match `input1`.

    Returns:
        Tensor: Tensorized equality result for corresponding input elements.
    """
```

## Example

```python
import torch
import kernel_gen_ops

input1 = torch.randn(32, 64, dtype=torch.float32)
input2 = torch.randn(32, 64, dtype=torch.float32)

result = kernel_gen_ops.equal(input1, input2)
print(result)
```

## Constraints and Limitations

- Tensor data format: ND.
- Input tensors must have compatible shapes and data types for element-wise comparison.
