# aclnnIsFinite

## Functional Description

### Operator Function
The `IsFinite` operator determines whether each element of the input tensor is finite. A finite value is neither positive infinity, negative infinity, nor NaN.

### Formula
For each input element `x_i`, the corresponding output element `y_i` is:

$$
y_i =
\begin{cases}
1, & \text{if } x_i \text{ is finite} \\
0, & \text{otherwise}
\end{cases}
$$

### Implementation Principle
The implementation can exploit the IEEE 754 representation of floating-point values. Non-finite values have an exponent field in which all bits are `1`. Finite values have at least one exponent bit equal to `0`.

A hardware-friendly strategy is:

1. **Mask and extract key bits**: use bitwise AND with a type-specific infinity mask, such as `0x7C00` for FP16, to remove the sign bit and isolate the exponent-related representation.
2. **Arithmetic comparison**: interpret the extracted bits as integers and subtract the mask value.
3. **Boolean materialization**: map negative results to `1` (finite) and zero or positive results to `0` (non-finite) using integer operations such as `Maxs` and `Muls`.

## Interface Definition

### Python Interface
The operation is exposed through a PyBind11 wrapper as `kernel_gen_ops.is_finite()`:

```python
def is_finite(x):
    """
    Determine whether each element of the input tensor is finite.

    Args:
        x (Tensor): Device-side input tensor.

    Returns:
        Tensor(bool): Boolean tensor with the same shape as the input.
    """
```

## Example

```python
import torch
import kernel_gen_ops

x_tensor = torch.tensor([1.0, float('inf'), -float('inf'), float('nan'), 0.0, 5.0], dtype=torch.float32)
is_finite_output = kernel_gen_ops.is_finite(x_tensor)
```

## Constraints and Limitations

- Tensor data format: ND.
- Output shape matches input shape.
- Output data type: BOOL.
- Supported input data types: `torch.float16`, `torch.float`, and `torch.bfloat16`.
