# aclnnIsInf

## Functional Description

### Operator Function
The `IsInf` operator determines whether each input element is positive infinity (`+inf`) or negative infinity (`-inf`).

### Formula
For each input element `x_i`, the corresponding output element `y_i` is:

$$
y_i =
\begin{cases}
\text{True}, & \text{if } x_i = +\infty \text{ or } x_i = -\infty \\
\text{False}, & \text{otherwise}
\end{cases}
$$

### Implementation Principle
The implementation can exploit the IEEE 754 binary representation of infinity. For floating-point infinity, the exponent field is all ones and the mantissa field is all zeros. A sign-mask operation can remove the sign bit so that `+inf` and `-inf` are processed uniformly.

A hardware-friendly strategy is:

1. **Extract key bits**: remove the sign bit with a type-specific mask.
2. **Pattern matching**: compare the extracted representation with the canonical infinity pattern for the current data type.
3. **Result generation**: output `True` if the pattern matches; otherwise output `False`.

## Interface Definition

### Python Interface
The operation is exposed through a PyBind11 wrapper as `kernel_gen_ops.is_inf()`:

```python
def is_inf(x):
    """
    Determine whether each element of the input tensor is positive or negative infinity.

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
is_inf_output = kernel_gen_ops.is_inf(x_tensor)
```

## Constraints and Limitations

- Tensor data format: ND.
- Output shape matches input shape.
- Output data type: BOOL.
- Supported input data types: `torch.float16`, `torch.float`, and `torch.bfloat16`.
