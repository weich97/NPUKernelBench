# aclnnNonFiniteCheckOp

## Functional Description

### Operator Function
The `NonFiniteCheckOp` operator checks whether an input tensor contains any non-finite value. Non-finite values include positive infinity (`+inf`), negative infinity (`-inf`), and NaN.

### Formula
The operator examines all elements `x_i` in the input tensor `x`:

$$
\text{out} =
\begin{cases}
1.0, & \text{if any } x_i \text{ is } \pm\infty \text{ or NaN} \\
0.0, & \text{otherwise}
\end{cases}
$$

The output `out` is a scalar tensor.

### Implementation Principle
A two-stage detection strategy can combine hardware parallel reduction with IEEE 754 bit-level analysis.

1. **Candidate screening**: use reduction instructions such as `ReduceMax` and `ReduceMin` to summarize a large input block. Infinity and NaN values can be exposed through these extremal candidates, reducing the number of values that require bit-level checks.
2. **Exponent extraction**: apply bitwise operations to remove the sign bit and isolate the exponent field.
3. **Pattern matching**: compare the extracted exponent bits with the all-ones exponent pattern (`0xFF` for `float`, `0x1F` for `half`). A match indicates a non-finite value.

## Interface Definition

### Python Interface
The operation is exposed through a PyBind11 wrapper as `kernel_gen_ops.non_finite_check_op()`:

```python
def non_finite_check_op(x):
    """
    Check whether the input tensor contains any non-finite value (+/-Inf or NaN).

    Args:
        x (Tensor): Device-side input tensor.

    Returns:
        Tensor: Scalar tensor (`0.0` or `1.0`) indicating whether non-finite values are present.
    """
```

## Example

```python
import torch
import kernel_gen_ops

x_tensor_non_finite = torch.tensor([1.0, float('inf'), 0.0, float('nan')], dtype=torch.float32)
x_tensor_finite = torch.tensor([1.0, -2.0, 3.0, 0.0], dtype=torch.float32)

output_non_finite = kernel_gen_ops.non_finite_check_op(x_tensor_non_finite)
output_finite = kernel_gen_ops.non_finite_check_op(x_tensor_finite)
```

## Constraints and Limitations

- Tensor data format: ND.
- Output is a scalar tensor.
- Output data type: `torch.float`.
- Supported input data types: `torch.float16`, `torch.float`, and `torch.bfloat16`.
