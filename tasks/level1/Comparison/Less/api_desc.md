# aclnnLess

## Functional Description

### Operator Function
The `Less` operator compares two input tensors element-wise and returns a tensor with the same logical shape. Each output element indicates whether the corresponding element of the first input is strictly smaller than the corresponding element of the second input.

### Formula

$$
x_1 = [x_{1,0}, x_{1,1}, \ldots, x_{1,n-1}], \quad
x_2 = [x_{2,0}, x_{2,1}, \ldots, x_{2,n-1}]
$$

$$
y_i =
\begin{cases}
\text{True}, & \text{if } x_{1,i} < x_{2,i} \\
\text{False}, & \text{otherwise}
\end{cases}
\quad i = 0, 1, \ldots, n-1
$$

### AscendC Implementation Pattern
Comparison operators in AscendC are commonly implemented with a `Compare -> Duplicate -> Select -> Cast` instruction sequence. This sequence materializes the hardware comparison mask into a standard tensor representation.

The pattern is:

1. **`Compare`**: compute the comparison and generate an internal boolean mask.
2. **`Duplicate` and `Select`**: select between predefined `1` and `0` values according to the mask.
3. **`Cast`**: cast the materialized result to the required output data type, such as `int8_t`.

## Interface Definition

### Python Interface
The operation is exposed through a PyBind11 wrapper as `kernel_gen_ops.less()`:

```python
def less(tensor1, tensor2):
    """
    Perform custom element-wise less-than comparison.

    Args:
        tensor1 (Tensor): First input tensor in ND format.
        tensor2 (Tensor): Second input tensor. Shape and data type must match `tensor1`.

    Returns:
        Tensor(bool): Boolean tensor with the same shape as the inputs.
    """
```

## Example

```python
import torch
import kernel_gen_ops

tensor1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
tensor2 = torch.tensor([2.0, 1.0, 4.0], dtype=torch.float32)

result = kernel_gen_ops.less(tensor1, tensor2)
```

## Constraints and Limitations

- Tensor data format: ND.
- Input shapes must be compatible for element-wise comparison.
