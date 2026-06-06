# aclnnLessEqual

## Functional Description

### Operator Function
The `LessEqual` operator compares two tensors element-wise. For each element pair, the output is `true` or `1` when `x1 <= x2`; otherwise the output is `false` or `0`.

### Formula

$$
y = x_1 \le x_2
$$

### AscendC Implementation Pattern
Comparison operators in AscendC are commonly implemented with a `Compare -> Duplicate -> Select -> Cast` instruction sequence. This sequence materializes the hardware comparison mask into a standard tensor representation.

The pattern is:

1. **`Compare`**: compute the comparison and generate an internal boolean mask.
2. **`Duplicate` and `Select`**: select between predefined `1` and `0` values according to the mask.
3. **`Cast`**: cast the materialized result to the required output data type, such as `int8_t`.

## Interface Definition

### Python Interface
The operation is exposed through a PyBind11 wrapper as `kernel_gen_ops.less_equal()`:

```python
def less_equal(input1, input2):
    """
    Perform custom element-wise less-than-or-equal comparison.

    Args:
        input1 (Tensor): First input tensor in ND format.
        input2 (Tensor): Second input tensor with a broadcast-compatible shape.

    Returns:
        Tensor: Boolean-like tensor after broadcasting-compatible element-wise comparison.
    """
```

## Example

```python
import torch
import kernel_gen_ops

input1 = torch.randint(-10, 10, (8, 2048), dtype=torch.int32)
input2 = torch.randint(-10, 10, (8, 2048), dtype=torch.int32)

result = kernel_gen_ops.less_equal(input1, input2)
print(result)
```

## Constraints and Limitations

- Tensor data format: ND.
- Input tensors must be shape-compatible under the task's comparison rule.
