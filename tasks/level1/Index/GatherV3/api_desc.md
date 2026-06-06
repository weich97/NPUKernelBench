# aclnnGatherV3

## Functional Description

### Operator Function
The `GatherV3` operator extracts elements from an input tensor along a specified axis according to an index tensor, and writes the selected values to the output tensor.

For example, given:

$$
self = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9\end{bmatrix}, \quad index = [1, 0]
$$

`self.index_select(0, index)` produces:

$$
\begin{bmatrix}4 & 5 & 6 \\ 1 & 2 & 3\end{bmatrix}
$$

`self.index_select(1, index)` produces:

$$
\begin{bmatrix}2 & 1 \\ 5 & 4 \\ 8 & 7\end{bmatrix}
$$

### Computation Rule
For a three-dimensional tensor `self` with logical indices `(l, m, n)` and an index vector `index`:

- `axis = 0`: `I = index[i]`, then `out[i][m][n] = self[I][m][n]`.
- `axis = 1`: `J = index[j]`, then `out[l][j][n] = self[l][J][n]`.
- `axis = 2`: `K = index[k]`, then `out[l][m][k] = self[l][m][K]`.

## Interface Definition

### Python Interface
The operation is exposed through a PyBind11 wrapper as `kernel_gen_ops.gather_v3()`:

```python
def gather_v3(self_tensor, indices, axis):
    """
    Execute a custom GatherV3/index_select operation.

    Args:
        self_tensor (Tensor): Device-side input tensor in ND format. Supported data types include float32, float16, bfloat16, int64, int16, int32, int8, uint64, uint16, uint32, uint8, and bool.
        indices (Tensor): Device-side index tensor in ND format. Supported data types are int32 and int64.
        axis (Tensor): Device-side scalar or tensor specifying the gather axis. Supported data type is int64.

    Returns:
        Tensor: Gathered output tensor with the same data type as `self_tensor`.
    """
```

## Example

```python
import torch
import kernel_gen_ops

self_tensor = torch.randn(4, 2, dtype=torch.float16)
indices = torch.tensor([1, 0], dtype=torch.int32)
axis = torch.tensor([1], dtype=torch.int64)

result = kernel_gen_ops.gather_v3(self_tensor, indices, axis)
```

## Constraints and Limitations

- Input and output tensor data format: ND.
- `indices` must contain valid positions along the selected axis.
- The output data type matches `self_tensor`.
