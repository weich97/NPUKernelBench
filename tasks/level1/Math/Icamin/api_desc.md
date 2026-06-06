# aclnnIcamin

## Functional Description

### Operator Semantics
`aclnnIcamin` is an Ascend NPU benchmark operator in the `level1` `Math` task family. The implementation should reproduce the reference tensor semantics used by the validation module and expose the custom kernel through `kernel_gen_ops.icamin()`.

The task specification is intended for kernel-generation research: candidate implementations should preserve reference-level mathematical behavior while optimizing the device-side execution path for the Ascend C runtime.

### Mathematical Definition
The operator follows the tensor relation below, with shape, dtype, broadcasting, and attribute constraints inherited from the benchmark task configuration when applicable.

$$
\text{index} = \underset{0 \le i < n}{\operatorname{argmin}} (|Re(x'_{i})| + |Im(x'_{i})|) + 1
$$

## Interface Definition

### Python Interface
The C++/Ascend implementation is bound to Python through PyBind11 and invoked from the benchmark harness as follows:

```python
def icamin(x, n, incx):
    """Execute `aclnnIcamin` on Ascend NPU tensors."""

```

### Inputs
- `x`: Operator argument supplied by the benchmark input generator. Tensor arguments reside on the device unless the task explicitly defines a host-side scalar or attribute.
- `n`: Operator argument supplied by the benchmark input generator. Tensor arguments reside on the device unless the task explicitly defines a host-side scalar or attribute.
- `incx`: Operator argument supplied by the benchmark input generator. Tensor arguments reside on the device unless the task explicitly defines a host-side scalar or attribute.

### Outputs
- Returns the tensor, tensor list, or in-place updated tensor specified by the reference implementation. Output shape, dtype, layout, and aliasing behavior must be consistent with the validation path.

## Usage Example

```python
import kernel_gen_ops

result = kernel_gen_ops.icamin(x, n, incx)
```

## Constraints and Notes

- The implementation must match the PyTorch/reference semantics used in `validation/module.py`.
- Unless otherwise specified by the task configuration, tensors use the `ND` layout and the dtype set declared in the benchmark metadata.
- Candidate kernels should avoid changing public signatures, generated build files, or validation-side calling conventions.
