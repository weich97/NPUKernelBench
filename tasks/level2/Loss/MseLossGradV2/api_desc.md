# aclnnMseLossGrad

## Functional Description

### Operator Semantics
`aclnnMseLossGrad` is an Ascend NPU benchmark operator in the `level2` `Loss` task family. The implementation should reproduce the reference tensor semantics used by the validation module and expose the custom kernel through `kernel_gen_ops.mse_loss_grad()`.

The task specification is intended for kernel-generation research: candidate implementations should preserve reference-level mathematical behavior while optimizing the device-side execution path for the Ascend C runtime.

### Mathematical Definition
The operator follows the tensor relation below, with shape, dtype, broadcasting, and attribute constraints inherited from the benchmark task configuration when applicable.

$$
\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (x_i - y_i)^2 \quad (\text{for reduction='mean'})
$$

$$
\text{Loss} = \sum_{i=1}^{N} (x_i - y_i)^2 \quad (\text{for reduction='sum'} \text{ or 'none'})
$$

$$
\text{out}_i = \text{gradOutput}_i \cdot \text{coefficient} \cdot (x_i - y_i)
$$

## Interface Definition

### Python Interface
The C++/Ascend implementation is bound to Python through PyBind11 and invoked from the benchmark harness as follows:

```python
def mse_loss_grad(gradOutput, self_input, target_input, reduction):
    """Execute `aclnnMseLossGrad` on Ascend NPU tensors."""

```

### Inputs
- `gradOutput`: Operator argument supplied by the benchmark input generator. Tensor arguments reside on the device unless the task explicitly defines a host-side scalar or attribute.
- `self_input`: Operator argument supplied by the benchmark input generator. Tensor arguments reside on the device unless the task explicitly defines a host-side scalar or attribute.
- `target_input`: Operator argument supplied by the benchmark input generator. Tensor arguments reside on the device unless the task explicitly defines a host-side scalar or attribute.
- `reduction`: Operator argument supplied by the benchmark input generator. Tensor arguments reside on the device unless the task explicitly defines a host-side scalar or attribute.

### Outputs
- Returns the tensor, tensor list, or in-place updated tensor specified by the reference implementation. Output shape, dtype, layout, and aliasing behavior must be consistent with the validation path.

## Usage Example

```python
import kernel_gen_ops

result = kernel_gen_ops.mse_loss_grad(gradOutput, self_input, target_input, reduction)
```

## Constraints and Notes

- The implementation must match the PyTorch/reference semantics used in `validation/module.py`.
- Unless otherwise specified by the task configuration, tensors use the `ND` layout and the dtype set declared in the benchmark metadata.
- Candidate kernels should avoid changing public signatures, generated build files, or validation-side calling conventions.
