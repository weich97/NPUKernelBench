import torch
import torch.nn as nn
import torch.nn.functional as F
import ast # Assuming this is used for literal_eval
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(
        self,
        tensor_dy: torch.Tensor,
        tensor_x: torch.Tensor,
        dim: int,
        approximate_int: int,  # Implementation note.
        activateLeft: bool  # Implementation note.
    ) -> torch.Tensor:
        """
        Reference implementation detail.
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        # Implementation note.
        approximate_map = {0: 'none', 1: 'tanh'}  # Implementation note.
        approximate_str = approximate_map.get(approximate_int, 'none')  # Implementation note.

        with torch.enable_grad():
            # Implementation note.
            # Implementation note.
            # Implementation note.
            # Implementation note.
            x_chunk, gate_chunk = torch.chunk(tensor_x, 2, dim=dim)
            x_for_mul, gate_for_gelu = gate_chunk, x_chunk  # Implementation note.

            # Implementation note.
            y_gelu = F.gelu(gate_for_gelu, approximate=approximate_str)

            # Implementation note.
            y = x_for_mul * y_gelu

            # Implementation note.
            # Implementation note.
            grad_tensor_x = torch.autograd.grad(outputs=y, inputs=tensor_x, grad_outputs=tensor_dy)[0]

        # Implementation note.
        return grad_tensor_x

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor, dim: int, approximate: int, activateLeft: bool) -> torch.Tensor:
        # Implementation note.
        approximate_map = {0: 'none', 1: 'tanh'}  # Implementation note.
        approximate_str = approximate_map.get(approximate, 'none')  # Implementation note.
        x_chunk, gate_chunk = torch.chunk(x, 2, dim=dim)
        x_for_mul, gate_for_gelu = gate_chunk, x_chunk  # Implementation note.

        gelu_output = F.gelu(gate_for_gelu, approximate=approximate_str)
        return kernel_gen_ops.ge_glu_grad_v2(dy, x, gelu_output, dim, approximate, activateLeft)
