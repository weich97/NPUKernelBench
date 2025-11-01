from typing import List, Tuple
import torch
import torch.nn as nn
import kernel_gen_ops

def gelu_compute_erf(input_x: torch.Tensor) -> torch.Tensor:
    """
    Computes a GELU approximation using a polynomial approximation of the erf function.
    This implementation mirrors the provided numpy version for float32 precision.

    Args:
        input_x: A torch.Tensor representing the input.

    Returns:
        A torch.Tensor with the computed GELU approximation.
    """
    # Ensure input is float32 as specified in the original function
    input_x = input_x.to(torch.float32)

    # Apply min/max clamps as in the original numpy code
    input_x_clamped_min = torch.max(input_x, torch.tensor(-13.25, dtype=torch.float32))
    x1 = torch.min(input_x_clamped_min, torch.tensor(5.75, dtype=torch.float32))

    x_pow = x1 * x1

    # Coefficients as float32
    a1 = torch.tensor(-0.3512339572e-8, dtype=torch.float32)
    a2 = torch.tensor(0.2645266170e-6, dtype=torch.float32)
    a3 = torch.tensor(-0.7929488134e-5, dtype=torch.float32)
    a4 = torch.tensor(0.1106123840e-3, dtype=torch.float32)
    a5 = torch.tensor(0.6518995814e-4, dtype=torch.float32)
    a6 = torch.tensor(-0.7266616915e-1, dtype=torch.float32)
    a7 = torch.tensor(-0.1595769883e1, dtype=torch.float32)

    y = x_pow * a1 + a2
    y = y * x_pow + a3
    y = y * x_pow + a4
    y = y * x_pow + a5
    y = y * x_pow + a6
    y = y * x_pow + a7
    y = y * x1

    y = torch.exp(y) + 1.0
    res = input_x / y
    return res

def tanh_parameter_compute(input_x: torch.Tensor) -> torch.Tensor:
    """
    Helper function to compute the x + 0.044715*x^3 term for the tanh GELU approximation.

    Args:
        input_x: A torch.Tensor representing the input.

    Returns:
        A torch.Tensor with the computed value.
    """
    # Ensure input is float32
    input_x = input_x.to(torch.float32)

    y = input_x * input_x
    y = y * input_x
    y = y * torch.tensor(0.044715, dtype=torch.float32)
    result = input_x + y
    return result

def gelu_compute_tanh(input_x: torch.Tensor) -> torch.Tensor:
    """
    Computes a GELU approximation using the tanh formula:
    gelu(x) = x / (1 + exp(-sqrt(8/pi) * (x + 0.044715*x^3)))
    This implementation mirrors the provided numpy version for float32 precision.

    Args:
        input_x: A torch.Tensor representing the input.

    Returns:
        A torch.Tensor with the computed GELU approximation.
    """
    # Ensure input is float32
    input_x = input_x.to(torch.float32)

    tanh_parameter = tanh_parameter_compute(input_x)    # x + 0.044715*x^3

    # -sqrt(8/pi) is approximately -1.5957691
    mul_0 = tanh_parameter * torch.tensor(-1.5957691, dtype=torch.float32)
    temp = torch.exp(mul_0) + 1.0

    res = input_x / temp
    return res

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                scale: torch.Tensor,
                offset: torch.Tensor,
                approximate: str = "tanh",
                quant_mode: str = "static") -> List[torch.Tensor]:

        x_f = x.float()
        scale = scale.float()
        offset = offset.float()

        # 1. GELU 实现，支持 approximate
        if approximate == "none":
            gelu = gelu_compute_erf(x)
        else:
            gelu = gelu_compute_tanh(x)

        # 2. scale 和 offset 支持 broadcast
        if scale.dim() == 1:
            scale = scale.view(*([1] * (x.dim() - 1)), -1)
        if offset is not None and offset.dim() == 1:
            offset = offset.view(*([1] * (x.dim() - 1)), -1)

        # 3. Quantization
        if quant_mode == "static":
            quant = torch.round(gelu * scale + offset).clamp(-128, 127).to(torch.int8)
            return [quant]
        else:
            mul_res = gelu * scale
            max_abs = torch.amax(mul_res.abs(), dim=-1, keepdim=True)  # 最后一维求最大值
            tmp_out_scale = 127.0 / (max_abs + 1e-6)
            out_scale = (1.0 / tmp_out_scale)

            tmp_out_scale = tmp_out_scale.expand_as(mul_res)
            quant = torch.round(mul_res * tmp_out_scale).clamp(-128, 127).to(torch.int8)
            return [quant, out_scale]


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                scale: torch.Tensor,
                offset: torch.Tensor,
                approximate: str = "tanh",
                quant_mode: str = "static") -> List[torch.Tensor]:
        if quant_mode == "static":
            return [kernel_gen_ops.gelu_quant(x, scale, offset, approximate, quant_mode)[0]]
        else:
            return kernel_gen_ops.gelu_quant(x, scale, offset, approximate, quant_mode)