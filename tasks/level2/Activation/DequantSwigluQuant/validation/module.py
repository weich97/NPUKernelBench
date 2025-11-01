import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from typing import Optional, Tuple
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x_tensor: torch.Tensor,
                weight_scale: Optional[torch.Tensor] = None,
                activate_scale: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None,
                quant_scale: Optional[torch.Tensor] = None,
                quant_offset: Optional[torch.Tensor] = None,
                group_index: Optional[torch.Tensor] = None,
                activate_left: bool = False,
                quant_mode: str = "static") -> List[torch.Tensor]:
        if group_index is None:
            group_index = torch.tensor([x_tensor.shape[0]])
        
        x_shape = list(x_tensor.shape)
        x_shape[-1] //= 2
        res_y = torch.zeros(x_shape, dtype=torch.float32, device=x_tensor.device)

        input_dtype = x_tensor.dtype
        offset = 0
        for g_idx in range(group_index.shape[0]):
            groupIdx = group_index[g_idx]
            x = x_tensor[offset: (offset+groupIdx)].float()
            if input_dtype == torch.int32:
                if bias is not None:
                    x = x + bias
                x = x * weight_scale[g_idx]
                if activate_scale is not None:
                    x = x * activate_scale[offset: (offset+groupIdx)]
            gate, up = torch.chunk(x, 2, dim=-1)
            if activate_left:
                output = torch.nn.functional.silu(gate) * up
            else:
                output = torch.nn.functional.silu(up) * gate
            if quant_mode == "static":
                output = output / quant_scale[g_idx] + quant_offset[g_idx]
            elif quant_mode == "dynamic":
                output = output * quant_scale[g_idx]
                abs = torch.abs(output)
                max_values = torch.amax(abs, dim = -1)
                scale_out = max_values / 127
                max_values = 127 / max_values
                output = output * max_values.unsqueeze(1)
            output = torch.clamp(output, -128, 127)
            output = torch.round(output)
            res_y[offset: (offset+groupIdx)] = output
            offset = offset + groupIdx
        return res_y.to(torch.int8)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor,
                weight_scale: Optional[torch.Tensor] = None,
                activate_scale: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None,
                quant_scale: Optional[torch.Tensor] = None,
                quant_offset: Optional[torch.Tensor] = None,
                group_index: Optional[torch.Tensor] = None,
                activate_left: bool = False,
                quant_mode: str = "static") -> List[torch.Tensor]:
        
        # 调用 C++ 实现的算子
        # 这里的参数名必须与 forward 方法的参数名一致
        return kernel_gen_ops.dequant_swiglu_quant(x, weight_scale, activate_scale,
                                                   bias, quant_scale, quant_offset,
                                                   group_index, activate_left, quant_mode)