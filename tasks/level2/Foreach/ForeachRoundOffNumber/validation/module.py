from typing import List, Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


def _flatten(lst):
    flat = []
    for item in lst:
        if isinstance(item, (list, tuple)):
            flat.extend(_flatten(item))
        elif isinstance(item, torch.Tensor):
            flat.append(item)
        else:
            raise TypeError("Unexpected element type")
    return flat


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: List[torch.Tensor], round_mode: torch.Tensor) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        round_mode = round_mode.item()
        for x_tensor in x:
            if round_mode == 1:
                # Round half to even (standard Python/NumPy/torch.round default)
                result_tensor = torch.round(x_tensor)
            elif round_mode == 2:
                # Round towards negative infinity (floor)
                result_tensor = torch.floor(x_tensor)
            elif round_mode == 3:
                # Round towards positive infinity (ceil)
                result_tensor = torch.ceil(x_tensor)
            elif round_mode == 4:
                # Round half up (round half away from zero for .5)
                # Example: 2.5 -> 3.0, -2.5 -> -3.0
                result_tensor = torch.where(x_tensor >= 0, (x_tensor + 0.5).floor(), (x_tensor - 0.5).ceil())
            elif round_mode == 5:
                # Round towards zero (truncation)
                result_tensor = torch.trunc(x_tensor)
            elif round_mode == 6:
                # Round to nearest odd.
                # This is a specialized rounding. Implementing based on common logic.
                int_part = x_tensor.trunc()
                frac_part_abs = (x_tensor - int_part).abs()

                result_tensor = torch.where(frac_part_abs == 0.5,
                                            torch.where(int_part % 2 == 0,  # If integer part is even
                                                        int_part + torch.sign(x_tensor),  # Nudge to nearest odd
                                                        int_part),  # If odd, keep
                                            torch.round(x_tensor))  # Standard round for non-half cases
            elif round_mode == 7:
                # CAST_FRAC (Fractional part)
                result_tensor = torch.frac(x_tensor)
            else:
                result_tensor = x_tensor

            results.append(result_tensor.to(x_tensor.dtype))
        return results


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: List[torch.Tensor], round_mode: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_round_off_number(x, round_mode)