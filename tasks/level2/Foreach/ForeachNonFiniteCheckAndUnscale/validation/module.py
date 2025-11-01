from typing import List, Tuple
import torch
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                scaled_grads: List[torch.Tensor],
                found_inf_tensor: torch.Tensor,
                in_scale_tensor: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        scale_value = in_scale_tensor.item()
        local_found_inf = torch.tensor(0.0, dtype=torch.float, device=found_inf_tensor.device)

        for grad in scaled_grads:
            non_finite = torch.isnan(grad) | torch.isinf(grad)
            if non_finite.any():
                local_found_inf.fill_(1.0)

            if scale_value == 0.0:
                grad.fill_(0.0)
            else:
                grad.mul_(scale_value)

        found_inf_tensor.copy_(local_found_inf)
        return [torch.concat([x.flatten() for x in scaled_grads]), found_inf_tensor]


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                scaled_grads: List[torch.Tensor],
                found_inf_tensor: torch.Tensor,
                in_scale_tensor: torch.Tensor):
        ret = kernel_gen_ops.foreach_non_finite_check_and_unscale(
            scaled_grads,
            found_inf_tensor,
            in_scale_tensor
        )
        return [torch.concat([x.flatten() for x in ret[:-1]]), ret[-1]]