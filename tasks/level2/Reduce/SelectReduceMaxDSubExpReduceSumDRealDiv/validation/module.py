import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, sel: torch.Tensor, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        input1_sel = input1 * sel
        input2_sel = input2 * (~sel)
        reduce_res = input1_sel + input2_sel

        max_res = torch.amax(reduce_res, dim=-1, keepdim=True)
        sub_res = reduce_res - max_res
        exp_res = torch.exp(sub_res)
        sum_res = torch.sum(exp_res, dim=-1, keepdim=True)
        output = exp_res / sum_res
        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, sel: torch.Tensor, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.select_reduce_max_d_sub_exp_reduce_sum_d_real_div(sel, input1, input2)