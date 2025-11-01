import torch
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, softmax_output: torch.Tensor, grad_output: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        grad_softmax = torch.matmul(grad_output, torch.transpose(values, -2, -1))
        output = (grad_softmax - (softmax_output * grad_softmax).sum(-1, keepdim=True)) * softmax_output
        return output
    
class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, softmax_output: torch.Tensor, grad_output: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.inplace_fused_matmul_softmax_grad(softmax_output, grad_output, values)