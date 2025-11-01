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
        approximate_int: int, # approximate 参数现在是整数类型
        activateLeft: bool # 这个参数在当前实现中未使用
    ) -> torch.Tensor:
        """
        实现 GeGLUGradV2 的前向和梯度计算。
        此方法旨在计算 y 对 tensor_x 的梯度，给定上游梯度 tensor_dy。

        Args:
            tensor_dy (torch.Tensor): 上游传来的梯度，形状应与 y 相同。
            tensor_x (torch.Tensor): 输入张量，将被分割并用于计算 GeLU。
            gelu_output (torch.Tensor): 原始代码中的占位符，修改后不再用于输出。
            dim (int): 用于 chunk 操作的维度。
            approximate_int (int): GeLU 函数的近似模式：0 表示 'none'/'erf'，1 表示 'tanh'。
            activateLeft (bool): 未在当前实现中使用。

        Returns:
            torch.Tensor: tensor_x 的梯度。
        """
        # 将整数形式的 approximate 映射到字符串形式
        approximate_map = {0: 'none', 1: 'tanh'} # F.gelu 官方支持 'none' (精确) 和 'tanh'
        approximate_str = approximate_map.get(approximate_int, 'none') # 如果没有匹配，默认使用 'none'

        with torch.enable_grad():
            # chunk 并交换 x 和 gate (GeGLU 结构)
            # 注意：GeGLU 通常是 (x1, x2) -> gelu(x1) * x2
            # 你的代码中是 x, gate = gate_chunk, x_chunk，这表示你把右半部分作为 gelu 的输入
            # 如果是标准的 GeGLU，通常是左半部分输入 gelu
            x_chunk, gate_chunk = torch.chunk(tensor_x, 2, dim=dim)
            x_for_mul, gate_for_gelu = gate_chunk, x_chunk # 交换，gate_for_gelu 是输入到 GELU 的部分

            ## 重新计算 gelu
            y_gelu = F.gelu(gate_for_gelu, approximate=approximate_str)

            # 前向计算：x 和 y_gelu 的逐元素乘积
            y = x_for_mul * y_gelu

            # 使用 torch.autograd.grad 计算 tensor_x 的梯度
            # inputs 必须是 tensor_x
            grad_tensor_x = torch.autograd.grad(outputs=y, inputs=tensor_x, grad_outputs=tensor_dy)[0]

        # 返回 tensor_x 的梯度
        return grad_tensor_x

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor, dim: int, approximate: int, activateLeft: bool) -> torch.Tensor:
        # 将整数形式的 approximate 映射到字符串形式
        approximate_map = {0: 'none', 1: 'tanh'} # F.gelu 官方支持 'none' (精确) 和 'tanh'
        approximate_str = approximate_map.get(approximate, 'none') # 如果没有匹配，默认使用 'none'
        x_chunk, gate_chunk = torch.chunk(x, 2, dim=dim)
        x_for_mul, gate_for_gelu = gate_chunk, x_chunk # 交换，gate_for_gelu 是输入到 GELU 的部分

        gelu_output = F.gelu(gate_for_gelu, approximate=approximate_str)
        return kernel_gen_ops.ge_glu_grad_v2(dy, x, gelu_output, dim, approximate, activateLeft)
