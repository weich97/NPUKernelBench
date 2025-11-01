import torch

def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成模型的输入张量列表和标量。
    """
    input_shape = eval(param.get('input_shape', '[1]'))
    normalized_shape = eval(param.get('normalized_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    alpha = param.get('alpha', 0.3)
    epsilon = param.get('epsilon', 1e-6)

    dy = torch.rand(input_shape, device=device, dtype=dtype)
    x = torch.rand(input_shape, device=device, dtype=dtype)
    gx = torch.rand(input_shape, device=device, dtype=dtype)
    gamma = torch.rand(normalized_shape, device=device, dtype=dtype)

    # Need to generate mean and rstd consistent with DeepNorm forward pass
    # DeepNorm: x_add = x * alpha + gx
    x_add = x.to(torch.float32) * alpha + gx.to(torch.float32)
    
    # Normalization typically over the last `len(normalized_shape)` dimensions
    reduction_dims = tuple(range(x_add.dim() - len(normalized_shape), x_add.dim()))
    
    mean = x_add.mean(dim=reduction_dims, keepdim=True)
    variance = (x_add - mean).pow(2).mean(dim=reduction_dims, keepdim=True)
    rstd = torch.rsqrt(variance + epsilon)

    # Ensure mean and rstd have broadcastable shapes for the grad op if needed
    # Usually they will have `1` in the normalized dimensions.
    mean_rstd_shape = list(input_shape)
    for i in range(len(normalized_shape)):
        mean_rstd_shape[len(input_shape) - 1 - i] = 1 # Set normalized dims to 1
    
    mean = mean.view(mean_rstd_shape)
    rstd = rstd.view(mean_rstd_shape)

    return (dy, x, gx, gamma, mean, rstd, alpha)

def get_init_inputs(param, device=None):
    """
    DeepNormGrad Model does not have initialization parameters.
    """
    return []