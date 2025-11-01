import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the BatchNormV3 operator's forward method.
    """
    input_shape = eval(param.get('input_shape', '[1]'))
    num_features = param.get('num_features', 1)
    dtype_str = param.get('dtype', 'float16')
    input_dtype = getattr(torch, dtype_str) # Use a distinct name for input dtype
    
    affine = param.get('affine', True)
    
    running_mean = torch.rand(num_features, device=device, dtype=input_dtype)
    running_var = torch.rand(num_features, device=device, dtype=input_dtype) + 1e-3 # Add small value to avoid zero variance
    
    training = bool(int(param.get('training', 1)))

    input_tensor = torch.rand(input_shape, device=device, dtype=input_dtype)
    
    weight = None
    bias = None
    if affine:
        # --- FIX START ---
        # Force weight and bias to float16, as float32 seems unsupported by aclnnBatchNorm for these tensors
        # Or use bfloat16 if your NPU prefers it and is specified in the error message (if it were populated)
        weight_bias_dtype = torch.float16 # or torch.bfloat16 if needed
        weight = torch.rand(num_features, device=device, dtype=weight_bias_dtype)
        bias = torch.rand(num_features, device=device, dtype=weight_bias_dtype)
        # --- FIX END ---

    return (input_tensor, weight, bias, running_mean, running_var, training)

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters (num_features, eps, momentum, affine) for the model.
    """
    num_features = param.get('num_features', 1)
    eps = float(param.get('epsilon', 1e-5))
    momentum = float(param.get('momentum', 0.1))
    affine = param.get('affine', True)
    
    return [num_features, eps, momentum, affine]