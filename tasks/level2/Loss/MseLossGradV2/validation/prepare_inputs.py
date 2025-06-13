import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the MseLossGrad operator's forward method.
    """
    input_shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)

    input_predict = torch.rand(input_shape, device=device, dtype=dtype) * 200 - 100 # Range -100 to 100
    input_label = torch.rand(input_shape, device=device, dtype=dtype) * 200 - 100
    input_dout = torch.rand(input_shape, device=device, dtype=dtype) * 200 - 100

    return (input_dout, input_predict, input_label) # Order: gradOutput, self, target as per aclnn signature

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters (reduction) for the MseLossGrad model.
    """
    reduction = param.get('reduction', 'mean') # String
    return [reduction]