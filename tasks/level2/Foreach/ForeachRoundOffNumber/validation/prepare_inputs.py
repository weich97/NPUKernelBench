import torch
import random


def get_inputs(param, device=None):
    """
    Generate input tensor list for the ForeachRoundOffNumber operator.
    """
    num_tensors = param.get('num_tensors', 1)
    input_shapes = eval(param.get('input_shapes', '[[10]]'))  # List of shapes
    dtype_str = param.get('dtype', 'float32')
    round_mode = int(param.get('round_mode', 1))

    torch_dtype = getattr(torch, dtype_str)
    scalar_dtype = torch.float32 if torch_dtype == torch.bfloat16 else torch_dtype
    round_mode_tensor = torch.tensor(round_mode, dtype=scalar_dtype, device=device)

    x_tensors: List[torch.Tensor] = []
    for i in range(num_tensors):
        shape = input_shapes[i % len(input_shapes)]  # Cycle through provided shapes

        # Generate random floats, including some with .5 fractional parts
        # to test rounding modes effectively.

        # Using torch.empty and then filling with uniform random values
        # This is equivalent to np.random.uniform for generating the base array
        torch_tensor = torch.empty(shape, dtype=torch_dtype, device=device).uniform_(-10.5, 10.5)

        # Inject exact .5 values or values near integer boundaries
        if torch_tensor.numel() > 0:
            # torch.randint for index, then direct assignment
            torch_tensor.view(-1)[torch.randint(0, torch_tensor.numel(), (1,))] = random.randint(-5, 5) + 0.5
        if torch_tensor.numel() > 1:
            torch_tensor.view(-1)[torch.randint(0, torch_tensor.numel(), (1,))] = random.randint(-5, 5) - 0.5
        if torch_tensor.numel() > 2:
            torch_tensor.view(-1)[torch.randint(0, torch_tensor.numel(), (1,))] = random.randint(-5, 5)  # Exact integer

        x_tensors.append(torch_tensor)

    return (x_tensors, round_mode_tensor)


def get_init_inputs(param, device=None):
    """

    """

    return []