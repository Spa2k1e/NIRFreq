import os
import torch
from typing import Tuple


def check_and_make(path: str) -> None:
    """
    Checks if a directory exists, and if not, creates it.

    Args:
        path (str): The path to the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def regularize_inputs(*args: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """
    Clips tensor values to the range [0.0, 1.0].

    Args:
        *args (torch.Tensor): A variable number of input tensors.

    Returns:
        A tuple of clipped tensors.
    """
    return tuple(torch.clamp(v, 0.0, 1.0) for v in args)


def count_model_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Counts the total and trainable parameters of a PyTorch model.

    Args:
        model (torch.nn.Module): The model to inspect.

    Returns:
        A tuple containing:
        - int: The total number of parameters.
        - int: The number of trainable parameters.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
