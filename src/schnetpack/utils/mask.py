from typing import Callable

import torch


def safe_mask(mask, fn: Callable, operand: torch.Tensor, placeholder: float = 0.) -> torch.Tensor:
    """
    Safe mask which ensures that gradients flow nicely.

    Args:
        mask (array_like): Array of booleans.
        fn (Callable): The function to apply at entries where mask=True.
        operand (array_like): The values to apply fn to.
        placeholder (int): The values to fill in if mask=False.

    Returns: New array with values either being the output of fn or the placeholder value.

    """
    masked = torch.where(mask, operand, 0)
    return torch.where(mask, fn(masked), placeholder)


def safe_mask_special(mask_op, mask_fn, fn: Callable, operand: torch.Tensor, placeholder: float = 0.) -> torch.Tensor:
    """
    Safe mask which ensures that gradients flow nicely. It is an extension of safe_mask to cases where operand and
        fn(operand) differ in their structure and default broadcasting does not yield the desired output.
    Args:
        mask_op (array_like): Array of booleans which selects entries in operand.
        mask_fn (array_like): Array of booleans which selects entries in fn(operand).
        fn (Callable): The function to apply at entries where mask=True.
        operand (array_like): The values to apply fn to.
        placeholder (int): The values to fill in if mask=False.

    Returns: New array with values either being the output of fn or the placeholder value.

    """
    masked = torch.where(mask_op, operand, 0)
    return torch.where(mask_fn, fn(masked), placeholder)


def safe_scale(x: torch.Tensor, scale: torch.Tensor, placeholder: float = 0):
    """
    Autograd safe scaling tensor of x with scale.

    Args:
        x (Array): Tensor to scale, shape: (...)
        scale (Array): Tensor by which x is scaled, shape: (1) or same as x
        placeholder (float): Value to put, when scale equals zero

    Returns: Scaled tensor.

    """
    scale_fn = lambda inputs: scale * inputs
    return safe_mask(mask=scale != 0, fn=scale_fn, operand=x, placeholder=placeholder)
