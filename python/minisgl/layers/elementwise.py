"""Pointwise tensor composition with portable PyTorch semantics."""

import torch


def add3(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
    """Compute (a + b) + c, preserving the intermediate storage rounding.

    A missing final addend keeps the ordinary binary addition. Native inference
    returns fresh storage; broadcasting, gradients and other dtypes retain
    PyTorch's original expression.
    """
    if c is None:
        return a + b
    if all(
        value.is_cuda and value.dtype == torch.bfloat16 and not value.requires_grad
        for value in (a, b, c)
    ):
        from minisgl.kernel.elementwise import try_add3

        result = try_add3(a, b, c)
        if result is not None:
            return result
    return (a + b) + c
