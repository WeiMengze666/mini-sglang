"""Contiguous weight storage for streamed checkpoints with many small tensors."""

import torch


def allocate_weight_views(
    expected: dict[str, torch.Tensor], device: torch.device, alignment: int = 256
) -> dict[str, torch.Tensor]:
    """Allocate one arena per dtype; return aligned, contiguous parameter views.

    Tensor-by-tensor device loading can exhaust allocation/mapping resources
    before VRAM fills on models with hundreds of thousands of small expert
    tensors. Views preserve each layer's ordinary tensor interface and ownership.
    """
    if alignment < 1:
        raise ValueError("Weight alignment must be positive")
    sizes: dict[torch.dtype, int] = {}
    offsets = {}
    for name, tensor in expected.items():
        elements = alignment // tensor.element_size()
        if alignment % tensor.element_size():
            raise ValueError("Weight alignment must be divisible by the element size")
        offset = (sizes.get(tensor.dtype, 0) + elements - 1) // elements * elements
        offsets[name] = offset
        sizes[tensor.dtype] = offset + tensor.numel()
    arenas = {}
    for dtype, size in sizes.items():
        arena = torch.empty(size + alignment // dtype.itemsize, dtype=dtype, device=device)
        shift = (-arena.data_ptr() % alignment) // dtype.itemsize
        arenas[dtype] = arena[shift : shift + size]
    return {
        name: arenas[tensor.dtype][offsets[name] : offsets[name] + tensor.numel()].view(
            tensor.shape
        )
        for name, tensor in expected.items()
    }
