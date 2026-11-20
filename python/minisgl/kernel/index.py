from __future__ import annotations

import functools
import importlib
from typing import TYPE_CHECKING, Tuple

import torch

from .utils import KernelConfig, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi import Module

DEFAULT_INDEX_KERNEL_CONFIG = KernelConfig(num_threads=128, max_occupancy=1, use_pdl=False)


@functools.cache
def _jit_index_module(
    element_size: int,
    *,
    num_splits: int = 1,
    config: KernelConfig = DEFAULT_INDEX_KERNEL_CONFIG,
) -> Module:
    args = make_cpp_args(element_size, num_splits, *config)
    return load_jit(
        "index",
        *args,
        cuda_files=["index.cu"],
        cuda_wrappers=[("launch", f"IndexKernel<{args}>::run")],
    )


@functools.cache
def _native_vocab_embedding():
    name = "minisgl.kernel.vocab_embedding"
    try:
        return importlib.import_module(name).vocab_parallel_embedding
    except ModuleNotFoundError as error:
        if error.name != name and not name.startswith(f"{error.name}."):
            raise
        return None


def indexing(
    weights: torch.Tensor,
    indices: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
    vocab_range: Tuple[int, int] | None = None,  # (start, length)
) -> torch.Tensor:
    if (
        output is None
        and vocab_range is not None
        and weights.is_cuda
        and indices.device == weights.device
        and weights.ndim == 2
        and weights.stride(1) == 1
        and weights.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and indices.ndim == 1
        and indices.dtype in (torch.int32, torch.int64)
        and indices.is_contiguous()
        and 0 <= vocab_range[1] <= weights.shape[0]
    ):
        native = _native_vocab_embedding()
        if native is not None:
            start, length = vocab_range
            return native(indices, weights, start, start + length, 0, 0, 0)

    if output is None:
        output = weights.new_empty(indices.shape[0], weights.shape[1])

    element_size = weights.shape[1] * weights.element_size()
    if element_size % 2048 == 0:
        num_splits = 4
    elif element_size % 1024 == 0:
        num_splits = 2
    else:
        num_splits = 1
    module = _jit_index_module(element_size, num_splits=num_splits)
    module.launch(weights, indices, output, vocab_range)
    return output
