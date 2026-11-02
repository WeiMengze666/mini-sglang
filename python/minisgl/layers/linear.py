from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.precision import use_strict_bf16
from minisgl.utils import div_even

from .base import BaseOP


def _linear(x, weight, bias):
    if (
        x.is_cuda
        and weight.device == x.device
        and x.ndim > 0
        and weight.ndim == 2
        and use_strict_bf16()
        and x.dtype == weight.dtype == torch.bfloat16
        and x.shape[-1] == weight.shape[1]
        and weight.is_contiguous()
        and (
            bias is None
            or (bias.device == x.device and bias.dtype == x.dtype and bias.is_contiguous())
        )
    ):
        from .linear_kernel import linear

        return linear(x, weight, bias)
    return F.linear(x, weight, bias)


def linear_bf16(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    """Run the shared device-aware BF16 linear dispatch."""
    return _linear(x, weight, bias)


def linear_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Accumulate BF16 projections in FP32 while retaining the BF16 weight layout."""
    if (
        x.is_cuda
        and weight.device == x.device
        and x.ndim > 0
        and weight.ndim == 2
        and use_strict_bf16()
        and x.dtype == weight.dtype == torch.bfloat16
        and x.shape[-1] == weight.shape[1]
        and weight.is_contiguous()
    ):
        from .linear_kernel import linear

        return linear(x, weight, dtype=torch.float32)
    return F.linear(x.float(), weight.float())


def headwise_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Project [tokens, heads, input] with separate weights for each head."""
    if (
        x.is_cuda
        and weight.device == x.device
        and x.ndim == weight.ndim == 3
        and use_strict_bf16()
        and x.dtype == weight.dtype == torch.bfloat16
        and x.shape[1] == weight.shape[0]
        and x.shape[2] == weight.shape[2]
    ):
        from .linear_kernel import headwise_linear as project

        return project(x, weight)
    return torch.einsum("thk,hnk->thn", x, weight)


class _LinearTPImpl(BaseOP):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        self.weight = torch.empty(local_osize, local_isize)
        self.bias = torch.empty(local_osize) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _linear(x, self.weight, self.bias)


class LinearReplicated(_LinearTPImpl):
    """
    Linear layer where weights are replicated (not sharded) across all TP ranks.
    Each GPU holds the full weight matrix.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        super().__init__(
            full_isize=input_size,
            full_osize=output_size,
            local_isize=input_size,
            local_osize=output_size,
            has_bias=has_bias,
        )


class LinearColParallelMerged(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_sizes: List[int],
        has_bias: bool,
    ):
        # check that all output sizes are divisible by tp_size
        tp_info = get_tp_info()
        tp_output_sizes = [div_even(size, tp_info.size) for size in output_sizes]
        output_size = sum(output_sizes)
        tp_output_size = sum(tp_output_sizes)
        super().__init__(input_size, output_size, input_size, tp_output_size, has_bias)


class LinearQKVMerged(_LinearTPImpl):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int,
        num_qo_heads: int,
        num_kv_heads: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()

        local_num_qo = div_even(num_qo_heads, tp_info.size)
        local_num_kv = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        full_isize = hidden_size
        full_osize = (num_qo_heads + 2 * num_kv_heads) * head_dim
        local_isize = hidden_size
        local_osize = (local_num_qo + 2 * local_num_kv) * head_dim
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)


class LinearOProj(_LinearTPImpl):
    def __init__(self, input_size: int, output_size: int, has_bias: bool):
        tp_info = get_tp_info()
        full_isize = input_size
        full_osize = output_size
        local_isize = div_even(input_size, tp_info.size)
        local_osize = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(full_isize, full_osize, local_isize, local_osize, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class LinearRowParallel(_LinearTPImpl):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool,
    ):
        tp_info = get_tp_info()
        local_input_size = div_even(input_size, tp_info.size)
        local_output_size = output_size
        self._comm = DistributedCommunicator()
        self._tp_size = tp_info.size
        super().__init__(input_size, output_size, local_input_size, local_output_size, has_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _linear(x, self.weight, self.bias)
        if self._tp_size > 1:
            y = self._comm.all_reduce(y)
        return y


class Linear(BaseOP):
    def __init__(self, input_size: int, output_size: int, has_bias: bool = False):
        self.weight = torch.empty(output_size, input_size)
        self.bias = torch.empty(output_size) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _linear(x, self.weight, self.bias)
