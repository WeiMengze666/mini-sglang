from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedCommunicator, get_tp_info
from minisgl.utils import div_even

from .base import BaseOP
from .weight import decode_mxfp4


def _linear(x, weight, bias):
    # Use the task-qualified PyTorch/cuBLAS dense path.
    return F.linear(x, weight, bias)


def linear_bf16(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    """Run the shared device-aware BF16 linear dispatch."""
    return _linear(x, weight, bias)


def linear_fp32(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Retain projection accumulators using the official PyTorch GEMM API."""
    if x.is_cuda and x.dtype == weight.dtype and x.dtype in (torch.bfloat16, torch.float16):
        flat = x.reshape(-1, x.shape[-1])
        result = torch.mm(flat, weight.t(), out_dtype=torch.float32)
        return result.view(*x.shape[:-1], weight.shape[0])
    return F.linear(x.float(), weight.float())


def headwise_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Project [tokens, heads, input] using the mature batched GEMM."""
    return torch.bmm(x.transpose(0, 1), weight.transpose(1, 2)).transpose(0, 1)


class MergedLinear:
    """Post-load storage plan for dense projections sharing an input.

    Parameter names stay on the original operators. Predeclared meta views let
    the weight loader place the group in one allocation, which this plan reuses.
    ``forward`` retains FP32 accumulators by default; input-dtype output uses
    the shared dense linear dispatch without an intermediate precision change.
    """

    def __init__(self, weight, widths, names):
        self.weight, self.widths, self.names = weight, widths, names

    @classmethod
    def try_create(cls, projections, *, pad_rows_to=1):
        if pad_rows_to <= 0:
            raise ValueError("Merged projection row alignment must be positive")
        ops = list(projections.values())
        weights = [getattr(op, "weight", None) for op in ops]
        if len(ops) < 2 or any(
            not isinstance(weight, torch.Tensor)
            or weight.ndim != 2
            or not weight.is_contiguous()
            or not weight.dtype.is_floating_point
            or weight.requires_grad
            or min(weight.shape) <= 0
            or getattr(op, "bias", None) is not None
            or getattr(op, "weight_format", "dense") != "dense"
            for op, weight in zip(ops, weights)
        ):
            return None
        first = weights[0]
        if any(
            (weight.shape[1], weight.dtype, weight.device)
            != (first.shape[1], first.dtype, first.device)
            for weight in weights
        ):
            return None
        widths = tuple(weight.shape[0] for weight in weights)
        offset, shared = first.storage_offset(), True
        for weight in weights:
            shared &= (
                weight.untyped_storage()._cdata == first.untyped_storage()._cdata
                and weight.storage_offset() == offset
            )
            offset += weight.numel()
        weight = (
            first.as_strided(
                (sum(widths), first.shape[1]),
                (first.shape[1], 1),
                storage_offset=first.storage_offset(),
            )
            if shared
            else torch.cat(weights, dim=0)
        )
        padding = (-sum(widths)) % pad_rows_to
        if padding:
            weight = torch.cat((weight, weight.new_zeros((padding, first.shape[1]))))
        for op, view in zip(ops, weight[: sum(widths)].split(widths, 0)):
            op.weight = view
        return cls(weight, widths, tuple(projections))

    def forward(self, x, *, output_dtype=torch.float32):
        flat = x.reshape(-1, x.shape[-1])
        if output_dtype == x.dtype and output_dtype != torch.float32:
            output = linear_bf16(flat, self.weight)
        elif output_dtype != torch.float32:
            raise ValueError("Merged projections support input dtype or float32 output")
        elif (
            x.is_cuda
            and x.dtype == self.weight.dtype
            and x.dtype in (torch.bfloat16, torch.float16)
        ):
            # Native merged fronts use caller-owned FP32 accumulation storage.
            output = torch.empty(
                (flat.shape[0], self.weight.shape[0]), device=x.device, dtype=torch.float32
            )
            torch.mm(flat, self.weight.t(), out=output, out_dtype=torch.float32)
        else:
            output = F.linear(flat.float(), self.weight.float())
        output = output.view(*x.shape[:-1], self.weight.shape[0])
        return dict(zip(self.names, output[..., : sum(self.widths)].split(self.widths, -1)))


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
    """A projection whose weight format selects its storage and execution path."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool = False,
        *,
        weight_format: str = "dense",
        device=None,
        dtype=None,
        chunk_rows: int = 128,
        max_decode_elements: int = 1 << 20,
    ):
        if weight_format not in ("dense", "mxfp4"):
            raise ValueError(f"Unsupported linear weight format: {weight_format}")
        self._weight_format = weight_format
        self._input_size, self._output_size = input_size, output_size
        if weight_format == "dense":
            self.weight = torch.empty(output_size, input_size, dtype=dtype, device=device)
        else:
            if input_size <= 0 or input_size % 32 or output_size <= 0:
                raise ValueError(
                    "Packed MXFP4 linear input width must be positive and divisible by 32"
                )
            if chunk_rows <= 0 or max_decode_elements <= 0:
                raise ValueError("Packed MXFP4 decode chunk limits must be positive")
            self.weight_packed = torch.empty(
                output_size, input_size // 2, dtype=torch.uint8, device=device
            )
            self.weight_scale = torch.empty(
                output_size, input_size // 32, dtype=torch.uint8, device=device
            )
            self._chunk_rows = min(chunk_rows, max(1, max_decode_elements // input_size))
            self._backend = "torch"
            self._activation_precision = "bf16"
        self.bias = torch.empty(output_size, dtype=dtype, device=device) if has_bias else None

    @property
    def weight_format(self):
        return self._weight_format

    def set_backend(self, backend, *, activation_precision="bf16"):
        if self.weight_format != "mxfp4":
            raise ValueError("Packed backend selection requires MXFP4 weights")
        if backend not in ("torch", "triton", "flashinfer"):
            raise ValueError("MXFP4 backend must be torch, triton or flashinfer")
        if activation_precision not in ("bf16", "mxfp8"):
            raise ValueError("MXFP4 activation precision must be bf16 or mxfp8")
        self._backend = backend
        self._activation_precision = activation_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_format == "dense":
            return _linear(x, self.weight, self.bias)
        return self._forward_mxfp4(x)

    @torch.no_grad()
    def _forward_mxfp4(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point() or x.shape[-1] != self._input_size:
            raise ValueError(
                f"Packed MXFP4 linear expects floating inputs of width {self._input_size}"
            )
        if self._backend == "triton":
            from .linear_kernel import mxfp4_linear

            return mxfp4_linear(x, self.weight_packed, self.weight_scale, self.bias)
        result = x.new_empty((*x.shape[:-1], self._output_size))
        for start in range(0, self._output_size, self._chunk_rows):
            stop = min(start + self._chunk_rows, self._output_size)
            weight = decode_mxfp4(
                self.weight_packed[start:stop], self.weight_scale[start:stop], dtype=x.dtype
            )
            bias = self.bias[start:stop] if self.bias is not None else None
            result[..., start:stop] = F.linear(x, weight, bias)
            del weight
        return result
