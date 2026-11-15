from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseOP
from .weight import decode_mxfp4

# Torch has no declared lower bound here; advertise only a present out ABI.
_LINEAR_OUT = getattr(torch.ops.aten.linear, "out", None)


def _linear(x, weight, bias, *, out=None):
    # The same mature ATen linear dispatcher, with an explicit destination.
    if out is not None:
        return _LINEAR_OUT(x, weight, bias, out=out)
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
    ``forward`` retains FP32 accumulators by default. Input-dtype output uses
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

    def _projection_bias(self):
        return self.bias

    def new_with_weight_format(self, weight_format: str, *, chunk_rows: int = 128):
        """Rebuild declared storage without discarding custom forward semantics."""
        if type(self) is not Linear:
            raise NotImplementedError(f"Packed execution for {type(self).__name__} is unsupported")
        return Linear(
            self._input_size,
            self._output_size,
            self.bias is not None,
            weight_format=weight_format,
            device=self.weight.device,
            dtype=self.weight.dtype,
            chunk_rows=chunk_rows,
        )

    def supports_out(self, dtype: torch.dtype) -> bool:
        """Whether loaded weights support same-dtype dense inference output.

        This metadata-only capability does not allocate storage. The direct-out
        call additionally requires 2D inputs and an exact contiguous destination
        on the same device, with storage separate from inputs and parameters.
        """
        return (
            _LINEAR_OUT is not None
            and self.weight_format == "dense"
            and not self.weight.is_meta
            and self.weight.shape == (self._output_size, self._input_size)
            and self.weight.device.type in ("cpu", "cuda")
            and self.weight.dtype == dtype
            and dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
            and not self.weight.requires_grad
            and (
                self.bias is None
                or (
                    self.bias.dtype == dtype
                    and self.bias.shape == (self._output_size,)
                    and self.bias.device == self.weight.device
                    and not self.bias.requires_grad
                )
            )
        )

    def forward(self, x: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
        if out is not None:
            if not self.supports_out(x.dtype):
                raise NotImplementedError(
                    "Caller-owned output requires supported dense inference weights"
                )
            if x.ndim != 2 or x.shape[1] != self._input_size:
                raise ValueError(
                    "Caller-owned linear output requires a 2D input with the configured width"
                )
            if out.shape != (x.shape[0], self._output_size):
                raise ValueError("Caller-owned linear output must have the exact projection shape")
            if x.device != self.weight.device or out.device != x.device or out.dtype != x.dtype:
                raise ValueError(
                    "Caller-owned linear output must preserve input dtype and local device"
                )
            if not out.is_contiguous():
                raise ValueError("Caller-owned linear output must be contiguous")
            if x.requires_grad or out.requires_grad:
                raise ValueError("Caller-owned linear output is inference-only")
            if any(
                torch._C._overlaps(out, source)
                for source in (x, self.weight, self.bias)
                if source is not None
            ):
                raise ValueError(
                    "Caller-owned linear output must use separate storage from inputs and parameters"
                )
            return _linear(x, self.weight, self._projection_bias(), out=out)
        if self.weight_format == "dense":
            return _linear(x, self.weight, self._projection_bias())
        return self._forward_mxfp4(x)

    @torch.no_grad()
    def _forward_mxfp4(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point() or x.shape[-1] != self._input_size:
            raise ValueError(
                f"Packed MXFP4 linear expects floating inputs of width {self._input_size}"
            )
        projection_bias = self._projection_bias()
        if self._backend == "triton":
            from .linear_kernel import mxfp4_linear

            return mxfp4_linear(x, self.weight_packed, self.weight_scale, projection_bias)
        result = x.new_empty((*x.shape[:-1], self._output_size))
        for start in range(0, self._output_size, self._chunk_rows):
            stop = min(start + self._chunk_rows, self._output_size)
            weight = decode_mxfp4(
                self.weight_packed[start:stop], self.weight_scale[start:stop], dtype=x.dtype
            )
            bias = projection_bias[start:stop] if projection_bias is not None else None
            result[..., start:stop] = F.linear(x, weight, bias)
            del weight
        return result


class _LinearTPImpl(Linear):
    """Real implementation of a linear layer with tensor parallelism."""

    def __init__(
        self,
        full_isize: int,
        full_osize: int,
        local_isize: int,
        local_osize: int,
        has_bias: bool,
        **kwargs,
    ):
        self.full_input_size = full_isize
        self.full_output_size = full_osize
        self.local_input_size = local_isize
        self.local_output_size = local_osize
        super().__init__(local_isize, local_osize, has_bias, **kwargs)


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


class LinearRowParallel(_LinearTPImpl):
    """Project TP-local input columns, then reduce the full-width output.

    Constructor input_size is global; local_input_size records its one TP
    partition. An explicit output_tensor receives the local GEMM partial;
    the return value is the fully reduced result, with any residual applied.

    output_key explicitly opts into shared provider-owned output storage. Its
    returned value is borrowed until the next write to the same key, including
    another layer's call; semantic owners copy any value they retain. Without
    preparation, the ordinary allocating projection/reduction/add stays intact.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool = False,
        *,
        output_key: str | None = None,
        tp=None,
        **kwargs,
    ):
        # input_size is GLOBAL; the producer consumes exactly this rank's shard.
        from minisgl.distributed.parallel import TensorParallel

        self._tp = tp if tp is not None else TensorParallel()
        self._tp_size = self._tp.size
        local_input_size = div_even(input_size, self._tp_size)
        super().__init__(input_size, output_size, local_input_size, output_size, has_bias, **kwargs)
        self._output_role = output_key
        self._output_dtype = (
            self.weight.dtype
            if self.weight_format == "dense"
            else kwargs.get("dtype") or torch.get_default_dtype()
        )
        self._output = None
        self._reserved_key = None

    def _projection_bias(self):
        # One full-width bias participates in the TP sum, as in mature row parallel.
        return self.bias if self.bias is not None and self._tp.rank == 0 else None

    def new_with_weight_format(self, weight_format: str, *, chunk_rows: int = 128):
        if type(self) is not LinearRowParallel:
            raise NotImplementedError(f"Packed execution for {type(self).__name__} is unsupported")
        return LinearRowParallel(
            self.full_input_size,
            self.full_output_size,
            self.bias is not None,
            weight_format=weight_format,
            device=self.weight.device,
            dtype=self._output_dtype,
            chunk_rows=chunk_rows,
            output_key=self._output_role,
            tp=self._tp,
        )

    def prepare_inference(self) -> None:
        from minisgl.core import get_global_ctx_or_none

        ctx = get_global_ctx_or_none()
        if self._output_role is None or ctx is None or ctx.max_forward_len is None:
            return
        key = f"{self._output_role}:{self._output_size}:{self._output_dtype}"
        # Every rank participates even when its producer ABI/weights decline.
        output = self._tp.reserve_output(
            key,
            (ctx.max_forward_len, self._output_size),
            self._output_dtype,
            enabled=self.supports_out(self._output_dtype),
        )
        self._output = output
        self._reserved_key = key if output is not None else None

    def forward(
        self,
        x: torch.Tensor,
        *,
        residual: torch.Tensor | None = None,
        output_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output_tensor is not None:
            # Like the SDK output_tensor ABI: producer destination is partial,
            # while this method returns the collective's fully reduced result.
            reduced = self._tp.all_reduce(super().forward(x, out=output_tensor))
            return reduced if residual is None else residual + reduced
        output = self._output
        # Shape/dtype are uniform call semantics; local pointers never select
        # another collective. No per-forward host agreement or buffer growth.
        if (
            output is not None
            and x.ndim == 2
            and x.dtype == self._output_dtype
            and x.shape[0] <= output.shape[0]
        ):
            partial = super().forward(x, out=output[: x.shape[0]])
            return self._tp.all_reduce_residual(partial, residual, output_key=self._reserved_key)
        partial = super().forward(x)
        reduced = self._tp.all_reduce(partial)
        return reduced if residual is None else residual + reduced
