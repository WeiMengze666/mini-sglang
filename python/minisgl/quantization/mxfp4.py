"""E2M1 weights with E8M0 block scales; portable bounded-row reference GEMM."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from minisgl.layers.base import BaseOP


def decode_mxfp4(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = 32,
) -> torch.Tensor:
    """Decode uint8 low-nibble-first E2M1 and E8M0 block exponents.

    E2M1 magnitudes are [0,.5,1,1.5,2,3,4,6]; bit 3 is sign.
    E8M0 exponent zero denotes 2**-127 (not zero); 255 denotes NaN.
    Scales cover consecutive groups of 32 columns, independently per row.
    """
    if packed.dtype != torch.uint8 or packed.ndim != 2:
        raise ValueError("MXFP4 packed weights must be a 2-D uint8 tensor")
    if scales.dtype != torch.uint8:
        raise ValueError("MXFP4 scales must be uint8 E8M0 exponents")
    if block_size != 32:
        raise ValueError("MXFP4 requires 32-element scale blocks")
    if not dtype.is_floating_point:
        raise ValueError("Decoded MXFP4 weights require a floating-point dtype")
    rows, half_cols = packed.shape
    cols = half_cols * 2
    # Some exporters serialize group scales as [out, 1, groups, 1].
    if scales.ndim == 4 and scales.shape[1] == scales.shape[3] == 1:
        scales = scales[:, 0, :, 0]
    expected = (rows, (cols + block_size - 1) // block_size)
    if tuple(scales.shape) != expected:
        raise ValueError(f"MXFP4 scale shape {tuple(scales.shape)} != {expected}")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, cols).long()
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32, device=packed.device
    )
    values = table[codes & 7] * torch.where((codes & 8) != 0, -1.0, 1.0)
    scales = scales.to(packed.device)
    factors = torch.exp2(scales.float() - 127)
    factors = factors.masked_fill(scales == 255, torch.nan)
    values *= factors.repeat_interleave(block_size, dim=-1)[:, :cols]
    return values.to(dtype)


class MXFP4Linear(BaseOP):
    """Inference-only linear with U8 E2M1 pairs and U8 E8M0 block scales."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        has_bias: bool = False,
        *,
        device=None,
        dtype=None,
        chunk_rows: int = 128,
        max_decode_elements: int = 1 << 20,
    ):
        if input_size <= 0 or input_size % 32 or output_size <= 0:
            raise ValueError("Packed MXFP4 linear input width must be positive and divisible by 32")
        if chunk_rows <= 0 or max_decode_elements <= 0:
            raise ValueError("Packed MXFP4 decode chunk limits must be positive")
        self.weight_packed = torch.empty(
            output_size, input_size // 2, dtype=torch.uint8, device=device
        )
        self.weight_scale = torch.empty(
            output_size, input_size // 32, dtype=torch.uint8, device=device
        )
        self.bias = torch.empty(output_size, dtype=dtype, device=device) if has_bias else None
        self._input_size = input_size
        self._output_size = output_size
        self._chunk_rows = min(chunk_rows, max(1, max_decode_elements // input_size))
        self._backend = "torch"

    def set_backend(self, backend):
        if backend not in ("torch", "triton"):
            raise ValueError("MXFP4 backend must be torch or triton")
        self._backend = backend

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point() or x.shape[-1] != self._input_size:
            raise ValueError(
                f"Packed MXFP4 linear expects floating inputs of width {self._input_size}"
            )
        if self._backend == "triton":
            from .mxfp4_kernel import mxfp4_linear

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


def read_mxfp4_pair(reader, raw_name, scales_reader, scale_name, spec, *, keep_packed):
    """Slice logical TP rows/columns before materializing packed bytes/scales."""
    weight_view = reader.get_slice(raw_name)
    scale_view = scales_reader.get_slice(scale_name)
    weight_shape, scale_shape = weight_view.get_shape(), scale_view.get_shape()
    if weight_view.get_dtype() != "U8" or len(weight_shape) != 2:
        raise ValueError("MXFP4 packed weights must be a 2-D uint8 tensor")
    if scale_view.get_dtype() != "U8":
        raise ValueError("MXFP4 scales must be uint8 E8M0 exponents")
    rows, half_cols = weight_shape
    cols = half_cols * 2
    scale_4d = len(scale_shape) == 4 and scale_shape[1] == scale_shape[3] == 1
    normalized_scale_shape = (scale_shape[0], scale_shape[2]) if scale_4d else tuple(scale_shape)
    expected = (rows, (cols + 31) // 32)
    if normalized_scale_shape != expected:
        raise ValueError(f"MXFP4 scale shape {tuple(scale_shape)} != {expected}")
    if keep_packed and (not rows or not cols or cols % 32):
        raise ValueError("Packed MXFP4 storage requires positive widths divisible by 32")
    row_slice, packed_slice, scale_slice = slice(None), slice(None), slice(None)
    if spec is not None:
        if spec.axis == 0:
            if not 0 <= spec.start < spec.stop <= rows:
                raise ValueError(f"Logical output slice exceeds checkpoint shape: {raw_name}")
            row_slice = slice(spec.start, spec.stop)
        elif spec.axis == 1:
            if not 0 <= spec.start < spec.stop <= cols:
                raise ValueError(f"Logical input slice exceeds checkpoint shape: {raw_name}")
            if spec.start % 32 or spec.stop % 32:
                raise ValueError("Packed MXFP4 input boundaries must align to 32-element blocks")
            packed_slice = slice(spec.start // 2, spec.stop // 2)
            scale_slice = slice(spec.start // 32, spec.stop // 32)
        else:
            raise ValueError("Packed MXFP4 only supports logical axes 0 and 1")
    weight = weight_view[row_slice, packed_slice]
    scale = (
        scale_view[row_slice, 0, scale_slice, 0] if scale_4d else scale_view[row_slice, scale_slice]
    )
    return weight.contiguous(), scale.contiguous()
