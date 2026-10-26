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

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point() or x.shape[-1] != self._input_size:
            raise ValueError(
                f"Packed MXFP4 linear expects floating inputs of width {self._input_size}"
            )
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
