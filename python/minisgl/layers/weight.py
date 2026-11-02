"""Weight storage, packed format decoding and checkpoint selection rules."""

from __future__ import annotations

import re
from collections.abc import Mapping

import torch
from minisgl.layers.base import BaseOP, OPList


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


def _patterns(value, field):
    if not isinstance(value, list) or any(not isinstance(p, str) or not p for p in value):
        raise ValueError(f"quantization {field} must be a list of nonempty strings")
    for pattern in value:
        if pattern.startswith("re:"):
            try:
                re.compile(pattern[3:])
            except re.error as exc:
                raise ValueError(f"Invalid quantization pattern {pattern!r}") from exc
    return value


def quantization_rules(quant) -> tuple[list[list[str]], list[str]]:
    """Validate the supported static weight-only MXFP4 config; return selectors."""
    if quant is None:
        return [], []
    if not isinstance(quant, Mapping):
        raise ValueError("quantization_config must be a mapping")
    if quant.get("format") != "mxfp4-pack-quantized":
        raise NotImplementedError(f"Unsupported quantization format: {quant.get('format')}")
    if quant.get("quant_method", "compressed-tensors") != "compressed-tensors":
        raise NotImplementedError("packed loading requires compressed-tensors MXFP4")
    if quant.get("kv_cache_scheme") is not None:
        raise NotImplementedError("packed loading does not support quantized KV caches")
    groups = quant.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("MXFP4 quantization requires nonempty config_groups")
    targets = []
    for name, group in groups.items():
        if not isinstance(group, Mapping):
            raise ValueError(f"quantization group {name!r} must be a mapping")
        if group.get("format", quant["format"]) != "mxfp4-pack-quantized":
            raise NotImplementedError("Mixed quantization formats are not supported")
        if (
            group.get("input_activations") is not None
            or group.get("output_activations") is not None
        ):
            raise NotImplementedError("packed execution supports weight-only quantization")
        weights = group.get("weights")
        if not isinstance(weights, Mapping):
            raise ValueError(f"quantization group {name!r} requires weights")
        required = {"num_bits": 4, "type": "float", "strategy": "group", "group_size": 32}
        if any(weights.get(key) != value for key, value in required.items()):
            raise ValueError("MXFP4 requires float4 weights with group_size=32")
        if weights.get("dynamic", False) or weights.get("symmetric", True) is not True:
            raise ValueError("MXFP4 requires static symmetric weights")
        if weights.get("scale_dtype", "torch.uint8") not in ("torch.uint8", "uint8"):
            raise ValueError("MXFP4 requires uint8 E8M0 scales")
        if any(
            weights.get(field) is not None for field in ("actorder", "block_structure", "zp_dtype")
        ):
            raise NotImplementedError("Unsupported MXFP4 weight ordering or zero-point scheme")
        targets.append(_patterns(group.get("targets"), "targets"))
    return targets, _patterns(quant.get("ignore", []), "ignore")


def _matches(pattern, name, aliases, linear_types):
    if pattern in linear_types:
        return True
    names = aliases(name) if aliases is not None else (name,)
    if pattern.startswith("re:"):
        return any(re.match(pattern[3:], alias) is not None for alias in names)
    return pattern in names


def replace_quantized_linears(
    model: BaseOP,
    quant,
    *,
    chunk_rows: int = 128,
    aliases=None,
    linear_types=("Linear",),
    exclude=None,
    validate=None,
) -> int:
    """Install weight-only operators before weights are materialized.

    The model adapter supplies name aliases and dense-parameter constraints.
    Checkpoint format validation and recursive replacement are shared.
    """
    from .linear import Linear

    targets, ignore = quantization_rules(quant)
    if not targets:
        return 0
    replacements = 0

    def visit(op: BaseOP, prefix: str):
        nonlocal replacements
        if isinstance(op, OPList):
            children = ((str(i), child) for i, child in op.items())
        else:
            children = ((key, child) for key, child in vars(op).items() if not key.startswith("_"))
        for key, child in list(children):
            name = f"{prefix}.{key}" if prefix else key
            if (
                isinstance(child, Linear)
                and child.weight_format == "dense"
                and not (exclude and exclude(name, child))
            ):
                if any(_matches(p, name, aliases, linear_types) for p in ignore):
                    continue
                matches = sum(
                    any(_matches(p, name, aliases, linear_types) for p in group)
                    for group in targets
                )
                if matches > 1:
                    raise ValueError(f"Multiple quantization groups match {name}")
                if matches:
                    if validate is not None:
                        validate(name, child)
                    # Preserve custom forward methods (e.g. vocab TP reduction)
                    # rather than silently discarding them during replacement.
                    if type(child) is not Linear:
                        raise NotImplementedError(
                            f"Packed execution for {type(child).__name__} is unsupported"
                        )
                    weight = child.weight
                    replacement = Linear(
                        weight.shape[1],
                        weight.shape[0],
                        child.bias is not None,
                        weight_format="mxfp4",
                        device=weight.device,
                        dtype=weight.dtype,
                        chunk_rows=chunk_rows,
                    )
                    replacement.bias = child.bias
                    if isinstance(op, OPList):
                        op.set_op(key, replacement)
                    else:
                        setattr(op, key, replacement)
                    replacements += 1
            elif isinstance(child, BaseOP):
                visit(child, name)

    visit(model, "")
    return replacements
