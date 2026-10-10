"""Capacity-preserving MXFP4 reference execution for Kimi-K3.

Only packed bytes and block scales are retained. Inference decodes a bounded
number of output rows at a time; this deliberately favors portability and
capacity over speed, and is not a fused low-bit GEMM implementation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from minisgl.layers.base import BaseOP, OPList

from .kimi_moe import KimiLinear

# These projections are explicitly unquantized in the pinned K3 reference,
# despite using our small KimiLinear class for the dense computation.
_NATIVE_DENSE = {
    "routed_expert_down_proj",
    "routed_expert_up_proj",
    "self_attention_res_proj",
    "mlp_res_proj",
    "output_attn_res_proj",
}


def _get(config, name, default=None):
    return (
        config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)
    )


def _patterns(value, field):
    if not isinstance(value, list) or any(not isinstance(p, str) or not p for p in value):
        raise ValueError(f"Kimi quantization {field} must be a list of nonempty strings")
    for pattern in value:
        if pattern.startswith("re:"):
            try:
                re.compile(pattern[3:])
            except re.error as exc:
                raise ValueError(f"Invalid Kimi quantization pattern {pattern!r}") from exc
    return value


def quantization_rules(config) -> tuple[list[list[str]], list[str]]:
    """Validate the supported static weight-only MXFP4 config; return selectors."""
    quant = _get(config, "quantization_config")
    if quant is None:
        return [], []
    if not isinstance(quant, Mapping):
        raise ValueError("Kimi quantization_config must be a mapping")
    if quant.get("format") != "mxfp4-pack-quantized":
        raise NotImplementedError(f"Unsupported Kimi quantization format: {quant.get('format')}")
    if quant.get("quant_method", "compressed-tensors") != "compressed-tensors":
        raise NotImplementedError("Kimi packed loading requires compressed-tensors MXFP4")
    if quant.get("kv_cache_scheme") is not None:
        raise NotImplementedError("Kimi packed loading does not support quantized KV caches")
    groups = quant.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise ValueError("Kimi MXFP4 quantization requires nonempty config_groups")
    targets = []
    for name, group in groups.items():
        if not isinstance(group, Mapping):
            raise ValueError(f"Kimi quantization group {name!r} must be a mapping")
        if group.get("format", quant["format"]) != "mxfp4-pack-quantized":
            raise NotImplementedError("Mixed Kimi quantization formats are not supported")
        if (
            group.get("input_activations") is not None
            or group.get("output_activations") is not None
        ):
            raise NotImplementedError("Kimi packed execution supports weight-only quantization")
        weights = group.get("weights")
        if not isinstance(weights, Mapping):
            raise ValueError(f"Kimi quantization group {name!r} requires weights")
        required = {"num_bits": 4, "type": "float", "strategy": "group", "group_size": 32}
        if any(weights.get(key) != value for key, value in required.items()):
            raise ValueError("Kimi MXFP4 requires float4 weights with group_size=32")
        if weights.get("dynamic", False) or weights.get("symmetric", True) is not True:
            raise ValueError("Kimi MXFP4 requires static symmetric weights")
        if weights.get("scale_dtype", "torch.uint8") not in ("torch.uint8", "uint8"):
            raise ValueError("Kimi MXFP4 requires uint8 E8M0 scales")
        if any(
            weights.get(field) is not None for field in ("actorder", "block_structure", "zp_dtype")
        ):
            raise NotImplementedError("Unsupported Kimi MXFP4 weight ordering or zero-point scheme")
        targets.append(_patterns(group.get("targets"), "targets"))
    return targets, _patterns(quant.get("ignore", []), "ignore")


def _matches(pattern: str, name: str) -> bool:
    # Config selectors can use either checkpoint-native or normalized names.
    names = {name, "language_model." + name}
    for alias in tuple(names):
        names.add(alias.replace(".mlp.", ".block_sparse_moe."))
    if pattern in ("Linear", "KimiLinear"):
        return True
    if pattern.startswith("re:"):
        return any(re.match(pattern[3:], alias) is not None for alias in names)
    return pattern in names


class KimiPackedLinear(BaseOP):
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
            raise ValueError("Packed Kimi linear input width must be positive and divisible by 32")
        if chunk_rows <= 0 or max_decode_elements <= 0:
            raise ValueError("Packed Kimi decode chunk limits must be positive")
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
        from .kimi_weight import decode_mxfp4

        if not x.is_floating_point() or x.shape[-1] != self._input_size:
            raise ValueError(
                f"Packed Kimi linear expects floating inputs of width {self._input_size}"
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


def replace_quantized_linears(model: BaseOP, config, *, chunk_rows: int = 128) -> int:
    """Replace selected linears, dropping dense storage. Returns replacement count.

    Run on a meta-device model before materializing checkpoint weights: converting
    an already allocated dense model cannot undo its construction-time peak.
    Native K3 router, latent-space and attention-residual score projections remain
    dense, as required by the public checkpoint and upstream architecture.
    """
    targets, ignore = quantization_rules(config)
    if not targets:
        return 0
    replacements = 0

    def visit(op: BaseOP, prefix: str):
        nonlocal replacements
        if isinstance(op, OPList):
            children = ((str(i), child) for i, child in enumerate(op.op_list))
        else:
            children = ((key, child) for key, child in vars(op).items() if not key.startswith("_"))
        for key, child in list(children):
            name = f"{prefix}.{key}" if prefix else key
            if isinstance(child, KimiLinear) and key not in _NATIVE_DENSE:
                if any(_matches(p, name) for p in ignore):
                    continue
                matches = sum(any(_matches(p, name) for p in group) for group in targets)
                if matches > 1:
                    raise ValueError(f"Multiple Kimi quantization groups match {name}")
                if matches:
                    if name.endswith(".self_attn.kv_b_proj") or name == "self_attn.kv_b_proj":
                        raise NotImplementedError(
                            "Packed Kimi MLA kv_b_proj is unsupported: latent attention "
                            "requires its dense weight; keep it ignored in quantization_config"
                        )
                    # Preserve custom forward methods (e.g. vocab TP reduction)
                    # rather than silently discarding them during replacement.
                    if type(child) is not KimiLinear:
                        raise NotImplementedError(
                            f"Packed execution for {type(child).__name__} is unsupported"
                        )
                    weight = child.weight
                    replacement = KimiPackedLinear(
                        weight.shape[1],
                        weight.shape[0],
                        child.bias is not None,
                        device=weight.device,
                        dtype=weight.dtype,
                        chunk_rows=chunk_rows,
                    )
                    replacement.bias = child.bias
                    if isinstance(op, OPList):
                        op.op_list[int(key)] = replacement
                    else:
                        setattr(op, key, replacement)
                    replacements += 1
            elif isinstance(child, BaseOP):
                visit(child, name)

    visit(model, "")
    return replacements
