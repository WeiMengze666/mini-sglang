"""Static compressed-tensors MXFP4 selection and operator installation."""

from __future__ import annotations

import re
from collections.abc import Mapping

from minisgl.layers.base import BaseOP, OPList
from minisgl.layers.linear import Linear

from .mxfp4 import MXFP4Linear


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
            if isinstance(child, Linear) and not (exclude and exclude(name, child)):
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
                    replacement = MXFP4Linear(
                        weight.shape[1],
                        weight.shape[0],
                        child.bias is not None,
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
