"""Streaming Kimi-K3 text loader with packed MXFP4 and logical TP slicing.

Packed bytes/scales may live in separate shards. Packed mode never expands the
checkpoint matrices, and TP slices are applied before tensor materialization.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

import safetensors
import torch
from minisgl.quantization.mxfp4 import decode_mxfp4

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def normalize_kimi_name(name: str, num_layers: int | None = None) -> str | None:
    """Keep native text names; exclude vision, projector and prediction heads."""
    if name.startswith("language_model.layers."):
        name = "language_model.model.layers." + name.removeprefix("language_model.layers.")
    name = name.removeprefix("language_model.").replace(".block_sparse_moe.", ".mlp.")
    components = name.split(".")
    if any(
        part
        in {
            "vision_tower",
            "mm_projector",
            "multi_modal_projector",
            "vision_model",
            "mtp",
            "mtp_layers",
            "nextn_predict_layers",
        }
        for part in components
    ):
        return None
    if any(part in {"inv_freq", "cos_cached", "sin_cached"} for part in components):
        return None
    layer = _LAYER.search(name)
    if layer is not None and num_layers is not None and int(layer.group(1)) >= num_layers:
        return None
    return name


def _index_entries(folder: Path) -> dict[str, str] | None:
    index_file = folder / "model.safetensors.index.json"
    if not index_file.exists():
        return None
    entries = json.loads(index_file.read_text()).get("weight_map")
    if not isinstance(entries, dict):
        raise ValueError("Kimi checkpoint index requires a weight_map mapping")
    for name, filename in entries.items():
        if not isinstance(name, str) or not isinstance(filename, str):
            raise ValueError("Kimi checkpoint index names and filenames must be strings")
        relative = Path(filename)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(char in filename for char in "\\*?[]")
            or not filename.endswith(".safetensors")
        ):
            raise ValueError(
                f"Checkpoint index points outside model folder or unsafe shard: {filename}"
            )
    return entries


def _checkpoint_index(folder: Path, num_layers: int | None = None) -> dict[str, Path]:
    entries = _index_entries(folder)
    if entries is not None:
        result = {}
        files_by_name = {}
        for name, filename in entries.items():
            if normalize_kimi_name(name, num_layers) is None:
                continue
            # HF snapshots legitimately symlink files into their blob cache.
            # Reject path traversal in the index, not those managed symlinks.
            if filename not in files_by_name:
                file = folder / filename
                if not file.is_file():
                    raise FileNotFoundError(file)
                files_by_name[filename] = file
            result[name] = files_by_name[filename]
        return result
    files = sorted(folder.glob("*.safetensors"))
    files = [p for p in files if p.name != "consolidated.safetensors"] or files
    if not files:
        raise FileNotFoundError(f"No safetensors checkpoint in {folder}")
    result = {}
    for file in files:
        with safetensors.safe_open(file, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if normalize_kimi_name(name, num_layers) is None:
                    continue
                if name in result:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                result[name] = file
    return result


def _shard_readers(index):
    # Parse/map each shard once rather than reopening it for every expert.
    by_file = {}
    for name, file in index.items():
        by_file.setdefault(file, []).append(name)
    for file, names in by_file.items():
        with safetensors.safe_open(file, framework="pt", device="cpu") as reader:
            for name in names:
                yield name, reader


def _validate_pairs(index, num_layers):
    logical_names = set()
    for raw_name in index:
        name = normalize_kimi_name(raw_name, num_layers)
        if name.endswith(".weight_scale"):
            partner = raw_name.removesuffix(".weight_scale") + ".weight_packed"
            if partner not in index:
                raise ValueError(f"Orphan or unsupported quantization scale: {raw_name}")
            continue
        if name.endswith((".weight_scale_inv", ".weight_zero_point", ".weight_global_scale")):
            raise NotImplementedError(f"Unsupported quantization auxiliary tensor: {raw_name}")
        if name.endswith(".weight_packed"):
            partner = raw_name.removesuffix(".weight_packed") + ".weight_scale"
            if partner not in index:
                raise ValueError(f"Missing MXFP4 scale for {raw_name}")
            name = name.removesuffix(".weight_packed") + ".weight"
        if name in logical_names:
            raise ValueError(f"Duplicate normalized checkpoint tensor: {name}")
        logical_names.add(name)


def _read_packed_pair(reader, raw_name, scales_reader, scale_name, spec, *, keep_packed):
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
        raise ValueError("Packed Kimi storage requires positive widths divisible by 32")
    row_slice, packed_slice, scale_slice = slice(None), slice(None), slice(None)
    if spec is not None:
        if spec.axis == 0:
            if not 0 <= spec.start < spec.stop <= rows:
                raise ValueError(f"Kimi TP output slice exceeds checkpoint shape: {raw_name}")
            row_slice = slice(spec.start, spec.stop)
        elif spec.axis == 1:
            if not 0 <= spec.start < spec.stop <= cols:
                raise ValueError(f"Kimi TP input slice exceeds checkpoint shape: {raw_name}")
            if spec.start % 32 or spec.stop % 32:
                raise ValueError("Kimi packed TP input boundaries must align to 32-element blocks")
            packed_slice = slice(spec.start // 2, spec.stop // 2)
            scale_slice = slice(spec.start // 32, spec.stop // 32)
        else:
            raise ValueError("Kimi packed TP only supports logical axes 0 and 1")
    weight = weight_view[row_slice, packed_slice]
    scale = (
        scale_view[row_slice, 0, scale_slice, 0] if scale_4d else scale_view[row_slice, scale_slice]
    )
    return weight.contiguous(), scale.contiguous()


def _read_dense(reader, raw_name, name, spec, config):
    view = reader.get_slice(raw_name)
    shape = view.get_shape()
    if view.get_dtype() not in ("F64", "F32", "F16", "BF16"):
        raise ValueError(f"Unexpected non-floating text weight {raw_name}: {view.get_dtype()}")
    if name.endswith(".self_attn.A_log"):
        # The public K3 checkpoint stores head_dim entries, although only the
        # first global num_heads entries are live. Slice local heads before read.
        legacy = len(shape) == 4 and shape[:2] == [1, 1] and shape[-1] == 1
        if len(shape) != 1 and not legacy:
            raise ValueError(f"Unsupported KDA A_log shape: {tuple(shape)}")
        count = shape[2] if legacy else shape[0]
        heads = (config.get("linear_attn_config") or {}).get("num_heads", count)
        if count < heads:
            raise ValueError(f"KDA A_log has fewer than {heads} heads")
        start, stop = (spec.start, spec.stop) if spec is not None else (0, heads)
        if not 0 <= start < stop <= heads:
            raise ValueError("Kimi TP A_log slice exceeds global head count")
        tensor = view[0, 0, start:stop, 0] if legacy else view[start:stop]
        return tensor.contiguous().float()
    if spec is None:
        return reader.get_tensor(raw_name)
    if spec.axis >= len(shape) or not 0 <= spec.start < spec.stop <= shape[spec.axis]:
        raise ValueError(f"Kimi TP slice exceeds checkpoint shape: {raw_name}")
    slices = [slice(None)] * len(shape)
    slices[spec.axis] = slice(spec.start, spec.stop)
    return view[tuple(slices)].contiguous()


def load_kimi_weight(
    model_path: str,
    device: torch.device | str,
    *,
    dtype: torch.dtype | None = None,
    num_layers: int | None = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    packed: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream native text weights, sharding before read/decode when applicable.

    With ``packed=True``, yield U8 ``weight_packed`` and ``weight_scale`` tensors
    separately and never create a dense copy of an MXFP4 matrix. ``packed=False``
    retains the legacy dense-decode behavior; ``dtype`` controls only decoded
    matrices, while native dense checkpoint tensors retain their original dtype.
    Remote downloads are data-only, revision-pinned, and index-filtered to shards
    containing requested text tensors (no vision-only or MTP-only shards).
    """
    from .kimi_config import KimiTextConfig
    from .kimi_parallel import shard_spec, validate_tp_config
    from .kimi_quant import quantization_rules

    if not 0 <= tp_rank < tp_size:
        raise ValueError("Kimi TP rank must be in [0, tp_size)")
    folder = Path(model_path)
    remote = not folder.is_dir()
    if remote:
        from huggingface_hub import snapshot_download

        # Fetch data-only metadata before any weights; pin both downloads to
        # the same snapshot. No checkpoint-provided Python is executed.
        folder = Path(
            snapshot_download(
                model_path, allow_patterns=["config.json", "model.safetensors.index.json"]
            )
        )
        if not (folder / "config.json").is_file():
            raise FileNotFoundError("Kimi checkpoint is missing config.json")
    config = {}
    config_file = folder / "config.json"
    if config_file.exists():
        config = json.loads(config_file.read_text())
        config = config.get("text_config", config)
        if num_layers is None:
            num_layers = config.get("num_hidden_layers")
        if dtype is None:
            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }.get(config.get("dtype", config.get("torch_dtype")), torch.bfloat16)
        quantization_rules(config)
    dtype = dtype or torch.bfloat16
    shard_config = KimiTextConfig(**config)
    if tp_size > 1:
        if not config:
            raise ValueError("Kimi tensor-parallel loading requires checkpoint config.json")
        validate_tp_config(shard_config, tp_size)
    if remote:
        entries = _index_entries(folder)
        patterns = (
            sorted(
                {
                    filename
                    for name, filename in entries.items()
                    if normalize_kimi_name(name, num_layers) is not None
                }
            )
            if entries is not None
            else ["*.safetensors"]
        )
        if patterns:
            folder = Path(
                snapshot_download(model_path, revision=folder.name, allow_patterns=patterns)
            )
    index = _checkpoint_index(folder, num_layers)
    _validate_pairs(index, num_layers)
    for raw_name, reader in _shard_readers(index):
        name = normalize_kimi_name(raw_name, num_layers)
        if name.endswith(".weight_scale"):
            continue
        logical_name = (
            name.removesuffix(".weight_packed") + ".weight"
            if name.endswith(".weight_packed")
            else name
        )
        spec = shard_spec(logical_name, shard_config, tp_rank, tp_size)
        if name.endswith(".weight_packed"):
            scale_name = raw_name.removesuffix(".weight_packed") + ".weight_scale"
            if index[scale_name] == index[raw_name]:
                tensor, scale = _read_packed_pair(
                    reader, raw_name, reader, scale_name, spec, keep_packed=packed
                )
            else:
                with safetensors.safe_open(index[scale_name], framework="pt", device="cpu") as sr:
                    tensor, scale = _read_packed_pair(
                        reader, raw_name, sr, scale_name, spec, keep_packed=packed
                    )
            if packed:
                yield name, tensor.to(device=device)
                del tensor
                yield name.removesuffix(".weight_packed") + ".weight_scale", scale.to(device=device)
                del scale
                continue
            tensor = decode_mxfp4(tensor, scale, dtype=dtype)
            del scale
            name = logical_name
        else:
            tensor = _read_dense(reader, raw_name, name, spec, config)
        yield name, tensor.to(device=device)
        del tensor
