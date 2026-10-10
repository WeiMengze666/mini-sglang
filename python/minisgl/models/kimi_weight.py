"""Streaming, unfused Kimi-K3 text checkpoint loader (TP=1).

MXFP4 is decoded to dense weights, not executed as low-bit GEMMs. The dense
model must fit in memory. Metadata is indexed first so a packed matrix and its
scale may live in different shards without buffering a model's worth of data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator

import safetensors
import torch

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


def _checkpoint_index(folder: Path) -> dict[str, Path]:
    index_file = folder / "model.safetensors.index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text())["weight_map"]
        result = {}
        files_by_name = {}
        for name, filename in index.items():
            # HF snapshots legitimately symlink files into their blob cache.
            # Reject path traversal in the index, not those managed symlinks.
            relative = Path(filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Checkpoint index points outside model folder: {filename}")
            if filename not in files_by_name:
                file = folder / relative
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


def load_kimi_weight(
    model_path: str,
    device: torch.device | str,
    *,
    dtype: torch.dtype | None = None,
    num_layers: int | None = None,
    tp_size: int = 1,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield native-name text weights; retain native dense tensor dtypes.

    ``dtype`` specifies decoded MXFP4 storage, defaulting to config dtype/BF16.
    No projection fusion, expert stacking or unverified tensor-parallel sharding.
    """
    if tp_size != 1:
        raise NotImplementedError("Kimi-K3 eager checkpoint loading supports TP=1 only")
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
        quant = config.get("quantization_config")
        if quant and quant.get("format") not in (None, "mxfp4-pack-quantized"):
            raise NotImplementedError(
                f"Unsupported Kimi quantization format: {quant.get('format')}"
            )
    dtype = dtype or torch.bfloat16
    if remote:
        folder = Path(
            snapshot_download(model_path, revision=folder.name, allow_patterns=["*.safetensors"])
        )
    index = _checkpoint_index(folder)
    emitted: set[str] = set()
    for raw_name, reader in _shard_readers(index):
        name = normalize_kimi_name(raw_name, num_layers)
        if name is None:
            continue
        if name.endswith(".weight_scale"):
            packed_name = raw_name.removesuffix(".weight_scale") + ".weight_packed"
            if packed_name not in index:
                raise ValueError(f"Orphan or unsupported quantization scale: {raw_name}")
            continue
        if name.endswith((".weight_scale_inv", ".weight_zero_point", ".weight_global_scale")):
            raise NotImplementedError(f"Unsupported quantization auxiliary tensor: {raw_name}")
        tensor = reader.get_tensor(raw_name)
        if name.endswith(".weight_packed"):
            scale_name = raw_name.removesuffix(".weight_packed") + ".weight_scale"
            if scale_name not in index:
                raise ValueError(f"Missing MXFP4 scale for {raw_name}")
            if index[scale_name] == index[raw_name]:
                scale = reader.get_tensor(scale_name)
            else:
                with safetensors.safe_open(
                    index[scale_name], framework="pt", device="cpu"
                ) as scales_reader:
                    scale = scales_reader.get_tensor(scale_name)
            tensor = decode_mxfp4(tensor, scale, dtype=dtype)
            del scale
            name = name.removesuffix(".weight_packed") + ".weight"
        elif not tensor.is_floating_point():
            raise ValueError(f"Unexpected non-floating text weight {raw_name}: {tensor.dtype}")
        if name.endswith(".self_attn.A_log"):
            # Pinned K3 reference narrows [head_dim] or legacy [1,1,H,1]
            # checkpoint storage to the first num_heads entries at TP=1.
            if tensor.ndim == 4 and tensor.shape[:2] == (1, 1) and tensor.shape[-1] == 1:
                tensor = tensor.reshape(-1)
            if tensor.ndim != 1:
                raise ValueError(f"Unsupported KDA A_log shape: {tuple(tensor.shape)}")
            heads = (config.get("linear_attn_config") or {}).get("num_heads")
            if heads is not None:
                if tensor.numel() < heads:
                    raise ValueError(f"KDA A_log has fewer than {heads} heads")
                tensor = tensor[:heads].contiguous()
            tensor = tensor.float()
        if name in emitted:
            raise ValueError(f"Duplicate normalized checkpoint tensor: {name}")
        emitted.add(name)
        yield name, tensor.to(device=device)
