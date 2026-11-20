"""Read local parameter slices and assemble the runtime weight stream."""

from __future__ import annotations

import json
import re
from contextlib import nullcontext
from pathlib import Path

import safetensors
import torch
from minisgl.distributed.layout import ParallelLayout
from minisgl.layers.weight import decode_mxfp4

from .safetensors import checkpoint_index, index_entries, read_mxfp4_pair, shard_readers

_EXPERT = re.compile(r"(?:^|\.)experts\.(\d+)\.")


class Checkpoint:
    """Interpret checkpoint tensors and assemble the parameters owned by one rank."""

    def __init__(self, config, layout):
        self.config = config
        self.layout = layout

    packed_weights = False

    def validate(self):
        """Validate configuration before opening any weight files."""

    def normalize(self, name):
        return name

    def owns(self, name):
        if self.layout.ep_size == 1:
            return True
        match = _EXPERT.search(name)
        if match is None:
            return True
        expert_id = int(match[1])
        if expert_id >= self.config.num_experts:
            raise ValueError("Checkpoint expert ID exceeds configured expert count")
        return expert_id in self.layout.expert_range(self.config.num_experts)

    def shard(self, name, shape):
        return None

    def read_dense(self, reader, raw_name, name, view, spec):
        if spec is None:
            return reader.get_tensor(raw_name)
        shape = view.get_shape()
        if spec.axis >= len(shape) or not 0 <= spec.start <= spec.stop <= shape[spec.axis]:
            raise ValueError(f"Weight slice exceeds checkpoint shape: {raw_name}")
        indices = [slice(None)] * len(shape)
        indices[spec.axis] = slice(spec.start, spec.stop)
        return view[tuple(indices)].contiguous()

    def assemble(self, weights):
        return weights


def _validate_pairs(index, normalize):
    logical_names = set()
    for raw_name in index:
        name = normalize(raw_name)
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


def _metadata_folder(model_path):
    folder = Path(model_path)
    if folder.is_dir():
        return folder
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_path, allow_patterns=["config.json", "model.safetensors.index.json"]
        )
    )


def _weight_folder(model_path, folder, normalize):
    if Path(model_path).is_dir():
        return folder
    from huggingface_hub import snapshot_download

    index = index_entries(folder)
    files = (
        sorted({file for key, file in index.items() if normalize(key) is not None})
        if index is not None
        else ["*.safetensors"]
    )
    return (
        Path(snapshot_download(model_path, revision=folder.name, allow_patterns=files))
        if files
        else folder
    )


def load_checkpoint(
    model_path,
    device,
    *,
    checkpoint_type,
    config=None,
    layout=None,
    packed=False,
    dtype=None,
    prefetch=False,
    prefetch_num_threads=4,
):
    """Read a local checkpoint or a pinned data-only Hub snapshot."""
    folder = _metadata_folder(model_path)
    layout = layout or ParallelLayout()
    if config is None:
        config_file = folder / "config.json"
        if not config_file.is_file() and layout.tensor.size > 1:
            raise ValueError("Parallel loading requires checkpoint config.json")
        if not config_file.is_file() and not Path(model_path).is_dir():
            raise FileNotFoundError("Remote checkpoint is missing config.json")
        metadata = json.loads(config_file.read_text()) if config_file.is_file() else {}
        config = checkpoint_type.from_metadata(metadata.get("text_config", metadata))
    checkpoint = checkpoint_type(config, layout)
    checkpoint.validate()
    if packed is None:
        packed = checkpoint.packed_weights
    dtype = dtype or getattr(checkpoint.config, "dtype", None) or torch.bfloat16
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    folder = _weight_folder(model_path, folder, checkpoint.normalize)
    index = checkpoint_index(folder, checkpoint.normalize)
    _validate_pairs(index, checkpoint.normalize)
    files = set(index.values()) if prefetch else None
    local = {key: file for key, file in index.items() if checkpoint.owns(checkpoint.normalize(key))}
    readers = shard_readers(
        local,
        prefetch=prefetch,
        prefetch_num_threads=prefetch_num_threads,
        tp_info=layout.tensor,
        prefetch_files=files,
    )
    for name, tensor in checkpoint.assemble(
        _read_tensors(readers, local, checkpoint, packed, dtype)
    ):
        yield name, tensor.to(device=device)


def _read_tensors(readers, index, checkpoint, packed, dtype):
    for raw_name, reader in readers:
        name = checkpoint.normalize(raw_name)
        if name.endswith(".weight_scale"):
            continue
        view = reader.get_slice(raw_name)
        if name.endswith(".weight_packed"):
            logical = name.removesuffix(".weight_packed") + ".weight"
            shape = view.get_shape()
            if len(shape) != 2:
                raise ValueError(f"Expected a 2-D MXFP4 packed tensor: {raw_name}")
            rows, byte_cols = shape
            spec = checkpoint.shard(logical, (rows, byte_cols * 2))
            scale = raw_name.removesuffix(".weight_packed") + ".weight_scale"
            other = (
                nullcontext(reader)
                if index[scale] == index[raw_name]
                else safetensors.safe_open(index[scale], framework="pt", device="cpu")
            )
            with other as scale_reader:
                weight, scales = read_mxfp4_pair(
                    reader, raw_name, scale_reader, scale, spec, keep_packed=packed
                )
            if packed:
                yield name, weight
                yield checkpoint.normalize(scale), scales
            else:
                yield logical, decode_mxfp4(weight, scales, dtype=dtype)
            del weight, scales
        else:
            spec = checkpoint.shard(name, view.get_shape())
            yield name, checkpoint.read_dense(reader, raw_name, name, view, spec)
