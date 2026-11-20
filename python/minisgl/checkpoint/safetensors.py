"""Data-only safetensors index validation and bounded shard iteration."""

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from minisgl.distributed import DistributedInfo

import safetensors


def index_entries(folder: Path) -> dict[str, str] | None:
    index_file = folder / "model.safetensors.index.json"
    if not index_file.exists():
        return None
    entries = json.loads(index_file.read_text()).get("weight_map")
    if not isinstance(entries, dict):
        raise ValueError("Checkpoint index requires a weight_map mapping")
    for name, filename in entries.items():
        if not isinstance(name, str) or not isinstance(filename, str):
            raise ValueError("Checkpoint index names and filenames must be strings")
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


def checkpoint_index(folder: Path, normalize) -> dict[str, Path]:
    entries = index_entries(folder)
    if entries is not None:
        result = {}
        files_by_name = {}
        for name, filename in entries.items():
            if normalize(name) is None:
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
                if normalize(name) is None:
                    continue
                if name in result:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                result[name] = file
    return result


def _prefetch_checkpoint_file(path, cancel):
    block_size = int(os.getenv("SGLANG_PREFETCH_BLOCK_SIZE_MB", "16")) << 20
    if block_size <= 0:
        raise ValueError("Checkpoint prefetch block size must be positive")
    with open(path, "rb") as shard:
        while not cancel.is_set() and shard.read(block_size):
            pass


@contextmanager
def _prefetch_shards(files, num_threads: int, tp_info: "DistributedInfo"):
    """Stage shared files without creating full-shard tensor copies.

    Mini's workers are node-local. Pass their coordinates explicitly rather than
    initializing the SDK's process-global distributed groups just for file I/O.
    """
    if num_threads < 1:
        raise ValueError("Checkpoint prefetch threads must be positive")
    if tp_info is None:
        raise ValueError("Checkpoint prefetch requires explicit local rank coordinates")
    files = sorted(set(files))[tp_info.rank :: tp_info.size]
    if not files:
        yield
        return
    cancel = threading.Event()
    executor = ThreadPoolExecutor(
        max_workers=min(num_threads, len(files)), thread_name_prefix="checkpoint-prefetch"
    )
    futures = []
    try:
        futures = [executor.submit(_prefetch_checkpoint_file, str(file), cancel) for file in files]
        yield
        for future in futures:
            future.result()
    finally:
        cancel.set()
        executor.shutdown(wait=True, cancel_futures=True)


def shard_readers(
    index,
    *,
    prefetch: bool = False,
    prefetch_num_threads: int = 4,
    tp_info: "DistributedInfo | None" = None,
    prefetch_files=None,
):
    # Parse/map each shard once rather than reopening it for every expert.
    by_file = {}
    for name, file in index.items():
        by_file.setdefault(file, []).append(name)
    staging = (
        _prefetch_shards(
            by_file if prefetch_files is None else prefetch_files,
            prefetch_num_threads,
            tp_info,
        )
        if prefetch
        else nullcontext()
    )
    with staging:
        for file, names in sorted(by_file.items()):
            with safetensors.safe_open(file, framework="pt", device="cpu") as reader:
                for name in sorted(names):
                    yield name, reader


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
