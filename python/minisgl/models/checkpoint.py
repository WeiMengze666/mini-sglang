"""Data-only safetensors index validation and bounded shard iteration."""

import json
from pathlib import Path

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


def shard_readers(index):
    # Parse/map each shard once rather than reopening it for every expert.
    by_file = {}
    for name, file in index.items():
        by_file.setdefault(file, []).append(name)
    for file, names in by_file.items():
        with safetensors.safe_open(file, framework="pt", device="cpu") as reader:
            for name in names:
                yield name, reader
