"""Offline packed K3 resident-storage audit against saved, verified headers.

Creates meta-device models at representative last ranks for TP=1/2/4/8. Never
opens safetensors, downloads weights, initializes a process group, or allocates
weight payloads. Byte counts cover model tensors only, excluding KV/KDA caches,
activations, allocator overhead, runtime metadata, and communication workspaces.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import torch
from audit_kimi_headers import (
    DTYPE_BYTES,
    MAX_HEADER,
    MAX_TOTAL,
    REVISION,
    sha256,
    shard_name,
    unique_object,
    validate_header,
    validate_manifest,
)
from minisgl.models.config import ModelConfig
from minisgl.models.kimi_config import KimiK3Config
from minisgl.models.kimi_k3 import KimiK3LinearForCausalLM
from minisgl.models.kimi_parallel import shard_spec
from minisgl.models.kimi_weight import normalize_kimi_name

_DTYPE_NAMES = {
    torch.uint8: "U8",
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
}


def verified_headers(config_path, index_path, evidence):
    """Read JSON metadata only, validating pinned source and every shard hash."""
    config_data, index_data = config_path.read_bytes(), index_path.read_bytes()
    manifest = json.loads((evidence / "manifest.json").read_text(), object_pairs_hook=unique_object)
    validate_manifest(manifest)
    for label, data in (("config", config_data), ("index", index_data)):
        proof = manifest["source_metadata"][label]
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if (
            manifest[label + "_sha256"] != sha256(data)
            or proof["bytes"] != len(data)
            or proof["git_blob_sha1"] != blob
        ):
            raise ValueError(f"{label} source metadata does not match local bytes")
    index_doc = json.loads(index_data, object_pairs_hook=unique_object)
    index = index_doc["weight_map"]
    records = {record["shard"]: record for record in manifest["shards"]}
    if len(records) != len(manifest["shards"]) or set(records) != set(index.values()):
        raise ValueError("Saved shard set differs from checkpoint index")
    tensors = {}
    total = 0
    for shard, record in records.items():
        path = evidence / (shard_name(shard) + ".header.json")
        size = path.stat().st_size
        total += size
        if size > MAX_HEADER or total > MAX_TOTAL:
            raise ValueError("Offline header byte budget exceeded")
        data = path.read_bytes()
        if len(data) != record["header_bytes"] or sha256(data) != record["header_sha256"]:
            raise ValueError(f"Header size or hash mismatch: {shard}")
        header = json.loads(data, object_pairs_hook=unique_object)
        errors = validate_header(header, record["file_bytes"], record["header_bytes"])
        if errors:
            raise ValueError(f"Invalid header {shard}: {errors}")
        for name, value in header.items():
            if name == "__metadata__":
                continue
            if name in tensors or index.get(name) != shard:
                raise ValueError(f"Duplicate tensor or index/header disagreement: {name}")
            tensors[name] = (tuple(value["shape"]), value["dtype"])
    if set(tensors) != set(index):
        raise ValueError("Index/header tensor set differs")
    checkpoint_bytes = sum(r["file_bytes"] - 8 - r["header_bytes"] for r in records.values())
    if index_doc.get("metadata", {}).get("total_size", checkpoint_bytes) != checkpoint_bytes:
        raise ValueError("Index/header total checkpoint bytes differ")
    return tensors, manifest


def sliced_descriptor(name, shape, dtype, config, rank, size):
    """Physical packed layout resulting from the loader's logical TP slice."""
    shape = list(shape)
    is_packed = name.endswith(".weight_packed")
    is_scale = name.endswith(".weight_scale")
    if is_scale:
        if len(shape) == 4 and shape[1] == shape[3] == 1:
            shape = [shape[0], shape[2]]
        if len(shape) != 2 or dtype != "U8":
            raise ValueError(f"Invalid packed scale descriptor: {name}")
    if is_packed and (len(shape) != 2 or dtype != "U8" or shape[1] % 16):
        raise ValueError(f"Invalid MXFP4 descriptor: {name}")
    if name.endswith(".self_attn.A_log"):
        if len(shape) == 4 and shape[:2] == [1, 1] and shape[-1] == 1:
            shape = [shape[2]]
        heads = config.kimi.linear_attn_config["num_heads"]
        if len(shape) != 1 or shape[0] < heads:
            raise ValueError(f"Invalid A_log descriptor: {name}")
        shape = [heads]
        dtype = "F32"
    logical_name = name.rsplit(".", 1)[0] + ".weight" if is_packed or is_scale else name
    spec = shard_spec(logical_name, config.kimi, rank, size)
    if spec is not None:
        divisor = (2 if is_packed else 32 if is_scale else 1) if spec.axis == 1 else 1
        if (is_packed or is_scale) and spec.axis == 1 and (spec.start % 32 or spec.stop % 32):
            raise ValueError(f"Packed TP slice is not scale-block aligned: {name}")
        if spec.axis >= len(shape) or not 0 <= spec.start < spec.stop <= shape[spec.axis] * divisor:
            raise ValueError(f"TP slice exceeds header shape: {name}")
        shape[spec.axis] = spec.length // divisor
    return tuple(shape), dtype, spec is not None


def audit_rank(config, tensors, rank, size):
    old_dtype = torch.get_default_dtype()
    dtype_name = getattr(config.kimi, "dtype", None) or getattr(config.kimi, "torch_dtype", None)
    dtype = (
        dtype_name
        if isinstance(dtype_name, torch.dtype)
        else {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(dtype_name, torch.bfloat16)
    )
    compute_element_bytes = DTYPE_BYTES[_DTYPE_NAMES[dtype]]
    try:
        torch.set_default_dtype(dtype)
        with (
            patch("minisgl.distributed.parallel.tp_info", return_value=(rank, size)),
            torch.device("meta"),
        ):
            model = KimiK3LinearForCausalLM(config)
        expected = {
            name: (tuple(t.shape), _DTYPE_NAMES[t.dtype]) for name, t in model.state_dict().items()
        }
        assert all(t.is_meta for t in model.state_dict().values())
        del model
        gc.collect()
    finally:
        torch.set_default_dtype(old_dtype)
    actual, errors = set(), []
    floating_conversions = []
    loaded_header_bytes = 0
    categories = Counter()
    dtypes = Counter()
    packed_matrices = 0
    dequantized_bytes = 0
    largest_dense_chunk_bytes = 0
    native_dense_roles = Counter()
    role_names = {
        "routed_expert_down_proj",
        "routed_expert_up_proj",
        "self_attention_res_proj",
        "mlp_res_proj",
        "output_attn_res_proj",
        "gate",
    }
    for raw_name, (shape, dtype) in tensors.items():
        name = normalize_kimi_name(raw_name, config.num_layers)
        if name is None:
            continue
        if name in actual:
            errors.append(f"Duplicate canonical key: {name}")
        actual.add(name)
        try:
            shape, dtype, sharded = sliced_descriptor(name, shape, dtype, config, rank, size)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        runtime = expected.get(name)
        if runtime != (shape, dtype):
            if (
                runtime is not None
                and runtime[0] == shape
                and dtype in {"F32", "BF16", "F16"}
                and runtime[1] in {"F32", "BF16", "F16"}
            ):
                floating_conversions.append(
                    {"name": name, "checkpoint": dtype, "runtime": runtime[1]}
                )
            else:
                errors.append(f"{name}: runtime={runtime} checkpoint={(shape, dtype)}")
        count = math.prod(shape)
        loaded_header_bytes += count * DTYPE_BYTES[dtype]
        if runtime is not None:
            dtype = runtime[1]
        nbytes = count * DTYPE_BYTES[dtype]
        categories["sharded_bytes" if sharded else "replicated_bytes"] += nbytes
        dtypes[dtype] += nbytes
        if name.endswith(".weight_packed"):
            categories["packed_weight_bytes"] += nbytes
            packed_matrices += 1
            element_bytes = compute_element_bytes
            dequantized_bytes += count * 2 * element_bytes
            input_width = shape[1] * 2
            chunk_rows = min(shape[0], 128, max(1, (1 << 20) // input_width))
            largest_dense_chunk_bytes = max(
                largest_dense_chunk_bytes, chunk_rows * input_width * element_bytes
            )
        elif name.endswith(".weight_scale"):
            categories["packed_scale_bytes"] += nbytes
        else:
            categories["native_dense_bytes"] += nbytes
            dequantized_bytes += nbytes
            if name.endswith(".weight") and name.rsplit(".", 2)[-2] in role_names:
                native_dense_roles[name.rsplit(".", 2)[-2]] += 1
    errors.extend(f"Missing runtime key: {name}" for name in expected.keys() - actual)
    storage = sum(dtypes.values())
    return {
        "tp_size": size,
        "representative_rank": rank,
        "runtime_keys": len(expected),
        "mapped_header_keys": len(actual),
        "packed_matrices": packed_matrices,
        "resident_model_bytes": storage,
        "resident_model_gib": storage / 2**30,
        "loaded_checkpoint_bytes_before_floating_casts": loaded_header_bytes,
        "floating_dtype_conversions": floating_conversions,
        "all_ranks_resident_model_bytes": storage * size,
        "equivalent_dequantized_model_bytes": dequantized_bytes,
        "largest_dense_decode_chunk_bytes": largest_dense_chunk_bytes,
        "bytes_by_storage": dict(categories),
        "bytes_by_dtype": dict(dtypes),
        "native_dense_role_counts": dict(native_dense_roles),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    tensors, manifest = verified_headers(args.config, args.index, args.evidence)
    config = ModelConfig.from_hf(KimiK3Config(**json.loads(args.config.read_text())))
    results = []
    for size in args.tp_sizes:
        result = audit_rank(config, tensors, size - 1, size)
        results.append(result)
        print(
            json.dumps(
                {
                    key: result[key]
                    for key in (
                        "tp_size",
                        "representative_rank",
                        "runtime_keys",
                        "packed_matrices",
                        "resident_model_bytes",
                        "resident_model_gib",
                        "errors",
                    )
                }
            ),
            flush=True,
        )
    report = {
        "repository": manifest["repository"],
        "revision": REVISION,
        "config_sha256": manifest["config_sha256"],
        "index_sha256": manifest["index_sha256"],
        "shards": len(manifest["shards"]),
        "metadata_only": True,
        "network_requests": 0,
        "tensor_payload_bytes_read": 0,
        "rank_coverage": "Representative last rank per TP size; equal contiguous partitions have equal storage on all ranks.",
        "capacity_scope": "Model tensor storage only. Excludes caches, activations, allocator overhead, runtime metadata, communication workspaces and decode temporary intermediates.",
        "execution_scope": "Unfused eager packed reference path; no throughput or full-model GPU execution claim.",
        "native_dense_exceptions": "Router, latent-space down/up projections and attention-residual score projections stay dense under the upstream K3 architecture; every one is verified against official headers.",
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if any(result["errors"] for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
