"""Audit Kimi tensor metadata without reading tensor values.

Offline by default. --fetch permits only two strict HTTP byte ranges per shard:
its 8-byte length prefix and its JSON header. No safetensors reader is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import quote, urljoin, urlsplit

REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
MAX_HEADER = 8 * 1024 * 1024
MAX_TOTAL = 128 * 1024 * 1024
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
TRUSTED_HOSTS = {"huggingface.co", "us.aws.cdn.hf.co"}
FLOAT_DTYPES = {"F16", "BF16", "F32", "F64"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def shard_name(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Invalid shard path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError("Shard must be a plain filename without traversal")
    if not re.fullmatch(r"[A-Za-z0-9._-]+\.safetensors", value):
        raise ValueError("Invalid shard filename")
    return value


class Budget:
    def __init__(self, maximum=MAX_TOTAL):
        self.maximum = maximum
        self.reserved = 0
        self.read_bytes = 0
        self.lock = threading.Lock()

    def reserve(self, amount):
        with self.lock:
            if self.reserved + amount > self.maximum:
                raise ValueError("Total metadata byte budget exceeded")
            self.reserved += amount

    def consumed(self, amount):
        with self.lock:
            self.read_bytes += amount


def read_range(session, url, start, end, budget):
    """Never consume a 200 or redirect body; refuse non-exact range responses."""
    expected = end - start + 1
    if start < 0 or expected <= 0 or expected > MAX_HEADER:
        raise ValueError("Range exceeds metadata limits")
    budget.reserve(expected)
    for redirects in range(7):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in TRUSTED_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in (None, 443)
        ):
            raise ValueError("Only trusted HTTPS Hugging Face metadata hosts are permitted")
        try:
            response = session.get(
                url,
                headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
                stream=True,
                allow_redirects=False,
                timeout=(15, 90),
            )
        except Exception as exc:
            # Requests exceptions include signed redirect URLs; never surface those.
            raise ValueError(f"Metadata transport failed ({type(exc).__name__})") from None
        try:
            if response.status_code in {301, 302, 303, 307, 308}:
                if redirects == 6 or not response.headers.get("Location"):
                    raise ValueError("Metadata redirect limit or missing location")
                url = urljoin(url, response.headers["Location"])
                continue
            if response.status_code != 206:
                raise ValueError(f"Expected partial response 206, got {response.status_code}")
            match = re.fullmatch(
                r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", "")
            )
            if not match or tuple(map(int, match.group(1, 2))) != (start, end):
                raise ValueError("Content-Range does not exactly match metadata range")
            total = int(match.group(3))
            if total <= end:
                raise ValueError("Invalid total shard size")
            length = response.headers.get("Content-Length")
            if length is None or not length.isdigit() or int(length) != expected:
                raise ValueError("Content-Length does not match metadata range")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("Compressed metadata responses are prohibited")
            response.raw.decode_content = False
            body = response.raw.read(expected)
            budget.consumed(len(body))
            if len(body) != expected:
                raise ValueError("Truncated or oversized metadata body")
            return body, total
        finally:
            response.close()
    raise AssertionError("Unreachable redirect loop")


def fetch_header(shard, directory, budget, revision=REVISION):
    shard_name(shard)
    url = f"https://huggingface.co/moonshotai/Kimi-K3/resolve/{revision}/{quote(shard)}"
    with public_session() as session:
        prefix, total = read_range(session, url, 0, 7, budget)
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= MAX_HEADER or 8 + length > total:
            raise ValueError("Invalid or oversized safetensors header length")
        data, second_total = read_range(session, url, 8, 8 + length - 1, budget)
        if total != second_total:
            raise ValueError("Shard size changed between requests")
    header = json.loads(data, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError("Safetensors header must be an object")
    (directory / (shard + ".header.json")).write_bytes(data)
    return {
        "shard": shard,
        "header_bytes": length,
        "file_bytes": total,
        "header_sha256": sha256(data),
        "prefix_sha256": sha256(prefix),
        "body_bytes": 8 + length,
    }


def validate_header(header, file_bytes, header_bytes):
    """Validate complete offset coverage, shapes and dtype-derived byte counts."""
    errors, spans = [], []
    payload_bytes = file_bytes - 8 - header_bytes
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            errors.append(f"{name}: metadata is not an object")
            continue
        shape, dtype, offsets = (metadata.get(k) for k in ("shape", "dtype", "data_offsets"))
        if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
            errors.append(f"{name}: invalid shape")
            continue
        if dtype not in DTYPE_BYTES:
            errors.append(f"{name}: unsupported dtype {dtype}")
            continue
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(n) is not int for n in offsets)
        ):
            errors.append(f"{name}: invalid offsets")
            continue
        lo, hi = offsets
        if not 0 <= lo <= hi <= payload_bytes:
            errors.append(f"{name}: offsets outside data region")
        if hi - lo != math.prod(shape) * DTYPE_BYTES[dtype]:
            errors.append(f"{name}: offset byte count disagrees with shape/dtype")
        spans.append((lo, hi, name))
    cursor = 0
    for lo, hi, name in sorted(spans):
        if lo != cursor:
            errors.append(f"{name}: overlapping offsets or gap before {lo} (expected {cursor})")
        cursor = hi
    if cursor != payload_bytes:
        errors.append(f"Data region coverage {cursor} != {payload_bytes}")
    return errors


def audit_text(tensors, expected, num_layers, normalize, decoded_dtype="BF16"):
    """expected maps normalized keys to (shape, safetensors dtype)."""
    errors, mapped, narrowings, dtype_mismatches = [], set(), [], []
    raw_hist, text_hist, loaded_hist = Counter(), Counter(), Counter()
    packed = 0
    for raw, metadata in tensors.items():
        raw_hist[metadata["dtype"]] += 1
        name = normalize(raw, num_layers)
        if name is None:
            continue
        text_hist[metadata["dtype"]] += 1
        shape, dtype = metadata["shape"], metadata["dtype"]
        if name.endswith(".weight_scale"):
            if raw.removesuffix(".weight_scale") + ".weight_packed" not in tensors:
                errors.append(f"{raw}: orphan scale")
            continue
        if name.endswith(".weight_packed"):
            packed += 1
            scale_name = raw.removesuffix(".weight_packed") + ".weight_scale"
            scale = tensors.get(scale_name)
            if len(shape) != 2 or dtype != "U8":
                errors.append(f"{raw}: packed storage must be U8 [out,in/2]")
                continue
            rows, cols = shape[0], shape[1] * 2
            if cols % 32:
                errors.append(f"{raw}: logical input columns not divisible by 32")
            scale_shape = scale["shape"] if scale else None
            if scale_shape == [rows, 1, cols // 32, 1]:
                scale_shape = [rows, cols // 32]
            if not scale or scale["dtype"] != "U8" or scale_shape != [rows, cols // 32]:
                errors.append(f"{raw}: scale must be E8M0 U8 [out,in/32] (or singleton 4-D)")
            shape, dtype = [rows, cols], decoded_dtype
            name = name.removesuffix(".weight_packed") + ".weight"
        elif dtype not in FLOAT_DTYPES:
            errors.append(f"{raw}: dense text tensor must be floating point, got {dtype}")
        if name in mapped:
            errors.append(f"{raw}: duplicate normalized key {name}")
        mapped.add(name)
        if name not in expected:
            errors.append(f"{raw}: unexpected runtime key {name}")
            continue
        want_shape, want_dtype = expected[name]
        if name.endswith(".self_attn.A_log"):
            original_shape = shape
            if len(shape) == 4 and shape[:2] == [1, 1] and shape[-1] == 1:
                shape = [shape[2]]
            if len(shape) == 1 and len(want_shape) == 1 and shape[0] >= want_shape[0]:
                if shape != list(want_shape):
                    narrowings.append(
                        {
                            "key": raw,
                            "stored_shape": original_shape,
                            "loaded_shape": list(want_shape),
                        }
                    )
                shape = list(want_shape)
            dtype = "F32"
        if list(shape) != list(want_shape):
            errors.append(f"{raw}: loaded shape {shape} != declared {list(want_shape)}")
        if dtype != want_dtype:
            dtype_mismatches.append(
                {"key": raw, "loaded_dtype": dtype, "declared_dtype": want_dtype}
            )

        loaded_hist[want_dtype] += 1
    missing = sorted(set(expected) - mapped)
    errors.extend(f"Missing runtime key: {name}" for name in missing)
    return {
        "raw_tensors": len(tensors),
        "mapped_tensors": len(mapped),
        "declared_tensors": len(expected),
        "packed_matrices": packed,
        "raw_dtype_histogram": dict(raw_hist),
        "text_storage_dtype_histogram": dict(text_hist),
        "loaded_dtype_histogram": dict(loaded_hist),
        "a_log_narrowings": narrowings,
        "engine_dtype_casts": dtype_mismatches,
        "engine_dtype_cast_histogram": dict(
            Counter(
                f"{item['loaded_dtype']}->{item['declared_dtype']}" for item in dtype_mismatches
            )
        ),
        "errors": errors,
    }


def declared_parameters(config_path):
    import torch
    from minisgl.models.config import ModelConfig
    from minisgl.models.kimi_config import KimiK3Config
    from minisgl.models.kimi_k3 import KimiK3LinearForCausalLM

    raw = json.loads(config_path.read_text())
    text = raw.get("text_config", raw)
    dtype = text.get("dtype", text.get("torch_dtype", raw.get("dtype", "bfloat16")))
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype
    ]
    config = ModelConfig.from_hf(KimiK3Config(**raw))
    # Logical dequantized shape audit; packed resident storage is audited separately.
    config.kimi.quantization_config = None
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch_dtype)
        with torch.device("meta"):
            state = KimiK3LinearForCausalLM(config).state_dict()
        names = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32"}
        expected = {name: (tuple(t.shape), names[t.dtype]) for name, t in state.items()}
    finally:
        torch.set_default_dtype(old_dtype)
    return expected, config.num_layers, names[torch_dtype]


def public_session():
    import requests

    class NoAuth(requests.auth.AuthBase):
        def __call__(self, request):
            request.headers.pop("Authorization", None)
            return request

    class MetadataSession(requests.Session):
        def send(self, request, **kwargs):
            # Session.send prepares response._next even with allow_redirects=False;
            # that path consumes redirect bodies. Dispatch directly to the adapter.
            # Session.request has already merged the required environment proxies.
            if kwargs.pop("allow_redirects", False):
                raise ValueError("Automatic redirects are prohibited")
            kwargs.setdefault("stream", True)
            kwargs.setdefault("verify", self.verify)
            kwargs.setdefault("cert", self.cert)
            kwargs.setdefault("proxies", {})
            request.headers.pop("Authorization", None)
            return self.get_adapter(request.url).send(request, **kwargs)

    session = MetadataSession()
    session.auth = NoAuth()  # Suppress implicit netrc; retain required proxy environment.
    return session


def verify_metadata_sources(config, index):
    """Verify local bytes against official pinned-revision content ETags via HEAD."""
    proof = {}
    for label, path, filename in (
        ("config", config, "config.json"),
        ("index", index, "model.safetensors.index.json"),
    ):
        data = path.read_bytes()
        url = f"https://huggingface.co/moonshotai/Kimi-K3/resolve/{REVISION}/{filename}"
        with (
            public_session() as session,
            session.head(url, allow_redirects=False, timeout=(15, 90)) as response,
        ):
            headers = response.headers
            if response.status_code not in {200, 302, 307}:
                raise ValueError("Pinned metadata HEAD failed")
            commit = headers.get("X-Repo-Commit")
            etag = headers.get("X-Linked-ETag", headers.get("ETag", "")).strip('"')
            git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            if commit != REVISION or etag not in {sha256(data), git_blob}:
                raise ValueError(f"Pinned {label} metadata content does not match local bytes")
            proof[label] = {
                "url": url,
                "revision": commit,
                "etag": etag,
                "sha256": sha256(data),
                "git_blob_sha1": git_blob,
                "bytes": len(data),
                "http_body_bytes_read": 0,
            }
    return proof


def validate_manifest(manifest):
    if (
        manifest.get("repository") != "moonshotai/Kimi-K3"
        or manifest.get("revision") != REVISION
        or manifest.get("metadata_only") is not True
        or type(manifest.get("tensor_payload_bytes_requested")) is not int
        or manifest["tensor_payload_bytes_requested"] != 0
    ):
        raise ValueError("Manifest does not establish pinned metadata-only provenance")
    source_metadata = manifest.get("source_metadata")
    if not isinstance(source_metadata, dict):
        raise ValueError("Missing pinned source metadata proof")
    for label, filename in (("config", "config.json"), ("index", "model.safetensors.index.json")):
        proof = source_metadata.get(label, {})
        expected_url = f"https://huggingface.co/moonshotai/Kimi-K3/resolve/{REVISION}/{filename}"
        digest = manifest.get(label + "_sha256", "")
        git_blob = proof.get("git_blob_sha1", "")
        if (
            not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not re.fullmatch(r"[0-9a-f]{40}", git_blob)
            or proof.get("url") != expected_url
            or proof.get("revision") != REVISION
            or proof.get("sha256") != digest
            or proof.get("etag") not in {digest, git_blob}
            or type(proof.get("bytes")) is not int
            or proof["bytes"] < 0
            or type(proof.get("http_body_bytes_read")) is not int
            or proof["http_body_bytes_read"] != 0
        ):
            raise ValueError(f"Invalid pinned {label} source metadata proof")
    if not isinstance(manifest.get("http_read_policy"), str) or not manifest["http_read_policy"]:
        raise ValueError("Missing historical HTTP read policy")
    total = 0
    for record in manifest["shards"]:
        shard_name(record["shard"])
        header, file_size = record.get("header_bytes"), record.get("file_bytes")
        if (
            type(header) is not int
            or not 2 <= header <= MAX_HEADER
            or type(file_size) is not int
            or file_size < 8 + header
        ):
            raise ValueError("Invalid shard sizes in manifest")
        total += 8 + header
    body_bytes = manifest.get("http_body_bytes_read")
    if (
        type(body_bytes) is not int
        or not 0 <= body_bytes <= MAX_TOTAL
        or total > MAX_TOTAL
        or body_bytes < total
    ):
        raise ValueError("Invalid metadata body count in manifest")
    retry_bytes = manifest.get("retry_body_bytes_read", 0)
    history = manifest.get("retry_history", [])
    if type(retry_bytes) is not int or retry_bytes < 0 or not isinstance(history, list):
        raise ValueError("Invalid retry accounting")
    accounted_retry_bytes = 0
    for attempt in history:
        count = attempt.get("successful_prefix_bytes")
        failures = attempt.get("failed_shards")
        if (
            attempt.get("exception") != "ReadTimeout"
            or attempt.get("stage") != "before_response"
            or attempt.get("unknown_partial_body") is not False
            or not isinstance(failures, list)
            or not failures
            or type(count) is not int
            or count < 0
            or count % 8
            or count > 8 * len(failures)
        ):
            raise ValueError("Unverified or unmetered retry history")
        for shard in failures:
            shard_name(shard)
        accounted_retry_bytes += count
    if retry_bytes != accounted_retry_bytes:
        raise ValueError("Retry history byte count mismatch")
    if not manifest.get("fetch_errors") and body_bytes != total + retry_bytes:
        raise ValueError("Metadata body count does not equal saved metadata plus retries")
    if manifest.get("fetch_errors"):
        raise ValueError("Incomplete fetch manifest; cannot assert metadata-only audit completion")
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("index", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument(
        "--fetch", action="store_true", help="Fetch headers only, never weight data"
    )
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    index_document = json.loads(args.index.read_text(), object_pairs_hook=unique_object)
    index = index_document["weight_map"]
    shards = sorted({shard_name(shard) for shard in index.values()})
    manifest_path = args.evidence / "manifest.json"
    errors = []
    if args.fetch:
        source_metadata = verify_metadata_sources(args.config, args.index)
        budget = Budget()
        records = []

        def fetch_one(shard):
            try:
                return fetch_header(shard, args.evidence, budget), None
            except Exception as exc:
                # Never include exception messages that may expose signed redirect URLs.
                return None, f"{shard}: fetch failed ({type(exc).__name__})"

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for record, error in pool.map(fetch_one, shards):
                if error:
                    errors.append(error)
                    print(error, flush=True)
                else:
                    records.append(record)
                    print(f"Header {len(records)}/{len(shards)}: {record['shard']}", flush=True)
        manifest = {
            "revision": REVISION,
            "repository": "moonshotai/Kimi-K3",
            "config_sha256": sha256(args.config.read_bytes()),
            "index_sha256": sha256(args.index.read_bytes()),
            "shards": records,
            "http_body_bytes_read": budget.read_bytes,
            "range_response_body_bytes_read": budget.read_bytes,
            "total_http_body_bytes_read": None if errors else budget.read_bytes,
            "http_body_count_scope": (
                "Successful application reads only; failures may have unmetered partial bodies"
                if errors
                else "All bodies; direct adapter dispatch does not consume redirect bodies"
            ),
            "metadata_only": True,
            "tensor_payload_bytes_requested": 0,
            "fetch_errors": errors.copy(),
            "source_metadata": source_metadata,
            "http_read_policy": "v3: direct adapter dispatch (no redirect body consumption), trusted HTTPS hosts, no implicit auth, 206, exact Range and mandatory Content-Length, exact bounded reads",
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    else:
        manifest = json.loads(manifest_path.read_text())
        errors.extend(manifest.get("fetch_errors", []))
    validate_manifest(manifest)
    for key, data in (("config", args.config.read_bytes()), ("index", args.index.read_bytes())):
        if manifest[key + "_sha256"] != sha256(data):
            errors.append(f"{key} hash does not match manifest")
        proof = manifest["source_metadata"][key]
        git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if proof["bytes"] != len(data) or proof["git_blob_sha1"] != git_blob:
            errors.append(f"{key} source proof does not match local size/Git blob hash")
    if manifest["revision"] != REVISION:
        errors.append("Manifest revision is not the pinned official revision")
    tensors = {}
    records = {record["shard"]: record for record in manifest["shards"]}
    if len(records) != len(manifest["shards"]):
        errors.append("Duplicate shard records")
    errors.extend(f"Missing shard header: {shard}" for shard in set(shards) - set(records))
    errors.extend(f"Unexpected shard header: {shard}" for shard in set(records) - set(shards))
    total_header_bytes = 0
    for shard, record in records.items():
        shard_name(shard)
        path = args.evidence / (shard + ".header.json")
        length = path.stat().st_size
        total_header_bytes += length
        if length > MAX_HEADER or total_header_bytes > MAX_TOTAL:
            raise ValueError("Offline metadata byte budget exceeded")
        data = path.read_bytes()
        if len(data) != record["header_bytes"] or sha256(data) != record["header_sha256"]:
            errors.append(f"{shard}: saved header hash or size mismatch")
        header = json.loads(data, object_pairs_hook=unique_object)
        errors.extend(
            f"{shard}: {error}"
            for error in validate_header(header, record["file_bytes"], record["header_bytes"])
        )
        for raw, metadata in header.items():
            if raw == "__metadata__":
                continue
            if raw in tensors:
                errors.append(f"Duplicate tensor across headers: {raw}")
            if index.get(raw) != shard:
                errors.append(f"{raw}: index/header shard disagreement")
            tensors[raw] = metadata
    errors.extend(f"Index tensor missing from headers: {raw}" for raw in set(index) - set(tensors))
    checkpoint_tensor_bytes = sum(
        record["file_bytes"] - 8 - record["header_bytes"] for record in records.values()
    )
    index_total_bytes = index_document.get("metadata", {}).get("total_size")
    if index_total_bytes is not None and (
        type(index_total_bytes) is not int or index_total_bytes != checkpoint_tensor_bytes
    ):
        errors.append("Index metadata.total_size disagrees with header tensor byte counts")
    expected, layers, dtype = declared_parameters(args.config)
    from minisgl.models.kimi_weight import normalize_kimi_name

    report = audit_text(tensors, expected, layers, normalize_kimi_name, dtype)
    report["errors"] = errors + report["errors"]
    report.update(
        {
            "revision": REVISION,
            "audited_shards": len(records),
            "expected_shards": len(shards),
            "header_json_bytes": total_header_bytes,
            "checkpoint_tensor_bytes_from_headers": checkpoint_tensor_bytes,
            "index_declared_tensor_bytes": index_total_bytes,
            "http_body_bytes_read": manifest["http_body_bytes_read"],
            "range_response_body_bytes_read": manifest["http_body_bytes_read"],
            "total_http_body_bytes_read": manifest.get("total_http_body_bytes_read"),
            "retry_history": manifest.get("retry_history", []),
            "retry_body_bytes_read": manifest.get("retry_body_bytes_read", 0),
            "http_body_count_scope": manifest.get(
                "http_body_count_scope", "Legacy range-response count; total HTTP bytes unverified"
            ),
            "config_sha256": manifest["config_sha256"],
            "index_sha256": manifest["index_sha256"],
            "tensor_payload_bytes_requested": 0,
            "http_read_policy": manifest["http_read_policy"],
            "source_metadata": manifest["source_metadata"],
            "scope": "Metadata only; no value, inference, GPU, distributed or performance validation",
        }
    )
    (args.evidence / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"errors", "a_log_narrowings", "engine_dtype_casts"}
            },
            indent=2,
        )
    )
    print(f"Errors: {len(report['errors'])}; see report.json")
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
