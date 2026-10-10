"""Synthetic metadata and HTTP safety tests; no remote or local weights read."""

import io
import struct

import pytest
from audit_kimi_headers import (
    MAX_HEADER,
    Budget,
    audit_text,
    fetch_header,
    read_range,
    shard_name,
    validate_header,
)


class Response:
    def __init__(self, status=206, body=b"12345678", headers=None):
        self.status_code = status
        self.headers = (
            {"Content-Range": "bytes 0-7/100", "Content-Length": "8"}
            if headers is None
            else headers
        )
        self.raw = io.BytesIO(body)
        self.closed = False

    def close(self):
        self.closed = True


class Session:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        assert kwargs["allow_redirects"] is False
        assert kwargs["stream"] is True
        assert kwargs["headers"]["Accept-Encoding"] == "identity"
        return next(self.responses)


def test_exact_range_reads_bounded_and_closes():
    response = Response()
    budget = Budget()
    assert read_range(Session(response), "https://huggingface.co/file", 0, 7, budget) == (
        b"12345678",
        100,
    )
    assert response.closed
    assert budget.read_bytes == 8


@pytest.mark.parametrize(
    "status,headers",
    [
        (200, {"Content-Length": "1000000000"}),
        (404, {}),
        (206, {"Content-Range": "bytes 1-8/100"}),
        (206, {"Content-Range": "bytes 0-8/100"}),
        (206, {"Content-Range": "bytes 0-7/*"}),
        (206, {"Content-Range": "bytes 0-7/7"}),
        (206, {"Content-Range": "bytes 0-7/100", "Content-Length": "100"}),
        (206, {"Content-Range": "bytes 0-7/100", "Content-Encoding": "gzip"}),
    ],
)
def test_rejects_unsafe_response_without_reading_body(status, headers):
    response = Response(status, b"sensitive payload", headers)
    budget = Budget()
    with pytest.raises(ValueError):
        read_range(Session(response), "https://huggingface.co/file", 0, 7, budget)
    assert response.raw.tell() == 0
    assert response.closed
    assert budget.read_bytes == 0


@pytest.mark.parametrize("body", [b"short", b"123456789" + b"payload" * 100])
def test_truncation_and_overrun_bounded(body):
    response = Response(body=body, headers={"Content-Range": "bytes 0-7/100"})
    with pytest.raises(ValueError):
        read_range(Session(response), "https://huggingface.co/file", 0, 7, Budget())
    assert response.raw.tell() == 0
    assert response.closed


def test_manual_redirect_does_not_read_body():
    redirect = Response(302, b"do not read", {"Location": "https://us.aws.cdn.hf.co/file"})
    result = read_range(
        Session(redirect, Response()), "https://huggingface.co/file", 0, 7, Budget()
    )
    assert result[0] == b"12345678"
    assert redirect.raw.tell() == 0
    assert redirect.closed


def test_redirect_https_and_limit():
    redirect = Response(302, headers={"Location": "http://us.aws.cdn.hf.co/file"})
    with pytest.raises(ValueError, match="HTTPS"):
        read_range(Session(redirect), "https://huggingface.co/file", 0, 7, Budget())
    redirects = [
        Response(302, headers={"Location": "https://us.aws.cdn.hf.co/file"}) for _ in range(7)
    ]
    with pytest.raises(ValueError, match="redirect limit"):
        read_range(Session(*redirects), "https://huggingface.co/file", 0, 7, Budget())
    assert all(r.closed and r.raw.tell() == 0 for r in redirects)


def test_budget_prevents_network_and_oversized_range():
    session = Session()
    with pytest.raises(ValueError, match="budget"):
        read_range(session, "https://huggingface.co/file", 0, 7, Budget(7))
    with pytest.raises(ValueError, match="limits"):
        read_range(session, "https://huggingface.co/file", 0, MAX_HEADER, Budget())
    assert session.calls == []


@pytest.mark.parametrize(
    "name",
    [
        "../a.safetensors",
        "/a.safetensors",
        "a/../b.safetensors",
        "a\\b.safetensors",
        "a?foo.safetensors",
        "a/b.safetensors",
    ],
)
def test_reject_shard_path_traversal(name):
    with pytest.raises(ValueError):
        shard_name(name)


def test_oversized_header_length_never_fetches_header(monkeypatch, tmp_path):
    import audit_kimi_headers as auditor

    calls = []

    def fake_range(*args):
        calls.append(args)
        return struct.pack("<Q", MAX_HEADER + 1), 10**12

    monkeypatch.setattr(auditor, "read_range", fake_range)
    with pytest.raises(ValueError, match="oversized"):
        fetch_header("model.safetensors", tmp_path, Budget())
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []


def tensor(shape, dtype, offsets=None):
    value = {"shape": shape, "dtype": dtype}
    if offsets is not None:
        value["data_offsets"] = offsets
    return value


def test_offset_bytes_and_complete_coverage():
    header = {"a": tensor([2, 2], "BF16", [0, 8]), "b": tensor([4], "U8", [8, 12])}
    assert validate_header(header, 40, 20) == []
    header["a"]["data_offsets"] = [0, 7]
    errors = validate_header(header, 40, 20)
    assert any("byte count" in e for e in errors)
    assert any("gap" in e for e in errors)
    header["a"]["data_offsets"] = [0, 13]
    assert any("outside" in e for e in validate_header(header, 40, 20))


def normalize(name, layers):
    return None if name.startswith("vision.") else name


def test_packing_dtype_shapes_and_a_log_narrowing():
    tensors = {
        "m.weight_packed": tensor([2, 32], "U8"),
        "m.weight_scale": tensor([2, 2], "U8"),
        "x.self_attn.A_log": tensor([128], "F32"),
        "vision.weight": tensor([3], "BF16"),
    }
    expected = {"m.weight": ([2, 64], "BF16"), "x.self_attn.A_log": ([64], "F32")}
    result = audit_text(tensors, expected, 2, normalize)
    assert result["errors"] == []
    assert result["packed_matrices"] == 1
    assert len(result["a_log_narrowings"]) == 1
    assert result["mapped_tensors"] == 2
    assert result["raw_dtype_histogram"] == {"U8": 2, "F32": 1, "BF16": 1}
    tensors["m.weight_scale"]["shape"] = [2, 1, 2, 1]
    assert audit_text(tensors, expected, 2, normalize)["errors"] == []
    tensors["x.self_attn.A_log"]["shape"] = [1, 1, 128, 1]
    assert audit_text(tensors, expected, 2, normalize)["errors"] == []


@pytest.mark.parametrize(
    "mutation,substring",
    [
        (lambda t: t["m.weight_packed"].update(dtype="BF16"), "packed storage"),
        (lambda t: t["m.weight_packed"].update(shape=[3, 32]), "loaded shape"),
        (lambda t: t["m.weight_scale"].update(shape=[2, 1]), "scale must"),
        (lambda t: t["m.weight_scale"].update(dtype="F32"), "scale must"),
        (lambda t: t.pop("m.weight_scale"), "scale must"),
        (lambda t: t["x.self_attn.A_log"].update(shape=[32]), "loaded shape"),
        (lambda t: t["x.self_attn.A_log"].update(shape=[2, 32]), "loaded shape"),
    ],
)
def test_mismatches_are_errors(mutation, substring):
    tensors = {
        "m.weight_packed": tensor([2, 32], "U8"),
        "m.weight_scale": tensor([2, 2], "U8"),
        "x.self_attn.A_log": tensor([128], "F32"),
    }
    expected = {"m.weight": ([2, 64], "BF16"), "x.self_attn.A_log": ([64], "F32")}
    mutation(tensors)
    assert any(substring in e for e in audit_text(tensors, expected, 2, normalize)["errors"])


def test_dense_dtype_missing_orphan_and_duplicate_errors():
    result = audit_text(
        {
            "m.weight": tensor([2, 32], "F32"),
            "m.weight_packed": tensor([2, 16], "U8"),
            "m.weight_scale": tensor([2, 1], "U8"),
            "orphan.weight_scale": tensor([1], "U8"),
        },
        {"m.weight": ([2, 32], "BF16"), "missing": ([1], "F32")},
        2,
        normalize,
    )
    for substring in ("duplicate", "Missing runtime key", "orphan scale"):
        assert any(substring in e for e in result["errors"])


@pytest.mark.parametrize("wrong_index", [False, True])
def test_offline_end_to_end_and_index_consistency(monkeypatch, tmp_path, wrong_index):
    import json
    import sys

    import audit_kimi_headers as auditor

    config = tmp_path / "config.json"
    config.write_text("{}")
    index = tmp_path / "index.json"
    shard = "model.safetensors"
    index.write_text(
        json.dumps({"weight_map": {"missing.weight" if wrong_index else "model.weight": shard}})
    )
    header = json.dumps({"model.weight": tensor([2, 2], "BF16", [0, 8])}).encode()
    (tmp_path / (shard + ".header.json")).write_bytes(header)
    manifest = {
        "revision": auditor.REVISION,
        "repository": "moonshotai/Kimi-K3",
        "metadata_only": True,
        "tensor_payload_bytes_requested": 0,
        "config_sha256": auditor.sha256(config.read_bytes()),
        "index_sha256": auditor.sha256(index.read_bytes()),
        "http_body_bytes_read": len(header) + 8,
        "shards": [
            {
                "shard": shard,
                "header_bytes": len(header),
                "header_sha256": auditor.sha256(header),
                "file_bytes": len(header) + 16,
            }
        ],
    }
    add_proof(manifest, config.read_bytes(), index.read_bytes())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(
        auditor, "declared_parameters", lambda _: ({"model.weight": ((2, 2), "BF16")}, 1, "BF16")
    )

    def prohibited_fetch(*args, **kwargs):
        raise AssertionError("Offline mode attempted a network request")

    monkeypatch.setattr(auditor, "fetch_header", prohibited_fetch)
    monkeypatch.setattr(sys, "argv", ["audit", str(config), str(index), str(tmp_path)])
    if wrong_index:
        with pytest.raises(SystemExit) as exc:
            auditor.main()
        assert exc.value.code == 1
    else:
        auditor.main()
    report = json.loads((tmp_path / "report.json").read_text())
    if wrong_index:
        assert any("index/header shard disagreement" in e for e in report["errors"])
        assert any("Index tensor missing" in e for e in report["errors"])
    else:
        assert report["errors"] == []
    assert report["tensor_payload_bytes_requested"] == 0


def test_exact_read_never_probes_past_header_boundary():
    response = Response(body=b"12345678TENSOR_VALUES")
    assert (
        read_range(Session(response), "https://huggingface.co/file", 0, 7, Budget())[0]
        == b"12345678"
    )
    assert response.raw.tell() == 8
    assert response.closed
    response = Response(body=b"short")
    with pytest.raises(ValueError, match="Truncated"):
        read_range(Session(response), "https://huggingface.co/file", 0, 7, Budget())


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/file",
        "https://127.0.0.1/file",
        "https://huggingface.co.evil.example/file",
        "https://user:password@huggingface.co/file",
        "https://huggingface.co:8443/file",
        "http://huggingface.co/file",
    ],
)
def test_untrusted_redirect_never_requested(url):
    redirect = Response(302, headers={"Location": url})
    session = Session(redirect)
    with pytest.raises(ValueError, match="trusted HTTPS"):
        read_range(session, "https://huggingface.co/file", 0, 7, Budget())
    assert len(session.calls) == 1
    assert redirect.closed and redirect.raw.tell() == 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(repository="other/repo"),
        lambda m: m.update(revision="main"),
        lambda m: m.update(metadata_only=False),
        lambda m: m.update(tensor_payload_bytes_requested=1),
        lambda m: m.update(http_body_bytes_read=17),
        lambda m: m.update(http_body_bytes_read=19),
        lambda m: m["shards"][0].update(header_bytes=0),
        lambda m: m["shards"][0].update(header_bytes=MAX_HEADER + 1),
        lambda m: m["shards"][0].update(file_bytes=4),
        lambda m: m.update(fetch_errors=["incomplete"]),
    ],
)
def test_manifest_provenance_and_counts_fail_closed(mutation):
    from audit_kimi_headers import REVISION, validate_manifest

    manifest = {
        "repository": "moonshotai/Kimi-K3",
        "revision": REVISION,
        "metadata_only": True,
        "tensor_payload_bytes_requested": 0,
        "http_body_bytes_read": 18,
        "shards": [{"shard": "model.safetensors", "header_bytes": 10, "file_bytes": 26}],
    }
    add_proof(manifest)
    assert validate_manifest(manifest) == 18
    mutation(manifest)
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_public_session_suppresses_implicit_credentials():
    import requests
    from audit_kimi_headers import public_session

    with public_session() as session:
        request = requests.Request(
            "GET", "https://huggingface.co/file", headers={"Authorization": "must be removed"}
        )
        prepared = session.prepare_request(request)
        assert "Authorization" not in prepared.headers
        assert session.trust_env is True


def add_proof(manifest, config=b"{}", index=b"{}"):
    import hashlib

    from audit_kimi_headers import REVISION, sha256

    manifest["source_metadata"] = {}
    manifest["http_read_policy"] = "Synthetic fixture, no network requests"
    for key, data, filename in (
        ("config", config, "config.json"),
        ("index", index, "model.safetensors.index.json"),
    ):
        digest = sha256(data)
        manifest[key + "_sha256"] = digest
        manifest["source_metadata"][key] = {
            "url": f"https://huggingface.co/moonshotai/Kimi-K3/resolve/{REVISION}/{filename}",
            "revision": REVISION,
            "sha256": digest,
            "etag": digest,
            "bytes": len(data),
            "git_blob_sha1": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest(),
            "http_body_bytes_read": 0,
        }


@pytest.mark.parametrize(
    "field,value",
    [
        ("url", "https://evil.test"),
        ("revision", "main"),
        ("sha256", "a" * 64),
        ("etag", "wrong"),
        ("bytes", -1),
        ("http_body_bytes_read", 1),
    ],
)
def test_invalid_pinned_source_proof(field, value):
    from audit_kimi_headers import REVISION, validate_manifest

    manifest = {
        "repository": "moonshotai/Kimi-K3",
        "revision": REVISION,
        "metadata_only": True,
        "tensor_payload_bytes_requested": 0,
        "http_body_bytes_read": 18,
        "shards": [{"shard": "model.safetensors", "header_bytes": 10, "file_bytes": 26}],
    }
    add_proof(manifest)
    manifest["source_metadata"]["config"][field] = value
    with pytest.raises(ValueError, match="source metadata proof"):
        validate_manifest(manifest)
    del manifest["source_metadata"]
    with pytest.raises(ValueError, match="Missing pinned"):
        validate_manifest(manifest)


def test_real_requests_session_does_not_consume_redirect_body():
    import requests
    from audit_kimi_headers import public_session

    class Adapter(requests.adapters.BaseAdapter):
        def __init__(self):
            self.responses = []

        def send(self, request, **kwargs):
            assert kwargs["stream"] is True
            assert "proxies" in kwargs
            response = requests.Response()
            response.status_code = 302
            response.url = request.url
            response.request = request
            response.headers["Location"] = "https://us.aws.cdn.hf.co/file"
            response.raw = io.BytesIO(b"REDIRECT BODY MUST NOT BE READ")
            self.responses.append(response)
            return response

        def close(self):
            pass

    adapter = Adapter()
    with public_session() as session:
        session.mount("https://", adapter)
        response = session.get("https://huggingface.co/file", stream=True, allow_redirects=False)
        assert response.status_code == 302
        assert response.raw.tell() == 0
        assert response._content is False
        response.close()


def test_transport_errors_never_expose_signed_urls():
    class FailingSession:
        def get(self, *args, **kwargs):
            raise RuntimeError("https://us.aws.cdn.hf.co/file?Signature=secret")

    with pytest.raises(ValueError, match="Metadata transport failed") as exc:
        read_range(FailingSession(), "https://huggingface.co/file", 0, 7, Budget())
    assert "Signature" not in str(exc.value)
    assert exc.value.__suppress_context__ is True


def test_retry_accounting_includes_repeated_prefixes():
    from audit_kimi_headers import REVISION, validate_manifest

    manifest = {
        "repository": "moonshotai/Kimi-K3",
        "revision": REVISION,
        "metadata_only": True,
        "tensor_payload_bytes_requested": 0,
        "http_body_bytes_read": 26,
        "shards": [{"shard": "model.safetensors", "header_bytes": 10, "file_bytes": 26}],
        "retry_body_bytes_read": 8,
        "retry_history": [
            {
                "exception": "ReadTimeout",
                "stage": "before_response",
                "unknown_partial_body": False,
                "successful_prefix_bytes": 8,
                "failed_shards": ["model.safetensors"],
            }
        ],
    }
    add_proof(manifest)
    assert validate_manifest(manifest) == 18
    manifest["retry_history"][0]["unknown_partial_body"] = True
    with pytest.raises(ValueError, match="unmetered"):
        validate_manifest(manifest)
