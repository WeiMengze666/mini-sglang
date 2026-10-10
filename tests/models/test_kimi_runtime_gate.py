"""Regression checks for the offline gate; these tests never request CUDA/network."""

import json
from pathlib import Path

import pytest
import torch
from minisgl.models.kimi_weight import load_kimi_weight
from run_kimi_runtime_gate import (
    generate_fixtures,
    load_fixture,
    main,
    run_runtime_gate,
    verify_fixtures,
)


@pytest.fixture(autouse=True)
def single_threaded():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def fixtures(tmp_path):
    root = tmp_path / "fixtures"
    generate_fixtures(root)
    return root


def test_repeatable_fixtures_and_cross_shard_dense_oracle(fixtures, tmp_path, monkeypatch):
    import huggingface_hub

    def no_network(*args, **kwargs):
        raise AssertionError("Local runtime fixtures must not contact the Hub")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", no_network)
    other = tmp_path / "second"
    manifest = generate_fixtures(other)
    assert manifest == verify_fixtures(fixtures)
    assert generate_fixtures(fixtures) == manifest
    assert 0 < manifest["parameter_count"] < 150_000
    assert manifest["packed_tensor_count"] > 20
    dense = dict(load_kimi_weight(str(fixtures / "dense"), "cpu", dtype=torch.float32))
    decoded = dict(load_kimi_weight(str(fixtures / "packed"), "cpu", dtype=torch.float32))
    assert dense.keys() == decoded.keys()
    for name in dense:
        torch.testing.assert_close(dense[name], decoded[name], rtol=0, atol=0)
    index = json.loads((fixtures / "packed/model.safetensors.index.json").read_text())["weight_map"]
    for name, shard in index.items():
        if name.endswith(".weight_packed"):
            assert index[name.removesuffix(".weight_packed") + ".weight_scale"] != shard


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_runtime_gate_dense_and_packed_integration(fixtures, dtype):
    report = run_runtime_gate(fixtures, dtype, torch.device("cpu"))
    assert report["status"] == "passed"
    assert report["engine"] == "not_run"
    assert report["released_checkpoint_accuracy"] == "not_tested"
    assert report["independent_full_hybrid_reference"] == "not_tested"
    dense, packed = (report["fixtures"][kind] for kind in ("dense", "packed"))
    assert dense["greedy_token_ids"] == packed["greedy_token_ids"]
    assert len(dense["greedy_token_ids"]) == 5
    assert packed["weight_bytes"] < dense["weight_bytes"] / 2
    assert "failed_forward_restart" in packed["checks"]
    assert "stale_abort_isolation" in packed["checks"]
    assert "packed_dense_logits" in packed["checks"]
    model, _ = load_fixture(fixtures / "packed", dtype, torch.device("cpu"))
    state = model.state_dict()
    names = verify_fixtures(fixtures)["packed_names"]
    for name in names:
        assert name not in state
        for suffix in (".weight_packed", ".weight_scale"):
            assert state[name.removesuffix(".weight") + suffix].dtype == torch.uint8


def test_reject_modified_fixture_without_overwriting(fixtures):
    path = fixtures / "dense/config.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        generate_fixtures(fixtures)
    assert path.read_text() == "{}"


def test_reject_nonempty_destination(tmp_path):
    (tmp_path / "keep.txt").write_text("existing user file")
    with pytest.raises(FileExistsError, match="empty"):
        generate_fixtures(tmp_path)
    assert (tmp_path / "keep.txt").read_text() == "existing user file"


def test_reject_changed_seed(fixtures):
    with pytest.raises(ValueError, match="seed"):
        generate_fixtures(fixtures, seed=123)


def test_cuda_requested_does_not_silently_fallback(fixtures, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        run_runtime_gate(fixtures, device=torch.device("cuda"))


def test_remote_identifier_rejected():
    with pytest.raises(ValueError, match="local fixture"):
        load_fixture(Path("moonshotai/Kimi-K3"), torch.float32, torch.device("cpu"))


def test_cli_generation_and_report(tmp_path, capsys):
    root, output = tmp_path / "fixtures", tmp_path / "report.json"
    assert main(["--fixture-dir", str(root), "--generate-only", "--report", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "generated_only"
    assert "generated_only" in capsys.readouterr().out
    assert main(["--fixture-dir", str(root), "--report", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["status"] == "passed" and report["device"] == "cpu"


@pytest.mark.parametrize(
    "arguments", [["--engine-smoke"], ["--generate-only"], ["--official-source", "unused.py"]]
)
def test_cli_rejects_incomplete_or_conflicting_options(arguments):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
