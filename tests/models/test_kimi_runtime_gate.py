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
    assert packed["nonfinite_faults"]["status"] == "passed"
    assert report["long_prefill"] == "not_run"
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


def test_provenance_identifies_actual_backend_and_sources():
    from kimi_gate_provenance import provenance

    report = provenance(torch.device("cpu"))
    assert report["implementation"]["revision"]
    assert "tests/models/run_kimi_runtime_gate.py" in report["implementation"]["source_sha256"]
    assert report["references"]["executed"] is False
    assert report["references"]["scope"] == "not_run"
    assert report["backends"]["prefill"]["kda"] == "minisgl_eager_sequential_fp32_recurrence"
    assert report["backends"]["decode"]["kda"] == report["backends"]["prefill"]["kda"]
    assert report["backends"]["sglang_gpu_kernels_executed"] is False
    assert report["device"]["model"]
    assert report["libraries"]["torch"] == torch.__version__
    json.dumps(report)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_long_prefill_short_smoke(fixtures, dtype):
    from kimi_long_prefill_gate import run_long_prefill_gate

    report = run_long_prefill_gate(fixtures, dtype, torch.device("cpu"), length=19)
    assert report["status"] == "passed"


def test_long_prefill_cli_is_opt_in(fixtures, tmp_path, monkeypatch, capsys):
    import kimi_long_prefill_gate

    called = []

    def fake(root, dtype, device):
        called.append((root, dtype, device))
        return {"status": "test_stub"}

    monkeypatch.setattr(kimi_long_prefill_gate, "run_long_prefill_gate", fake)
    report_path = tmp_path / "long.json"
    main(["--fixture-dir", str(fixtures), "--long-prefill", "--report", str(report_path)])
    assert len(called) == 1
    assert json.loads(report_path.read_text())["long_prefill"] == {"status": "test_stub"}
    capsys.readouterr()


def test_provenance_export_without_git(monkeypatch):
    import kimi_gate_provenance

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(kimi_gate_provenance.subprocess, "check_output", unavailable)
    report = kimi_gate_provenance.provenance("cpu")
    assert report["implementation"]["revision"] is None
    assert report["implementation"]["worktree_status"] is None
    assert report["implementation"]["source_sha256"]


def test_reject_long_prefill_with_generate_only(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["--fixture-dir", str(tmp_path), "--generate-only", "--long-prefill"])
    assert error.value.code == 2


@pytest.mark.parametrize("component", ["recurrent", "convolution", "mla"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_finite_detector_rejects_each_live_component(fixtures, component, value):
    from kimi_fault_gate import NonFiniteStateError, assert_finite_live_state
    from run_kimi_runtime_gate import PROMPT, batch_for, cache_for, page_table

    device = torch.device("cpu")
    model, config = load_fixture(fixtures / "dense", torch.float32, device)
    cache, table = cache_for(config, torch.float32, device), page_table(device)
    cache._mla.fill_(float("nan"))
    with torch.inference_mode():
        model.forward_batch(batch_for(PROMPT, device), cache, table)
    assert_finite_live_state(cache, table)
    tensor = {
        "recurrent": cache.recurrent._recurrent[:, 0],
        "convolution": cache.recurrent._conv[:, 0],
        "mla": cache._mla[:, 0],
    }[component]
    tensor[(0,) * tensor.ndim] = value
    with pytest.raises(NonFiniteStateError) as error:
        assert_finite_live_state(cache, table)
    assert error.value.component == component
    assert (error.value.slot, error.value.uid) == (0, 10)
    cache.release_request(0, 10)
    assert_finite_live_state(cache, table)


def test_generate_only_cuda_does_not_probe_device(tmp_path, monkeypatch, capsys):
    def unexpected_probe(*args, **kwargs):
        raise AssertionError("Fixture generation must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_name", unexpected_probe)
    output = tmp_path / "report.json"
    main(
        [
            "--fixture-dir",
            str(tmp_path / "fixtures"),
            "--generate-only",
            "--device",
            "cuda",
            "--report",
            str(output),
        ]
    )
    report = json.loads(output.read_text())
    assert report["provenance"]["backends"]["executed"] is False
    assert report["provenance"]["device"]["model"] is None
    capsys.readouterr()


def test_fault_gate_requires_detection(monkeypatch):
    import kimi_fault_gate

    monkeypatch.setattr(kimi_fault_gate, "assert_finite_live_state", lambda *args: None)
    with pytest.raises(AssertionError, match="was not detected"):
        kimi_fault_gate._expect_recurrent_fault(None, None, 0, 10)


@pytest.mark.parametrize("error_type", [RuntimeError, AssertionError])
def test_fault_gate_does_not_swallow_unexpected_failures(monkeypatch, error_type):
    import kimi_fault_gate

    failure = error_type("unrelated detector failure")

    def fail(*args):
        raise failure

    monkeypatch.setattr(kimi_fault_gate, "assert_finite_live_state", fail)
    with pytest.raises(error_type) as error:
        kimi_fault_gate._expect_recurrent_fault(None, None, 0, 10)
    assert error.value is failure
