"""Diagnostic capture must retain values/layouts without changing execution."""

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

TensorTrace = runpy.run_path(str(Path(__file__).parents[1] / "validation/tensor_trace.py"))[
    "TensorTrace"
]


def update_without_torch_version(x, state, cache_indices):
    state.data.add_(x)
    return state


def test_trace_keeps_inplace_identity_and_records_layout_state_indices(tmp_path):
    trace = TensorTrace(tmp_path / "trace", prefixes=[__name__])
    x = torch.arange(12).bfloat16().reshape(3, 4)[:, ::2]
    state = torch.zeros_like(x)
    version = state._version
    previous = sys.getprofile()
    trace.install()
    try:
        with trace.step("prefill", table_idx=2):
            result = update_without_torch_version(x, state, torch.tensor([2], dtype=torch.int32))
        assert result is state and state._version == version
    finally:
        trace.close()
    assert sys.getprofile() is previous
    rows = [json.loads(line) for line in (trace.output / "events.jsonl").read_text().splitlines()]
    call = next(
        row
        for row in rows
        if row["kind"] == "call" and row["function"].endswith(".update_without_torch_version")
    )
    end = next(row for row in rows if row["kind"] == "return" and row["call"] == call["sequence"])
    before, after = call["values"]["items"], end["values"]["items"]["locals"]["items"]
    assert before["x"]["stride"] == list(x.stride())
    assert before["x"]["dtype"] == "torch.bfloat16"
    assert call["state_indices"]["cache_indices"] == [2]
    assert before["state"]["alias"] == after["state"]["alias"]
    saved_before = torch.load(trace.output / before["state"]["payload"], weights_only=True)
    saved_after = torch.load(trace.output / after["state"]["payload"], weights_only=True)
    torch.testing.assert_close(saved_before, torch.zeros_like(state), rtol=0, atol=0)
    torch.testing.assert_close(saved_after, x, rtol=0, atol=0)
    assert rows[-1]["complete"] and rows[-1]["pending_frames"] == 0


def test_parameter_views_are_metadata_and_dynamic_values_have_compact_payloads(tmp_path):
    trace = TensorTrace(tmp_path / "trace", prefixes=[])
    parameter = torch.ones(20, 30)
    trace.register_model(SimpleNamespace(weight=parameter))
    result = trace.snapshot(parameter[2:, ::2])
    assert result["role"] == "parameter" and trace.payloads == 0
    x = torch.arange(100).reshape(10, 10)[::3, ::3]
    result = trace.snapshot(x)
    saved = torch.load(trace.output / result["payload"], weights_only=True)
    assert result["stride"] == list(x.stride()) and saved.is_contiguous()
    assert saved.untyped_storage().nbytes() == saved.numel() * saved.element_size()
    torch.testing.assert_close(saved, x, rtol=0, atol=0)
    trace.close()


def test_capture_budget_cannot_silently_omit_tensor_values(tmp_path):
    trace = TensorTrace(tmp_path / "trace", prefixes=[], max_bytes=1)
    with pytest.raises(RuntimeError, match="incomplete"):
        trace.snapshot(torch.ones(2))
    trace.close(complete=False)
    rows = [json.loads(line) for line in (trace.output / "events.jsonl").read_text().splitlines()]
    assert not rows[-1]["complete"]


def test_kernel_observer_preserves_arguments_returns_and_restores_hooks(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import triton.runtime.jit as jit

    state = torch.zeros(2)
    indices = torch.tensor([1], dtype=torch.int32)
    marker = SimpleNamespace(metadata=SimpleNamespace(num_warps=4, num_ctas=1, shared=0))

    class Kernel:
        fn = update_without_torch_version

        def run(self, value, *, cache_indices, grid, num_warps):
            assert value is state and cache_indices is indices
            assert grid == (1, 1, 1) and num_warps == 4
            value.data.add_(1)
            return marker

    original = Kernel.run
    monkeypatch.setattr(jit, "JITFunction", Kernel)
    trace = TensorTrace(tmp_path / "trace", prefixes=[]).install(kernels=True)
    try:
        with trace.step("decode"):
            assert Kernel().run(state, cache_indices=indices, grid=(1, 1, 1), num_warps=4) is marker
    finally:
        trace.close()
    assert Kernel.run is original
    rows = [json.loads(line) for line in (trace.output / "events.jsonl").read_text().splitlines()]
    launch = next(row for row in rows if row["kind"] == "triton_launch")
    complete = next(row for row in rows if row["kind"] == "triton_complete")
    assert complete["launch"] == launch["sequence"]
    assert complete["configuration"]["items"]["num_warps"] == 4
    torch.testing.assert_close(state, torch.ones(2), rtol=0, atol=0)


def test_trace_keeps_pool_layout_and_saves_only_indexed_state_rows(tmp_path):
    trace = TensorTrace(tmp_path / "trace", prefixes=[])
    pool = torch.arange(9 * 3 * 4 * 4).reshape(9, 3, 4, 4)
    values = {
        "initial_state": pool,
        "initial_state_indices": torch.tensor([-1, 7, 1], dtype=torch.int32),
    }
    captured = trace.snapshot(values)["items"]["initial_state"]
    assert captured["shape"] == list(pool.shape) and captured["stride"] == list(pool.stride())
    assert captured["selection"] == {"axis": 0, "stored_indices": [1, 7]}
    saved = torch.load(trace.output / captured["payload"], weights_only=True)
    torch.testing.assert_close(saved, pool[[1, 7]], rtol=0, atol=0)
    trace.close()


def test_workspace_and_static_lookup_metadata_do_not_dump_uninitialized_storage(tmp_path):
    trace = TensorTrace(tmp_path / "trace", prefixes=[], max_bytes=1)
    result = trace.snapshot(
        {"float_workspace_buffer": torch.empty(128), "cos_sin_cache": torch.ones(32)}
    )["items"]
    assert result["float_workspace_buffer"]["role"] == "workspace"
    assert result["cos_sin_cache"]["role"] == "lookup_table"
    assert trace.payloads == trace.bytes == 0
    trace.close()


@pytest.mark.parametrize("indices", [[7, 1], [-1, 1, 1], []])
def test_selected_rows_are_selected_before_host_transfer(monkeypatch, tmp_path, indices):
    pool = torch.arange(9 * 3 * 4 * 8).reshape(9, 3, 4, 8)[..., ::2]
    expected = pool[indices].clone()
    selected_bytes = expected.numel() * expected.element_size()
    trace = TensorTrace(tmp_path / "trace", prefixes=[], max_bytes=selected_bytes)
    transfers = []
    original = torch.Tensor.to

    def observed_to(value, *args, **kwargs):
        transfers.append((tuple(value.shape), str(value.device), kwargs.get("device")))
        return original(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", observed_to)
        captured = trace.snapshot(pool, rows=indices)
    assert transfers == [(tuple(expected.shape), "cpu", "cpu")]
    assert captured["shape"] == list(pool.shape)
    assert captured["stride"] == list(pool.stride())
    assert captured["data_ptr"] == pool.data_ptr()
    assert captured["storage_offset"] == pool.storage_offset()
    assert captured["nbytes"] == pool.numel() * pool.element_size()
    assert captured["selection"] == {"axis": 0, "stored_indices": indices}
    saved = torch.load(trace.output / captured["payload"], weights_only=True)
    torch.testing.assert_close(saved, expected, rtol=0, atol=0)
    assert saved.is_contiguous() and saved.untyped_storage().nbytes() == selected_bytes
    assert trace.bytes == selected_bytes
    trace.close()


@pytest.mark.parametrize("indices", [[9], [-10]])
def test_direct_out_of_bounds_rows_fail_before_source_indexing(monkeypatch, tmp_path, indices):
    trace = TensorTrace(tmp_path / "trace", prefixes=[])
    pool = torch.zeros(9, 3, 4, 4)

    def unexpected_transfer(*args, **kwargs):
        pytest.fail("An invalid selection must not transfer any pool values")

    def unexpected_index(*args, **kwargs):
        pytest.fail("An invalid selection must fail before source-device indexing")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", unexpected_transfer)
        patch.setattr(torch.Tensor, "__getitem__", unexpected_index)
        with pytest.raises(IndexError):
            trace.snapshot(pool, rows=indices)
    assert trace.bytes == trace.payloads == 0
    trace.close(complete=False)


def test_container_pool_selection_keeps_filtering_contract(monkeypatch, tmp_path):
    trace = TensorTrace(tmp_path / "trace", prefixes=[])
    pool = torch.arange(9 * 3 * 4 * 4).reshape(9, 3, 4, 4)
    transfers = []
    original = torch.Tensor.to

    def observed_to(value, *args, **kwargs):
        transfers.append(tuple(value.shape))
        return original(value, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", observed_to)
        captured = trace.snapshot(
            {"initial_state": pool, "initial_state_indices": torch.tensor([-1, 7, 7, 1, 9])}
        )["items"]["initial_state"]
    assert tuple(pool.shape) not in transfers
    assert (2, 3, 4, 4) in transfers
    assert captured["selection"] == {"axis": 0, "stored_indices": [1, 7]}
    saved = torch.load(trace.output / captured["payload"], weights_only=True)
    torch.testing.assert_close(saved, pool[[1, 7]], rtol=0, atol=0)
    trace.close()


def test_selected_byte_budget_rejects_before_host_transfer(monkeypatch, tmp_path):
    pool = torch.zeros(9, 3, 4, 4)
    selected_bytes = 2 * pool[0].numel() * pool.element_size()
    trace = TensorTrace(tmp_path / "trace", prefixes=[], max_bytes=selected_bytes - 1)

    def unexpected_transfer(*args, **kwargs):
        pytest.fail("A rejected selection must not transfer any pool values")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", unexpected_transfer)
        with pytest.raises(RuntimeError, match="incomplete"):
            trace.snapshot(pool, rows=[1, 7])
    assert trace.bytes == trace.payloads == 0
    trace.close(complete=False)


@pytest.mark.parametrize("indices", [[], [0]])
def test_scalar_rows_raise_index_error_before_source_indexing(monkeypatch, tmp_path, indices):
    trace = TensorTrace(tmp_path / "trace", prefixes=[])
    scalar = torch.tensor(1.0)

    def unexpected_operation(*args, **kwargs):
        pytest.fail("A scalar row selection must fail before indexing or transfer")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", unexpected_operation)
        patch.setattr(torch.Tensor, "__getitem__", unexpected_operation)
        with pytest.raises(IndexError, match="scalar"):
            trace.snapshot(scalar, rows=indices)
    assert trace.bytes == trace.payloads == 0
    trace.close(complete=False)


def test_metadata_only_rows_do_not_validate_or_index(monkeypatch, tmp_path):
    trace = TensorTrace(tmp_path / "trace", prefixes=[], max_bytes=1)
    parameter = torch.ones(2, 3)
    trace.register_model(SimpleNamespace(weight=parameter))
    scalar = torch.tensor(1.0)

    def unexpected_operation(*args, **kwargs):
        pytest.fail("Metadata-only snapshots must not index or transfer")

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", unexpected_operation)
        patch.setattr(torch.Tensor, "__getitem__", unexpected_operation)
        assert trace.snapshot(parameter, rows=[99])["role"] == "parameter"
        assert trace.snapshot(scalar, rows=[99], role="workspace")["role"] == "workspace"
    assert trace.bytes == trace.payloads == 0
    trace.close()
