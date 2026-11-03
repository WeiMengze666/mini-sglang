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
