"""CPU-only checks for V3's independent mature-entry observation and case matrix."""

import importlib.util
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from minisgl.attention.kda import sgl


@pytest.fixture
def diagnostic():
    path = Path(__file__).parents[1] / "validation/kda_native_prefill_diagnostic.py"
    spec = importlib.util.spec_from_file_location("prefill_diagnostic_v3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v3_retains_two_state_dtypes_and_72_large_cases(diagnostic):
    assert diagnostic.STATE_DTYPES == (torch.bfloat16, torch.float32)
    cases = diagnostic.case_matrix(True)
    assert len(cases) * 2 * 3 * 2 == 72
    assert cases[1] == (1,) * 8 and cases[3] == (1,) * 48
    assert cases[4] == (1, 65, 2048, 32768)
    assert len(cases[5]) == 48
    assert not torch.cuda.is_initialized()


@pytest.fixture
def native_chunk(monkeypatch):
    calls = []

    def chunk(**kw):
        calls.append(kw)
        return kw["v"].clone()

    name = "sglang.srt.layers.attention.linear.kernels.kda_triton"
    native = ModuleType(name)
    native.chunk_kda = chunk
    monkeypatch.setitem(sys.modules, name, native)
    original_kernels = lambda: (chunk, None, nullcontext)
    monkeypatch.setattr(sgl, "_kernels", original_kernels)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: SimpleNamespace(cuda_stream=7))
    monkeypatch.setitem(
        sys.modules,
        "tvm_ffi",
        SimpleNamespace(
            get_raw_stream=lambda device: 7,
            device=lambda name: name,
        ),
    )
    return native, chunk, calls, original_kernels


@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_v3_observes_both_actual_chunk_entries_without_changing_arguments(
    diagnostic, native_chunk, state_dtype
):
    native, original, calls, original_kernels = native_chunk
    values = [torch.randn(1, 2, 1, 128).bfloat16() for _ in range(4)]
    kw = {
        **dict(zip(("q", "k", "v", "g"), values)),
        "beta": torch.randn(1, 2, 1).bfloat16(),
        "initial_state": torch.zeros(5, 1, 128, 128, dtype=state_dtype),
        "initial_state_indices": torch.tensor([1, 4], dtype=torch.int32),
        "cu_seqlens": torch.tensor([0, 1, 2], dtype=torch.int32),
        "A_log": torch.zeros(1),
        "dt_bias": torch.arange(128, dtype=torch.float32),
        "lower_bound": -5.0,
        "beta_is_raw": True,
        "use_qk_l2norm_in_kernel": True,
        "chunk_indices": torch.tensor([[0, 0], [1, 0]], dtype=torch.int32),
    }
    with diagnostic.ChunkTrace(object, state_dtype=state_dtype, expected_stream=7) as tracer:
        direct = native.chunk_kda(**kw)
        tracer.mode = "mini_adapter"
        actual = sgl._kernels()[0](**kw)
        evidence = tracer.compare()
        assert len(evidence) == 2
        assert all(row["native_backend"] == "sgl_chunk" for row in evidence)
        assert all(row["dt_bias"]["shape"] == [128] for row in evidence)
        assert all(row["ffi_stream"] == row["torch_stream"] == 7 for row in evidence)
    assert native.chunk_kda is original and sgl._kernels is original_kernels
    assert len(calls) == 2
    for recorded in calls:
        assert recorded.keys() == kw.keys()
        assert all(recorded[name] is value for name, value in kw.items())
    torch.testing.assert_close(actual, direct, rtol=0, atol=0)
    assert not torch.cuda.is_initialized()


def test_v3_does_not_accept_missing_native_observation(diagnostic):
    with pytest.raises(AssertionError, match="Expected one observed native entry"):
        diagnostic.ChunkTrace(object, state_dtype=torch.bfloat16).compare()
