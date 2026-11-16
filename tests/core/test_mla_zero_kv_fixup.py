"""CPU protocols for native zero-KV fixup; no CUDA/native math execution."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.mla.cutedsl import CuTeMLADecodePlan
from minisgl.kernel import mla as native
from test_mla_cutedsl_adapter import metadata


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("CPU fixup check initialized CUDA")

    monkeypatch.setattr(torch.cuda, "init", forbid)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    native._warm_zero_kv_fixup.cache_clear()
    yield
    native._warm_zero_kv_fixup.cache_clear()


class GuardTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def guarded(value):
    return value.as_subclass(GuardTensor)


@pytest.fixture
def protocol(monkeypatch):
    out = guarded(torch.ones(2, 3, 8, dtype=torch.bfloat16))
    lse = guarded(torch.full((2, 3), -2.25))
    lengths = guarded(torch.tensor([4, 0], dtype=torch.int32))
    offsets = guarded(torch.arange(3, dtype=torch.int32))
    out[1].fill_(torch.nan)
    lse[1].fill_(torch.nan)
    events, mode = [], SimpleNamespace(capture=False, error=False)

    def jit(dtype):
        assert dtype == torch.bfloat16
        events.append("jit")

    def fixup(actual, stats, local, cu_q, max_q):
        assert actual is out and stats is lse and local is lengths and cu_q is offsets
        assert max_q == 1
        events.append("native")
        if mode.error:
            raise RuntimeError("selected native failure")
        actual[1].zero_()
        stats[1].fill_(-torch.inf)

    ops = SimpleNamespace(_jit_fixup_module=jit, fixup_zero_kv_rows=fixup)
    monkeypatch.setattr(native, "_zero_kv_kernels", lambda: (ops, nullcontext))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: mode.capture)
    return (out, lse, lengths, offsets), events, mode, ops


def test_native_neutral_rows_keep_units_inputs_and_warm_once(protocol):
    args, events, mode, _ = protocol
    out, lse, lengths, offsets = args
    before_lengths, before_offsets = lengths.clone(), offsets.clone()
    assert native.try_fixup_zero_kv_rows(*args)
    assert out[0].eq(1).all() and lse[0].eq(-2.25).all()
    assert out[1].eq(0).all() and lse[1].eq(-torch.inf).all()
    mode.capture = True
    assert native.try_fixup_zero_kv_rows(*args)
    assert events == ["jit", "native", "native"]
    assert torch.equal(lengths, before_lengths) and torch.equal(offsets, before_offsets)
    # A nonempty corrupt producer stays visible; fixup is not a finite sanitizer.
    lse[0, 0] = torch.nan
    lse[0, 1] = torch.inf
    native.try_fixup_zero_kv_rows(*args)
    assert lse[0, 0].isnan() and lse[0, 1].isposinf()


@pytest.mark.parametrize("bad", ["dtype", "layout", "device", "grad", "alias", "offset_dtype"])
def test_unsupported_metadata_declines_before_jit(protocol, bad):
    args, events, _, _ = protocol
    out, lse, lengths, offsets = args
    if bad == "dtype":
        out = out.float()
    elif bad == "layout":
        out = guarded(torch.ones(2, 3, 16, dtype=torch.bfloat16))[..., ::2]
    elif bad == "device":

        class OtherDevice(GuardTensor):
            @property
            def device(self):
                return torch.device("cuda:1")

        lse = lse.as_subclass(OtherDevice)
    elif bad == "grad":
        out.requires_grad_()
    elif bad == "alias":
        # Construct exact compact alias with the correct shape.
        lse = out.view(torch.float32).as_strided((2, 3), (3, 1))
    else:
        offsets = offsets.long()
    assert not native.try_fixup_zero_kv_rows(out, lse, lengths, offsets)
    assert events == []


def test_cold_capture_and_selected_errors_propagate_once(protocol, monkeypatch):
    args, events, mode, ops = protocol
    mode.capture = True
    with pytest.raises(RuntimeError, match="Warm native zero-KV"):
        native.try_fixup_zero_kv_rows(*args)
    assert events == []
    mode.capture = False
    mode.error = True
    with pytest.raises(RuntimeError, match="selected native failure"):
        native.try_fixup_zero_kv_rows(*args)
    assert events == ["jit", "native"]
    native._warm_zero_kv_fixup.cache_clear()

    def failed_jit(dtype):
        raise ImportError("selected JIT failed")

    monkeypatch.setattr(ops, "_jit_fixup_module", failed_jit)
    with pytest.raises(ImportError, match="selected JIT"):
        native.try_fixup_zero_kv_rows(*args)
    assert events == ["jit", "native"]


def test_optional_module_declines_only_before_selection(protocol, monkeypatch):
    args, events, _, _ = protocol

    def missing():
        raise ImportError("optional SDK unavailable")

    monkeypatch.setattr(native, "_zero_kv_kernels", missing)
    assert not native.try_fixup_zero_kv_rows(*args) and events == []


@pytest.mark.parametrize("graph", [False, True])
def test_query_offsets_prepared_reused_and_not_created_in_run(graph):
    plan = CuTeMLADecodePlan(torch.empty(1), **({"batch_size": 2, "max_pages": 2} if graph else {}))
    plan.prepare(metadata([8, 0], [[0], [0]]), {})
    saved = plan.cu_seqlens_q
    assert saved.dtype == torch.int32 and saved.tolist() == [0, 1, 2]
    plan.prepare(metadata([9, 1], [[0], [0]]), {})
    assert plan.cu_seqlens_q is saved
    if not graph:
        plan.prepare(metadata([1], [[0]]), {})
        assert plan.cu_seqlens_q.tolist() == [0, 1]


def test_missing_query_offsets_rejects_cold_capture_before_allocation(monkeypatch):
    plan = object.__new__(CuTeMLADecodePlan)
    plan.device, plan.cu_seqlens_q = torch.device("cuda:0"), None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="Prepare MLA query offsets"):
        plan._prepare_query_offsets(2)


@pytest.mark.parametrize("case", ["native", "strided", "error"])
def test_plan_handoff_or_original_layout_fallback_without_second_decode(monkeypatch, case):
    calls, results = [], []

    def decode(**options):
        calls.append(options)
        if case == "strided":
            out = torch.ones(2, 1, 512, 3, dtype=torch.bfloat16).transpose(-1, -2)
            lse = torch.ones(3, 2).t()
        else:
            out = torch.ones(2, 1, 3, 512, dtype=torch.bfloat16)
            lse = torch.full((2, 3), -2.25)
        out[1].fill_(torch.nan)
        lse[1].fill_(torch.nan)
        results.append((out, lse))
        return out, lse

    def fixup(out, lse, local, cu_q):
        assert case != "strided"
        assert local is plan.local and cu_q is plan.cu_seqlens_q
        assert out.data_ptr() == results[0][0].data_ptr()
        assert lse.data_ptr() == results[0][1].data_ptr()
        if case == "error":
            raise RuntimeError("selected fixup failed")
        out[1].zero_()
        lse[1].fill_(-torch.inf)
        return True

    monkeypatch.setitem(
        sys.modules,
        "flashinfer.decode",
        SimpleNamespace(trtllm_batch_decode_with_kv_cache_mla=decode),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.kernels.ops.attention.utils",
        SimpleNamespace(concat_mla_absorb_q_general=lambda *args: pytest.fail("concat")),
    )
    monkeypatch.setitem(sys.modules, "tvm_ffi", SimpleNamespace(use_torch_stream=nullcontext))
    monkeypatch.setattr(native, "try_fixup_zero_kv_rows", fixup)
    plan = CuTeMLADecodePlan(torch.empty(1))
    plan.prepare(metadata([4, 0], [[0], [0]], cp=8), {})
    q = torch.zeros(2, 3, 576).to(torch.float8_e4m3fn)
    pages = torch.zeros(1, 64, 576).to(torch.float8_e4m3fn)
    if case == "error":
        with pytest.raises(RuntimeError, match="selected fixup failed"):
            plan.run(q, pages, nope_dim=128, scale=0.25)
    else:
        out, lse = plan.run(q, pages, nope_dim=128, scale=0.25)
        assert out[0].eq(1).all() and out[1].eq(0).all()
        assert lse[0].eq(1 if case == "strided" else -2.25).all()
        assert lse[1].eq(-torch.inf).all()
    assert len(calls) == 1 and calls[0]["return_lse_base"] == "basee"
    assert calls[0]["cp_world"] == 8
