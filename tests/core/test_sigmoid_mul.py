"""CPU fallback math and native adapter protocols; no CUDA/native math run."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.mla import MLAOutput
from minisgl.kernel import activation as native
from minisgl.layers import mla
from minisgl.layers.activation import sigmoid_mul


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("CPU sigmoid-mul test initialized CUDA")

    monkeypatch.setattr(torch.cuda, "init", forbid)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)
    native._warm_sigmoid_mul.cache_clear()
    yield
    native._warm_sigmoid_mul.cache_clear()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["dense", "strided", "broadcast", "empty"])
def test_fallback_keeps_math_inputs_and_owned_output(monkeypatch, dtype, layout):
    monkeypatch.setattr(native, "_sigmoid_mul_ops", lambda: pytest.fail("CPU loaded native SDK"))
    x = torch.linspace(-8, 8, 48).reshape(3, 16).to(dtype)
    gate = x.flip(-1).clone()
    if layout == "strided":
        x, gate = x[:, ::2], gate[:, ::2]
    elif layout == "broadcast":
        gate = gate[:1]
    elif layout == "empty":
        x, gate = x[:0], gate[:0]
    a, b = x.clone(), gate.clone()
    expected = x * torch.sigmoid(gate)
    actual = sigmoid_mul(x, gate)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.equal(x, a) and torch.equal(gate, b)
    assert not torch._C._overlaps(actual, x) and not torch._C._overlaps(actual, gate)


class CudaValue:
    is_cuda = True
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    requires_grad = False

    def __init__(self, value, *, pointer=256):
        self.value, self.pointer = value, pointer
        self.shape = value.shape
        self.fallback_calls = 0

    def is_contiguous(self):
        return self.value.is_contiguous()

    def numel(self):
        return self.value.numel()

    def data_ptr(self):
        return self.pointer

    def sigmoid(self):
        return self.value.sigmoid()

    def __mul__(self, other):
        self.fallback_calls += 1
        return self.value * other


@pytest.fixture
def native_protocol(monkeypatch):
    events, mode = [], SimpleNamespace(capturing=False, covered=True, error=False)
    x = CudaValue(torch.ones(2, 16, dtype=torch.bfloat16))
    gate = CudaValue(torch.full((2, 16), 2, dtype=torch.bfloat16))
    owned = torch.full((2, 16), 3, dtype=torch.bfloat16)

    def warm():
        events.append("warm")
        return object()

    def op(a, b):
        assert a is x and b is gate
        events.append("native")
        if mode.error:
            raise RuntimeError("native failure")
        return owned

    @contextmanager
    def stream():
        events.append("stream-enter")
        yield
        events.append("stream-exit")

    ops = SimpleNamespace(
        covered=lambda a, b: mode.covered,
        _jit_mla_output_gate_module=warm,
        kimi_k3_mla_output_gate=op,
    )
    monkeypatch.setattr(native, "_sigmoid_mul_ops", lambda: (ops, stream))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: mode.capturing)
    return x, gate, owned, events, mode


def test_native_uses_public_owned_operation_and_warms_before_capture(native_protocol):
    x, gate, owned, events, mode = native_protocol
    a, b = x.value.clone(), gate.value.clone()
    assert sigmoid_mul(x, gate) is owned
    mode.capturing = True
    assert sigmoid_mul(x, gate) is owned
    assert events == [
        "warm",
        "stream-enter",
        "native",
        "stream-exit",
        "stream-enter",
        "native",
        "stream-exit",
    ]
    assert x.fallback_calls == 0
    assert torch.equal(x.value, a) and torch.equal(gate.value, b)
    assert not torch._C._overlaps(owned, x.value) and not torch._C._overlaps(owned, gate.value)


def test_cold_native_capture_rejects_before_compilation_or_launch(native_protocol):
    x, gate, _, events, mode = native_protocol
    mode.capturing = True
    with pytest.raises(RuntimeError, match="Warm native"):
        sigmoid_mul(x, gate)
    assert events == [] and x.fallback_calls == 0


def test_sdk_decline_keeps_original_expression_without_warmup(native_protocol):
    x, gate, _, events, mode = native_protocol
    mode.covered = False
    result = sigmoid_mul(x, gate)
    assert torch.equal(result, x.value * gate.value.sigmoid())
    assert events == [] and x.fallback_calls == 1


def test_native_error_propagates_once_without_fallback(native_protocol):
    x, gate, _, events, mode = native_protocol
    mode.error = True
    with pytest.raises(RuntimeError, match="native failure"):
        sigmoid_mul(x, gate)
    assert events.count("native") == 1 and x.fallback_calls == 0


def test_unaligned_input_keeps_the_expression_before_sdk(monkeypatch):
    monkeypatch.setattr(
        native, "_sigmoid_mul_ops", lambda: pytest.fail("unaligned native selection")
    )
    x = CudaValue(torch.ones(2, 16, dtype=torch.bfloat16), pointer=258)
    gate = CudaValue(torch.ones(2, 16, dtype=torch.bfloat16))
    assert torch.equal(sigmoid_mul(x, gate), x.value * gate.value.sigmoid())
    assert x.fallback_calls == 1


def test_mla_finish_calls_generic_gate_once(monkeypatch):
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    values = torch.ones(2, 3, 8, dtype=torch.bfloat16)
    gate = torch.zeros(2, 24, dtype=torch.bfloat16)
    owned, calls = torch.full((2, 24), 7, dtype=torch.bfloat16), []

    def apply(actual, actual_gate):
        calls.append((actual, actual_gate))
        assert actual_gate is gate and actual.shape == gate.shape
        return owned

    monkeypatch.setattr(mla, "sigmoid_mul", apply)
    layer = object.__new__(mla.MLALayer)
    layer._cp = SimpleNamespace(size=1)
    layer._output_gate = True
    layer.g_proj = SimpleNamespace(forward=lambda actual: gate if actual is x else None)
    layer.o_proj = SimpleNamespace(forward=lambda actual, residual=None: actual)
    assert layer.finish(x, MLAOutput(values)) is owned
    assert len(calls) == 1


def test_missing_optional_sdk_keeps_the_expression_before_selection(native_protocol, monkeypatch):
    x, gate, _, events, _ = native_protocol

    def missing():
        raise ImportError("optional SDK unavailable")

    monkeypatch.setattr(native, "_sigmoid_mul_ops", missing)
    assert torch.equal(sigmoid_mul(x, gate), x.value * gate.value.sigmoid())
    assert events == [] and x.fallback_calls == 1


def test_selected_jit_import_error_propagates_without_fallback(native_protocol, monkeypatch):
    x, gate, _, events, _ = native_protocol
    ops, _ = native._sigmoid_mul_ops()

    def failed_jit():
        events.append("jit-failure")
        raise ImportError("selected JIT dependency failed")

    monkeypatch.setattr(ops, "_jit_mla_output_gate_module", failed_jit)
    with pytest.raises(ImportError, match="selected JIT"):
        sigmoid_mul(x, gate)
    assert events == ["jit-failure"] and x.fallback_calls == 0


def test_native_count_declines_uint32_overflow_before_sdk(monkeypatch):
    monkeypatch.setattr(
        native, "_sigmoid_mul_ops", lambda: pytest.fail("overflow native selection")
    )
    x = CudaValue(torch.ones(2, 16, dtype=torch.bfloat16))
    gate = CudaValue(torch.ones(2, 16, dtype=torch.bfloat16))
    monkeypatch.setattr(x, "numel", lambda: 1 << 32)
    assert torch.equal(sigmoid_mul(x, gate), x.value * gate.value.sigmoid())
    assert x.fallback_calls == 1
