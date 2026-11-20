"""CPU elementwise contracts and native protocol checks; no CUDA math."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from minisgl.kernel import elementwise as native
from minisgl.layers.elementwise import add3
from minisgl.layers.latent_moe import GatedMLP, LatentMoE
from minisgl.moe import create_moe_backend


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("CPU add3 test initialized CUDA")

    monkeypatch.setattr(torch.cuda, "init", forbid)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)
    native._warm_add3.cache_clear()
    yield
    native._warm_add3.cache_clear()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_cpu_preserves_association_and_storage_rounding(monkeypatch, dtype):
    monkeypatch.setattr(native, "_add3_ops", lambda: pytest.fail("CPU loaded SDK"))
    a, b, c = [torch.full((2, 16), value, dtype=dtype) for value in (256, 1, -256)]
    snapshots = [value.clone() for value in (a, b, c)]
    result = add3(a, b, c)
    assert torch.equal(result, (a + b) + c)
    assert result.eq(0 if dtype == torch.bfloat16 else 1).all()
    if dtype == torch.bfloat16:
        assert not torch.equal(result, (a.float() + b.float() + c.float()).bfloat16())
        assert not torch.equal(result, a + (b + c))
    assert all(torch.equal(value, snapshot) for value, snapshot in zip((a, b, c), snapshots))
    assert all(not torch._C._overlaps(result, value) for value in (a, b, c))


def test_portable_broadcast_stride_empty_and_optional_addend():
    a = torch.arange(32.0).reshape(2, 16)[:, ::2]
    b, c = a[:1], torch.ones(8)
    assert torch.equal(add3(a, b, c), (a + b) + c)
    assert add3(a[:0], b[:0], c).shape == (0, 8)
    assert torch.equal(add3(a, b), a + b)


class CudaValue:
    is_cuda = True
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    requires_grad = False

    def __init__(self, value, pointer=256):
        self.value, self.pointer = value, pointer
        self.shape = value.shape
        self.fallback_calls = 0

    def data_ptr(self):
        return self.pointer

    def __add__(self, other):
        self.fallback_calls += 1
        return self.value + (other.value if isinstance(other, CudaValue) else other)

    def __radd__(self, other):
        return other + self.value


@pytest.fixture
def protocol(monkeypatch):
    events, mode = [], SimpleNamespace(covered=True, capture=False, error=False)
    values = tuple(CudaValue(torch.full((2, 16), x, dtype=torch.bfloat16)) for x in (256, 1, -256))
    owned = torch.zeros((2, 16), dtype=torch.bfloat16)

    def warm():
        events.append("warm")

    def op(a, b, c, *, prefetch_bc):
        assert (a, b, c) == values and prefetch_bc is False
        events.append("native")
        if mode.error:
            raise RuntimeError("selected native failure")
        return owned

    @contextmanager
    def stream():
        events.append("stream")
        yield

    ops = SimpleNamespace(covered=lambda *args: mode.covered, _jit_add3_module=warm, add3=op)
    monkeypatch.setattr(native, "_add3_ops", lambda: (ops, stream))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: mode.capture)
    return values, owned, events, mode, ops


def test_native_public_operation_once_owned_and_warm_capture(protocol):
    values, owned, events, mode, _ = protocol
    assert add3(*values) is owned
    mode.capture = True
    assert add3(*values) is owned
    assert events == ["warm", "stream", "native", "stream", "native"]
    assert values[0].fallback_calls == 0
    assert all(not torch._C._overlaps(owned, value.value) for value in values)


def test_cold_capture_and_selected_errors_do_not_retry(protocol, monkeypatch):
    values, _, events, mode, ops = protocol
    mode.capture = True
    with pytest.raises(RuntimeError, match="Warm native add3"):
        add3(*values)
    assert events == [] and values[0].fallback_calls == 0
    mode.capture = False
    mode.error = True
    with pytest.raises(RuntimeError, match="selected native failure"):
        add3(*values)
    assert events.count("native") == 1 and values[0].fallback_calls == 0
    native._warm_add3.cache_clear()

    def failed_jit():
        raise ImportError("selected JIT failure")

    monkeypatch.setattr(ops, "_jit_add3_module", failed_jit)
    with pytest.raises(ImportError, match="selected JIT failure"):
        add3(*values)
    assert events.count("native") == 1 and values[0].fallback_calls == 0


def test_sdk_decline_or_optional_import_keeps_original_expression(protocol, monkeypatch):
    values, _, events, mode, _ = protocol
    mode.covered = False
    assert torch.equal(add3(*values), (values[0].value + values[1].value) + values[2].value)
    assert events == [] and values[0].fallback_calls == 1

    def missing():
        raise ImportError("optional SDK missing")

    monkeypatch.setattr(native, "_add3_ops", missing)
    assert add3(*values).eq(0).all()
    assert values[0].fallback_calls == 2


@pytest.mark.parametrize("bad", ["alignment", "device", "grad", "dtype"])
def test_unsupported_preselection_keeps_portable_path(protocol, monkeypatch, bad):
    values, _, events, _, _ = protocol
    if bad == "alignment":
        values[1].pointer = 272  # 16B, but not the SDK's widest 32B vector.
    elif bad == "device":
        values[1].device = torch.device("cuda:1")
    elif bad == "grad":
        values[1].requires_grad = True
    else:
        values[1].dtype = torch.float32
    monkeypatch.setattr(native, "_add3_ops", lambda: pytest.fail("invalid native selection"))
    assert add3(*values).eq(0).all() and events == []


def test_projection_preserves_fp32_output_and_casts_shared_before_prefix():
    backend = create_moe_backend("fused")
    x = torch.full((2, 4), 1.003, dtype=torch.float32)
    shared = torch.full_like(x, 0.07)
    residual = torch.full((2, 4), -0.2, dtype=torch.bfloat16)
    result = backend.epilogue.project_outputs(
        SimpleNamespace(), x, shared, output_dtype=torch.bfloat16, residual=residual
    )
    assert result.dtype == torch.float32
    assert torch.equal(result, (x + shared.bfloat16()) + residual)


def test_dense_mlp_optional_residual_keeps_default_and_adds_once():
    layer = object.__new__(GatedMLP)
    partial = torch.ones(2, 4)
    layer.forward_local = lambda x: partial
    layer._tp = SimpleNamespace(all_reduce=lambda value: value * 2)
    assert layer.forward(None).eq(2).all()
    assert layer.forward(None, residual=torch.full_like(partial, 3)).eq(5).all()
    assert partial.eq(1).all()


@pytest.mark.parametrize("case", ["exact", "broadcast", "empty"])
def test_public_latent_optional_residual_survives_flattening(monkeypatch, case):
    layer = object.__new__(LatentMoE)
    layer._front = layer._output = layer._reserved_key = None
    layer.shared_experts = layer.routed_expert_down_proj = None
    layer.routed_expert_norm = layer.routed_expert_up_proj = None
    layer._backend = create_moe_backend("fused")
    monkeypatch.setattr(layer._backend, "supports_deferred_output", lambda *args: False)
    layer._tp = SimpleNamespace(all_reduce=lambda value: value)
    route = SimpleNamespace(topk=1, select=lambda backend: (None, None))
    layer.gate = SimpleNamespace(weight=torch.zeros(1, 4), routing=lambda logits: route)
    layer._expert_executor = SimpleNamespace(forward=lambda x, weights, ids: x * 2)
    x = torch.arange(24.0).reshape(2, 3, 4)
    if case == "empty":
        x = x[:0]
    residual = torch.ones(4) if case == "broadcast" else torch.ones_like(x)
    assert torch.equal(layer.forward(x, residual=residual), x * 2 + residual)


def test_decoder_passes_head_once_to_mlp():
    from minisgl.models.kimi_k3 import KimiDecoderLayer

    layer = object.__new__(KimiDecoderLayer)
    layer._block_size = None
    layer.input_layernorm = layer.post_attention_layernorm = SimpleNamespace(forward=lambda x: x)
    layer._attention = lambda x, *args, **kwargs: x + 3
    calls = []

    def mlp(x, *, residual):
        calls.append((x, residual))
        assert x is residual
        return x + residual  # Explicitly consume the head inside the MLP boundary.

    layer.mlp = SimpleNamespace(forward=mlp)
    result = layer.forward(torch.ones(2, 4), [], None, None, None)
    assert result.eq(8).all() and len(calls) == 1
