"""Host ABI fixtures: preserve caller routing precision without touching CUDA."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from minisgl.moe.flashinfer import FlashInferMXFP4Experts


class DeviceInput(torch.Tensor):
    # Only bypass the executor's device guard; storage and all operations stay CPU.
    @property
    def is_cuda(self):
        return True


@pytest.fixture
def executor(monkeypatch):
    def no_cuda(*args, **kwargs):
        raise AssertionError("This host ABI test must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "init", no_cuda)
    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        kwargs["output"].zero_()

    modules = {
        name: ModuleType(name)
        for name in (
            "flashinfer",
            "flashinfer.fused_moe",
            "flashinfer.fused_moe.core",
            "flashinfer.tllm_enums",
        )
    }
    modules["flashinfer"].trtllm_fp4_block_scale_routed_moe = run
    modules["flashinfer.fused_moe.core"].ActivationType = SimpleNamespace(
        Swiglu=SimpleNamespace(value=1), Situ=SimpleNamespace(value=2)
    )
    modules["flashinfer.tllm_enums"].RoutingMethodType = SimpleNamespace(
        TopK=SimpleNamespace(value=0)
    )
    modules["flashinfer.tllm_enums"].SfLayout = SimpleNamespace(layout_linear=0)
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    obj = object.__new__(FlashInferMXFP4Experts)
    obj._hidden, obj._intermediate = 256, 128
    obj._precision, obj._activation = "bf16", "silu"
    obj._global_experts, obj._experts, obj._start = 8, 2, 0
    for name in ("_w13", "_w13_scale", "_w2", "_w2_scale", "_alpha", "_linear"):
        setattr(obj, name, None)
    return obj, calls


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strided", [False, True])
def test_route_dtype_reaches_sdk_unchanged(executor, dtype, strided):
    obj, calls = executor
    x = torch.zeros(3, 256, dtype=torch.bfloat16).as_subclass(DeviceInput)
    ids = torch.arange(6, dtype=torch.int64).reshape(3, 2)
    columns = 4 if strided else 2
    weights = torch.linspace(0.1, 0.9, 3 * columns, dtype=dtype).reshape(3, columns)[:, :2]
    result = obj.forward(x, weights, ids)
    routed_ids, routed_weights = calls[0]["topk_ids"]
    assert routed_weights.dtype == dtype
    assert routed_weights.is_contiguous()
    torch.testing.assert_close(routed_weights, weights, rtol=0, atol=0)
    if not strided:
        assert routed_weights.data_ptr() == weights.data_ptr()
    assert routed_ids.dtype == torch.int32
    torch.testing.assert_close(routed_ids, ids.to(torch.int32), rtol=0, atol=0)
    assert result is calls[0]["output"]
    assert result.dtype == torch.bfloat16


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
def test_unsupported_route_dtype_is_rejected_before_sdk(executor, dtype):
    obj, calls = executor
    x = torch.zeros(3, 256, dtype=torch.bfloat16).as_subclass(DeviceInput)
    ids = torch.zeros(3, 2, dtype=torch.int32)
    with pytest.raises(ValueError, match="route weights must be bfloat16 or float32"):
        obj.forward(x, torch.ones(3, 2, dtype=dtype), ids)
    assert calls == []


def test_prepared_routes_reach_sdk_without_duplicate_routing_or_quantization(executor, monkeypatch):
    from minisgl.moe import sgl
    from minisgl.moe.fused import FusedMoe
    from minisgl.moe.routing import GroupedRouting

    obj, calls = executor
    obj._precision = "mxfp8"
    x = torch.zeros(3, 256, dtype=torch.float32).as_subclass(DeviceInput)
    routing = GroupedRouting(torch.zeros(3, 8), torch.zeros(8), 2, 1, 1, "sigmoid", True, 2.5)
    packed = torch.zeros(3, 2, dtype=torch.int32)
    quantized = torch.empty(3, 256, dtype=torch.float8_e4m3fn)
    scales = torch.zeros(3, 2, dtype=torch.int32)

    def prepare(actual_x, actual_routing):
        assert actual_x is x and actual_routing is routing
        return None, None, packed, quantized, scales

    monkeypatch.setattr(sgl, "fused_route_quant", prepare)
    backend = FusedMoe()
    monkeypatch.setattr(backend, "route", lambda *a, **kw: pytest.fail("duplicate routing"))
    result = backend.forward_experts(obj, x, routing)
    assert len(calls) == 1 and result is calls[0]["output"]
    assert calls[0]["topk_ids"] is packed
    assert calls[0]["hidden_states"] is quantized
    assert calls[0]["hidden_states_scale"].data_ptr() == scales.data_ptr()
    assert calls[0]["hidden_states_scale"].dtype == torch.float8_e4m3fn
    assert calls[0]["hidden_states_scale"].shape == (3, 8)
    assert calls[0]["top_k"] == 2


def test_uncovered_preparation_keeps_unpacked_fp32_routes(executor, monkeypatch):
    from minisgl.moe import sgl
    from minisgl.moe.fused import FusedMoe
    from minisgl.moe.routing import GroupedRouting

    obj, calls = executor
    obj._precision = "mxfp8"
    x = torch.zeros(3, 256, dtype=torch.float32).as_subclass(DeviceInput)
    routing = GroupedRouting(torch.zeros(3, 8), None, 2, 1, 1, "softmax", True, 1.0)
    weights = torch.full((3, 2), 0.5001, dtype=torch.float32)
    ids = torch.zeros(3, 2, dtype=torch.int32)
    backend = FusedMoe()
    monkeypatch.setattr(backend, "route", lambda *a, **kw: (weights, ids))
    monkeypatch.setattr(sgl, "fused_route_quant", lambda *a: None)
    quantized = torch.empty(3, 256, dtype=torch.float8_e4m3fn)
    scales = torch.zeros(3, 2, dtype=torch.int32)
    quant = ModuleType("sglang.kernels.ops.quantization.per_token_group_quant")
    quant.per_token_group_quant = lambda *a, **kw: (quantized, scales)
    monkeypatch.setitem(sys.modules, quant.__name__, quant)
    backend.forward_experts(obj, x, routing)
    assert calls[0]["topk_ids"][1] is weights
    assert calls[0]["topk_ids"][1].dtype == torch.float32


@pytest.mark.parametrize(
    "condition",
    ["covered", "cpu", "bias", "scoring", "groups", "selected_groups", "uncovered", "unavailable"],
)
def test_native_preparation_respects_semantics_and_sdk_coverage(executor, monkeypatch, condition):
    from contextlib import contextmanager
    from dataclasses import replace

    from minisgl.moe.routing import GroupedRouting
    from minisgl.moe.sgl import fused_route_quant

    x = torch.zeros(3, 256, dtype=torch.float32).as_subclass(DeviceInput)
    routing = GroupedRouting(
        torch.zeros(3, 8).as_subclass(DeviceInput),
        torch.zeros(8),
        2,
        1,
        1,
        "sigmoid",
        True,
        2.5,
    )
    if condition == "cpu":
        x = x.as_subclass(torch.Tensor)
    elif condition == "bias":
        routing = replace(routing, bias=None)
    elif condition == "scoring":
        routing = replace(routing, scoring_func="softmax")
    elif condition == "groups":
        routing = replace(routing, num_groups=2)
    elif condition == "selected_groups":
        routing = replace(routing, topk_groups=2)
    events = []
    result = object()

    def kernel(*a):
        assert events == ["stream-enter"]
        assert a == (routing.logits, routing.bias, x, 2, True, 2.5, True)
        return result

    module = ModuleType("sglang.kernels.ops.moe")
    module.moe_route_quant_fused = SimpleNamespace(
        covered=lambda *a: condition != "uncovered",
        available=lambda: condition != "unavailable",
        route_quant_fused=kernel,
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    ffi = ModuleType("tvm_ffi")

    @contextmanager
    def stream():
        events.append("stream-enter")
        yield
        events.append("stream-exit")

    ffi.use_torch_stream = stream
    monkeypatch.setitem(sys.modules, ffi.__name__, ffi)
    actual = fused_route_quant(x, routing)
    if condition == "covered":
        assert actual is result and events == ["stream-enter", "stream-exit"]
    else:
        assert actual is None and events == []
