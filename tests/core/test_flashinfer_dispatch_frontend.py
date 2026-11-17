"""CPU ABI spies only: no native communication, quantization or expert math."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from minisgl.moe.flashinfer import FlashInferMXFP4Experts


class DeviceInput(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def device(tensor):
    return tensor.as_subclass(DeviceInput)


@pytest.fixture
def executor(monkeypatch):
    def no_cuda(*args, **kwargs):
        raise AssertionError("CPU frontend fixtures must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "init", no_cuda)
    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    obj = object.__new__(FlashInferMXFP4Experts)
    obj._hidden, obj._intermediate = 256, 128
    obj._precision, obj._activation = "mxfp8", "situ"
    obj._global_experts, obj._experts, obj._start = 8, 2, 4
    for name in ("_w13", "_w13_scale", "_w2", "_w2_scale", "_alpha", "_linear"):
        setattr(obj, name, None)
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
    return obj, calls


def inputs(rows=3):
    x = device(torch.zeros(rows, 256, dtype=torch.float32))
    ids = device(torch.tensor([4, 7], dtype=torch.int32).repeat(rows, 1))
    weights = torch.full((rows, 2), 0.5001, dtype=torch.float32)
    bits = weights.bfloat16().view(torch.int16).to(torch.int32) & 0xFFFF
    packed = device((ids << 16) | bits)
    q = device(torch.zeros(rows, 256, dtype=torch.float8_e4m3fn))
    scales = device(torch.arange(rows * 2, dtype=torch.int32).view(rows, 2))
    return x, ids, weights, packed, q, scales


@pytest.mark.parametrize("rows", [1, 48, 64])
def test_fused_same_call_ids_carrier_and_scale_bytes(executor, monkeypatch, rows):
    from minisgl.moe import sgl

    obj, calls = executor
    x, ids, weights, packed, q, scales = inputs(rows)
    native_calls = []

    def native(actual_x, routing):
        native_calls.append(actual_x.shape[0])
        return weights, ids, packed, q, scales

    monkeypatch.setattr(sgl, "fused_route_quant", native)
    routing = SimpleNamespace(topk=2, select=lambda backend: pytest.fail("second router"))
    actual_ids, payloads = obj.prepare_dispatch_inputs(object(), x, routing)
    assert actual_ids is ids and payloads[0] is q and payloads[2] is packed
    assert payloads[1].dtype == torch.float8_e4m3fn
    assert payloads[1].shape == (rows, 8)
    assert payloads[1].data_ptr() == scales.data_ptr()
    torch.testing.assert_close(
        payloads[1].view(torch.uint8), scales.view(torch.uint8), rtol=0, atol=0
    )
    # Owner slicing is explicitly AFTER the same full-batch frontend choice.
    owned = tuple(value[1::8].contiguous() for value in payloads)
    assert owned[0].shape[0] == len(range(1, rows, 8))
    assert native_calls == [rows] and calls == []


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_uncovered_preserves_unpacked_weight_abi_and_one_quantization(executor, monkeypatch, dtype):
    from minisgl.moe import sgl

    obj, calls = executor
    x, ids, weights, packed, q, scales = inputs(65)
    weights = device(weights.to(dtype))
    events = []
    monkeypatch.setattr(sgl, "fused_route_quant", lambda x, r: events.append(("fused", len(x))))

    def select(backend):
        events.append("route")
        return weights, ids

    def quant(actual_x, **kwargs):
        assert actual_x is x and kwargs == {"group_size": 32, "scale_ue8m0": True}
        events.append("quant")
        return q, scales

    module = ModuleType("sglang.kernels.ops.quantization.per_token_group_quant")
    module.per_token_group_quant = quant
    monkeypatch.setitem(sys.modules, module.__name__, module)
    actual_ids, payloads = obj.prepare_dispatch_inputs(object(), x, SimpleNamespace(select=select))
    assert actual_ids is ids and payloads[2] is weights and payloads[2].dtype == dtype
    torch.testing.assert_close(payloads[2], weights, rtol=0, atol=0)
    assert events == [("fused", 65), "route", "quant"] and calls == []


def receive(dtype=torch.int32):
    _, _, _, packed, q, scales = inputs(4)
    ids = device(torch.tensor([[[4, 7], [-1, -1]], [[0, 5], [-1, -1]]], dtype=torch.int32))
    if dtype == torch.int32:
        carrier = ((ids << 16) | (packed.view(2, 2, 2) & 0xFFFF)).clone()
        carrier[0, 1] = 1234567
        carrier[1, 1] = -7654321
    else:
        carrier = device(torch.full((2, 2, 2), 0.5001, dtype=dtype))
        carrier[0, 1] = float("nan")
        carrier[1, 1] = float("nan")
    return ids, (q.view(2, 2, 256), scales.view(torch.float8_e4m3fn).view(2, 2, 8), carrier)


@pytest.mark.parametrize("dtype", [torch.int32, torch.bfloat16, torch.float32])
def test_receive_padding_only_preserves_real_and_foreign_bits(executor, monkeypatch, dtype):
    obj, calls = executor
    ids, payloads = receive(dtype)
    snapshots = [value.view(torch.uint8).clone() for value in (ids, *payloads)]
    monkeypatch.setattr(obj, "_prepare_inputs", lambda *a: pytest.fail("second quantization"))
    monkeypatch.setattr(obj, "_prepare_routed_inputs", lambda *a: pytest.fail("second router"))
    result = obj.forward_dispatched(ids, payloads)
    assert len(calls) == 1 and result is calls[0]["output"]
    kwargs = calls[0]
    assert kwargs["hidden_states"].data_ptr() == payloads[0].data_ptr()
    assert kwargs["hidden_states_scale"].data_ptr() == payloads[1].data_ptr()
    assert kwargs["local_expert_offset"] == 4 and kwargs["local_num_experts"] == 2
    assert kwargs["routing_bias"] is None and kwargs["routed_scaling_factor"] is None
    assert kwargs["do_finalize"] is True
    native_routes = kwargs["topk_ids"]
    if dtype == torch.int32:
        actual = native_routes
        assert actual.dtype == torch.int32
        assert torch.equal(actual[[1, 3]], torch.full((2, 2), -65536, dtype=torch.int32))
    else:
        actual_ids, actual = native_routes
        assert actual.dtype == dtype
        torch.testing.assert_close(actual_ids, ids.view(4, 2), rtol=0, atol=0)
        assert torch.equal(actual[[1, 3]], torch.zeros(2, 2, dtype=dtype))
    # Rows 0/2 include foreign experts 7/0; every original carrier bit survives.
    torch.testing.assert_close(
        actual[[0, 2]].view(torch.uint8),
        payloads[2].view(4, 2)[[0, 2]].view(torch.uint8),
        rtol=0,
        atol=0,
    )
    for value, snapshot in zip((ids, *payloads), snapshots):
        torch.testing.assert_close(value.view(torch.uint8), snapshot, rtol=0, atol=0)


@pytest.mark.parametrize(
    "invalid",
    [
        "precision",
        "ids_dtype",
        "ids_shape",
        "scale_dtype",
        "scale_shape",
        "scale_stride",
        "carrier_dtype",
        "device",
        "packed_expert_range",
        "requires_grad",
    ],
)
def test_unsupported_receive_declines_before_native(executor, invalid):
    obj, calls = executor
    ids, payloads = receive()
    q, scale, carrier = payloads
    if invalid == "precision":
        obj._precision = "bf16"
    elif invalid == "ids_dtype":
        ids = ids.to(torch.int64)
    elif invalid == "ids_shape":
        ids = ids.view(4, 2)
        q, scale, carrier = q.view(4, 256), scale.view(4, 8), carrier.view(4, 2)
    elif invalid == "scale_dtype":
        scale = scale.to(torch.float32)
    elif invalid == "scale_shape":
        scale = scale[..., :4]
    elif invalid == "scale_stride":
        scale = device(torch.empty(2, 2, 16, dtype=torch.float8_e4m3fn))[..., ::2]
    elif invalid == "carrier_dtype":
        carrier = carrier.to(torch.int64)
    elif invalid == "device":
        ids = ids.as_subclass(torch.Tensor)
    elif invalid == "packed_expert_range":
        obj._global_experts = 32769
    else:
        carrier = device(torch.ones(2, 2, 2, dtype=torch.float32, requires_grad=True))
    with pytest.raises((ValueError, NotImplementedError)):
        obj.forward_dispatched(ids, (q, scale, carrier))
    assert calls == []


def test_unsupported_preparation_declines_before_routing(executor, monkeypatch):
    obj, calls = executor
    obj._precision = "bf16"
    monkeypatch.setattr(obj, "_prepare_routed_inputs_with_ids", lambda *a: pytest.fail("routing"))
    with pytest.raises(NotImplementedError, match="MXFP8"):
        obj.prepare_dispatch_inputs(object(), object(), object())
    assert calls == []


def test_ep1_preparation_retains_original_four_tuple(executor, monkeypatch):
    from minisgl.moe import sgl

    obj, calls = executor
    x, ids, weights, packed, q, scales = inputs(48)
    monkeypatch.setattr(sgl, "fused_route_quant", lambda *a: (weights, ids, packed, q, scales))
    routing = SimpleNamespace(topk=2, select=lambda *a: pytest.fail("second router"))
    prepared = obj.prepare_inputs(object(), x, routing)
    assert len(prepared) == 4 and prepared[0] is q and prepared[2] is packed
    assert prepared[3] == 2 and prepared[1].data_ptr() == scales.data_ptr()
    obj.try_forward_routed(x, routing)
    assert len(calls) == 1 and calls[0]["topk_ids"] is packed


def test_global_empty_preparation_skips_before_any_frontend(executor, monkeypatch):
    obj, calls = executor
    monkeypatch.setattr(obj, "_prepare_routed_inputs_with_ids", lambda *a: pytest.fail("routing"))
    with pytest.raises(ValueError, match="empty global batch"):
        obj.prepare_dispatch_inputs(object(), device(torch.empty(0, 256)), object())
    assert calls == []


def test_empty_receive_geometry_never_launches_native(executor):
    obj, calls = executor
    ids = device(torch.empty(2, 0, 2, dtype=torch.int32))
    q = device(torch.empty(2, 0, 256, dtype=torch.float8_e4m3fn))
    scales = device(torch.empty(2, 0, 8, dtype=torch.float8_e4m3fn))
    carrier = device(torch.empty(2, 0, 2, dtype=torch.int32))
    with pytest.raises(ValueError, match="nonempty"):
        obj.forward_dispatched(ids, (q, scales, carrier))
    assert calls == []
