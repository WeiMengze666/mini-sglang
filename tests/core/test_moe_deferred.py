"""CPU-only deferred FI/native-consumer ABI and producer scheduling checks."""

import sys
import weakref
from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed.impl import (
    DistributedCommunicator,
    DistributedImpl,
    FallbackDistributedImpl,
)
from minisgl.distributed.info import DistributedInfo
from minisgl.distributed.mnnvl import MNNVLDistributedImpl
from minisgl.distributed.parallel import TensorParallel
from minisgl.layers.norm import RMSNormEpilogue, TorchRMSNorm
from minisgl.moe import create_moe_backend
from minisgl.moe.base import DeferredMoEOutput
from test_collective_outputs import cuda_tensor
from test_collective_outputs import host as collective_host
from test_flashinfer_route_dtype import DeviceInput
from test_flashinfer_route_dtype import executor as existing_fi_executor
from test_moe_overlap import streams as overlap_streams

host = collective_host
fi_executor = existing_fi_executor
streams = overlap_streams


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("CPU deferred test initialized CUDA")

    monkeypatch.setattr(torch.cuda, "init", forbid)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fi_native_triple_order_routed_precision_and_explicit_output(
    fi_executor, monkeypatch, dtype
):
    obj, calls = fi_executor
    obj._precision, obj._activation = "mxfp8", "situ"
    obj._w2 = torch.empty(1).as_subclass(DeviceInput)
    x = torch.ones(3, 256, dtype=torch.float32).as_subclass(DeviceInput)
    out = torch.full((3, 256), -17, dtype=torch.bfloat16)
    values = torch.ones(8, 256, dtype=torch.bfloat16)
    weights = torch.full((3, 2), 0.75, dtype=dtype)
    indices = torch.arange(6, dtype=torch.int32)
    routes = (torch.zeros(3, 2, dtype=torch.int32), weights)
    quantized = torch.empty(3, 256, dtype=torch.float8_e4m3fn)
    scale = torch.empty(3, 8, dtype=torch.float8_e4m3fn)
    routing = object()
    prepared = []

    def prepare(backend, actual_x, actual_routing):
        assert actual_x is x and actual_x.dtype == torch.float32
        assert actual_routing is routing
        prepared.append(1)
        return quantized, scale, routes, 2

    def native(**kwargs):
        calls.append(kwargs)
        assert kwargs["do_finalize"] is False and kwargs["output"] is out
        assert kwargs["hidden_states"] is quantized
        assert kwargs["topk_ids"] is routes
        return values, weights, indices  # The actual FI ABI orders weights before indices.

    monkeypatch.setattr(obj, "prepare_inputs", prepare)
    monkeypatch.setattr(sys.modules["flashinfer"], "trtllm_fp4_block_scale_routed_moe", native)
    result = create_moe_backend("fused").forward_experts(obj, x, routing, out=out, deferred=True)
    assert isinstance(result, DeferredMoEOutput)
    assert result.values is values and result.indices is indices and result.weights is weights
    assert result.weights.dtype == dtype and prepared == [1] and len(calls) == 1
    assert out.eq(-17).all()  # The FI call did not publish an already-finalized output.


def test_ep_or_non_situ_declines_deferred_before_fi(fi_executor, monkeypatch):
    from minisgl.distributed import layout

    obj, calls = fi_executor
    obj._w2 = torch.empty(1).as_subclass(DeviceInput)
    backend = create_moe_backend("fused")
    assert not backend.supports_deferred_output(obj, torch.bfloat16)
    obj._precision, obj._activation = "mxfp8", "situ"
    assert backend.supports_deferred_output(obj, torch.bfloat16)
    monkeypatch.setattr(layout, "get_parallel_layout", lambda: SimpleNamespace(ep_size=2))
    assert not backend.supports_deferred_output(obj, torch.bfloat16)
    with pytest.raises(NotImplementedError, match="plain[ -]TP"):
        backend.forward_experts(
            obj, None, None, out=torch.empty(3, 256, dtype=torch.bfloat16), deferred=True
        )
    assert calls == []


@pytest.mark.parametrize("supported", [False, True])
def test_prelaunch_cap_shared_join_and_exactly_one_consumer(streams, supported):
    events, _, side, active = streams
    backend = create_moe_backend("fused")
    backend._shared_stream = side
    packed = torch.empty(12, dtype=torch.bfloat16)
    latent, shared_out = packed[:8].view(2, 4), packed[8:].view(2, 2)
    norm = TorchRMSNorm(4)
    norm.weight = torch.tensor([0.5, 1, 1.5, 2], dtype=torch.bfloat16)
    normalizations, lifetime = [], []
    original = norm.forward

    def normalize(value):
        normalizations.append(1)
        return original(value)

    norm.forward = normalize

    def cap(out, *, norm, topk):
        events.append("cap")
        assert out is latent and isinstance(norm, RMSNormEpilogue) and topk == 2
        return supported

    def experts(out):
        assert not supported
        events.append("finalized:" + active[0].name)
        return out.fill_(2)

    def deferred(out):
        assert supported
        events.append("deferred:" + active[0].name)
        values = torch.full((4, 4), 2, dtype=torch.bfloat16)
        indices = torch.arange(4, dtype=torch.int32)
        weights = torch.ones(2, 2, dtype=torch.float32)
        lifetime.extend(weakref.ref(t) for t in (values, indices, weights))
        return DeferredMoEOutput(values, indices, weights)

    def shared(out):
        events.append("shared:" + active[0].name)
        return out.fill_(3)

    def reduce(value, *, norm=None, output_key=None, overlap=False):
        assert output_key == "role"
        events.append(("shared_AR:" if overlap else "latent_AR:") + active[0].name)
        assert overlap or not supported
        result = value * 2
        return norm(result) if norm else result

    def weighted(values, indices, weights, *, out, norm):
        events.append("weighted:" + active[0].name)
        assert all(r() is t for r, t in zip(lifetime, (values, indices, weights)))
        assert out is latent and weights.dtype == torch.float32
        out.copy_(norm.normalizer.forward(torch.full_like(out, 4)))
        return out

    parallel = SimpleNamespace(
        supports_weighted_all_reduce=cap,
        weighted_all_reduce=weighted,
        all_reduce_residual=reduce,
    )
    result, tail = backend.forward_outputs(
        parallel,
        experts,
        shared,
        expert_out=latent,
        shared_out=shared_out,
        packed=packed,
        output_key="role",
        norm=norm,
        output_dtype=torch.bfloat16,
        deferred_producer=deferred,
        topk=2,
    )
    assert events == [
        "cap",
        "wait:shared<-main",
        ("deferred:" if supported else "finalized:") + "main",
        "enter:shared",
        "shared:shared",
        "shared_AR:shared",
        "exit:shared",
        "wait:main<-shared",
        ("weighted:" if supported else "latent_AR:") + "main",
    ]
    assert len(normalizations) == 1 and tail.eq(6).all()
    if supported:
        assert result is latent
    # The finalized tensor, not DeferredMoEOutput, reaches the optional prefix tail.
    prefix = torch.full_like(tail, 7)
    projection_calls = []

    def project(value):
        projection_calls.append(1)
        return value[:, :2] * 2

    expected = (project(result) + tail) + prefix
    projection_calls.clear()
    final = backend.project_outputs(
        parallel,
        result,
        tail,
        SimpleNamespace(forward=project),
        output_dtype=torch.bfloat16,
        residual=prefix,
    )
    assert torch.equal(final, expected) and projection_calls == [1]
    assert len(normalizations) == 1 and prefix.eq(7).all() and tail.eq(6).all()


def test_fp32_fallback_never_launches_deferred_or_changes_cast_norm_order():
    backend = create_moe_backend("fused")
    partial = torch.tensor([[1.003, 0.995, -0.7, 0.2]], dtype=torch.float32)
    norm = TorchRMSNorm(4)
    norm.weight = torch.ones(4, dtype=torch.bfloat16)
    parallel = SimpleNamespace(all_reduce=lambda value: value * 3)
    result, tail = backend.forward_outputs(
        parallel,
        lambda out: partial,
        norm=norm,
        output_dtype=torch.bfloat16,
        deferred_producer=lambda out: pytest.fail("FP32 fallback launched deferred"),
        topk=2,
    )
    torch.testing.assert_close(result, norm.forward((partial * 3).bfloat16()), rtol=0, atol=0)
    assert tail is None


def native_inputs(host, rows=2):
    provider, events, _ = host
    provider.comm.config = SimpleNamespace(num_push_blocks=96)
    provider.__dict__["_push_max_bytes"] = 512 * 1024
    norm = TorchRMSNorm(3584)
    norm.weight = cuda_tensor(3584)
    epilogue = RMSNormEpilogue(norm, rows)
    return (
        provider,
        events,
        cuda_tensor(rows, 3584),
        cuda_tensor(rows * 16, 3584),
        cuda_tensor(rows * 16, dtype=torch.int32),
        cuda_tensor(rows, 16, dtype=torch.float32),
        epilogue,
    )


@pytest.mark.parametrize("change", ["width", "dtype", "topk", "zero", "large", "rows", "blocks"])
def test_native_cap_declines_group_metadata_before_any_producer(host, change):
    provider, _, out, _, _, _, norm = native_inputs(host)
    topk = 16
    if change == "width":
        out = cuda_tensor(2, 256)
    elif change == "dtype":
        out = cuda_tensor(2, 3584, dtype=torch.float32)
    elif change == "topk":
        topk = 8
    elif change == "zero":
        out = cuda_tensor(0, 3584)
        norm = RMSNormEpilogue(norm.normalizer, 0)
    elif change == "large":
        out = cuda_tensor(74, 3584)
        norm = RMSNormEpilogue(norm.normalizer, 74)
    elif change == "rows":
        norm = RMSNormEpilogue(norm.normalizer, 1)
    else:
        provider.comm.config.num_push_blocks = 2
    assert not provider.supports_weighted_all_reduce(out, norm=norm, topk=topk)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_native_same_instance_sdk_abi_and_facade_returns_output_only(host, monkeypatch, dtype):
    provider, _, out, values, indices, weights, norm = native_inputs(host)
    weights = weights.to(dtype)
    calls = []

    def finalize(comm, actual_out, actual_values, actual_indices, actual_weights, weight, eps):
        assert comm is provider.comm.obj and actual_out.data_ptr() == out.data_ptr()
        assert actual_values is values and actual_indices is indices and actual_weights is weights
        assert weight is norm.weight and eps == norm.eps
        calls.append(1)
        actual_out.fill_(7)

    provider.__dict__["_residual_kernel"] = SimpleNamespace(finalize_push_norm=finalize)
    fallback = DistributedImpl.__new__(
        type(
            "Unsupported",
            (DistributedImpl,),
            {
                "all_reduce": lambda self, x: pytest.fail(
                    "uninitialized output was ordinarily reduced"
                ),
                "all_gather": lambda self, x: x,
            },
        )
    )
    backend = FallbackDistributedImpl(MNNVLDistributedImpl(provider), fallback)
    monkeypatch.setattr(DistributedCommunicator, "plugins", [backend])
    parallel = TensorParallel()
    parallel.group = DistributedInfo(0, 8)
    assert parallel.supports_weighted_all_reduce(out, norm=norm, topk=16)
    assert parallel.weighted_all_reduce(values, indices, weights, out=out, norm=norm) is out
    assert calls == [1] and out.eq(7).all()


@pytest.mark.parametrize(
    "change",
    [
        "indices_dtype",
        "weights_dtype",
        "values_dtype",
        "alias",
        "stride",
        "weight_stride",
        "device",
    ],
)
def test_selected_native_bad_triple_or_local_storage_raises_without_retry(host, change):
    provider, _, out, values, indices, weights, norm = native_inputs(host)
    calls = []
    provider.__dict__["_residual_kernel"] = SimpleNamespace(
        finalize_push_norm=lambda *args: calls.append(args)
    )
    if change == "indices_dtype":
        indices = indices.to(torch.int64)
    elif change == "weights_dtype":
        weights = weights.to(torch.float64)
    elif change == "values_dtype":
        values = values.float()
    elif change == "alias":
        values = out
    elif change == "stride":
        values = cuda_tensor(32, 7168)[:, ::2]
    elif change == "device":

        class OtherDevice(type(out)):
            @property
            def device(self):
                return torch.device("cuda:1")

        out = out.as_subclass(OtherDevice)
    else:
        norm.normalizer.weight = cuda_tensor(7168)[::2]
    assert provider.supports_weighted_all_reduce(out, norm=norm, topk=16)
    with pytest.raises(ValueError):
        provider.weighted_all_reduce(values, indices, weights, out=out, norm=norm)
    assert calls == []


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_routing_weights_follow_native_vector_alignment_without_rank_local_fallback(host, dtype):
    provider, _, out, values, indices, _, norm = native_inputs(host)
    provider.input_alignment = 32
    offset = 16 // torch.empty((), dtype=dtype).element_size()
    storage = cuda_tensor(out.shape[0] * 16 + offset, dtype=dtype)
    weights = storage[offset:].view(out.shape[0], 16)
    assert weights.is_contiguous() and weights.data_ptr() % 32 == 16
    calls = []
    provider.__dict__["_residual_kernel"] = SimpleNamespace(
        finalize_push_norm=lambda *args: calls.append(args)
    )
    # Address validity is a selected local ABI check, never a collective route switch.
    assert provider.supports_weighted_all_reduce(out, norm=norm, topk=16)
    with pytest.raises(ValueError, match="aligned contiguous"):
        provider.weighted_all_reduce(values, indices, weights, out=out, norm=norm)
    assert calls == []

    # The same view meets the SDK's 16B requirement on the other provider ABI.
    provider.input_alignment = 16
    assert provider.supports_weighted_all_reduce(out, norm=norm, topk=16)
    assert provider.weighted_all_reduce(values, indices, weights, out=out, norm=norm) is out
    assert len(calls) == 1

    # Do not accidentally impose the routing-weight vector width on the output.
    provider.input_alignment = 32
    weights = cuda_tensor(out.shape[0], 16, dtype=dtype)
    shifted_out = cuda_tensor(out.numel() + 8)[8:].view_as(out)
    assert weights.data_ptr() % 32 == 0 and shifted_out.data_ptr() % 32 == 16
    calls.clear()
    assert provider.supports_weighted_all_reduce(shifted_out, norm=norm, topk=16)
    assert (
        provider.weighted_all_reduce(values, indices, weights, out=shifted_out, norm=norm)
        is shifted_out
    )
    assert len(calls) == 1


def test_cold_native_capture_fails_before_jit(host, monkeypatch):
    provider, _, out, values, indices, weights, norm = native_inputs(host)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before graph capture"):
        provider.weighted_all_reduce(values, indices, weights, out=out, norm=norm)
    assert "_residual_kernel" not in provider.__dict__


@pytest.mark.parametrize("tokens,enabled", [(8192, True), (8193, False)])
def test_official_moe_pdl_token_limit_reaches_fi(fi_executor, monkeypatch, tokens, enabled):
    obj, calls = fi_executor
    monkeypatch.delenv("SGLANG_TRTLLM_MOE_PDL_MAX_TOKENS", raising=False)
    x = torch.empty(tokens, 256, dtype=torch.bfloat16).as_subclass(DeviceInput)
    obj.forward(x, torch.ones(tokens, 2), torch.zeros(tokens, 2, dtype=torch.int32))
    assert len(calls) == 1 and calls[0]["enable_pdl"] is enabled


def test_selected_native_failure_is_not_retried(host):
    provider, _, out, values, indices, weights, norm = native_inputs(host)
    calls = []

    def fail(*args):
        calls.append(1)
        raise RuntimeError("native CUDA failure")

    provider.__dict__["_residual_kernel"] = SimpleNamespace(finalize_push_norm=fail)
    with pytest.raises(RuntimeError, match="native CUDA failure"):
        provider.weighted_all_reduce(values, indices, weights, out=out, norm=norm)
    assert calls == [1]
