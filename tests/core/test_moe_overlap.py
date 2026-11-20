"""CPU scheduling/hint contracts; no CUDA execution or performance evidence."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed.impl import (
    DistributedCommunicator,
    DistributedImpl,
    FallbackDistributedImpl,
)
from minisgl.distributed.mnnvl import MNNVLDistributedImpl
from minisgl.distributed.parallel import TensorParallel
from minisgl.layers.norm import TorchRMSNorm
from minisgl.moe import create_moe_backend
from test_collective_outputs import cuda_tensor
from test_collective_outputs import host as collective_host

host = collective_host


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("CPU overlap test initialized CUDA")

    monkeypatch.setattr(torch.cuda, "init", forbid)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)


@pytest.fixture
def streams(monkeypatch):
    events = []

    class Stream:
        def __init__(self, name, device):
            self.name, self.device = name, device

        def wait_stream(self, other):
            events.append("wait:" + self.name + "<-" + other.name)

    main = Stream("main", torch.device("cpu"))
    shared = Stream("shared", torch.device("cpu"))
    active = [main]

    @contextmanager
    def use(stream):
        before = active[0]
        active[0] = stream
        events.append("enter:" + stream.name)
        try:
            yield
        finally:
            events.append("exit:" + stream.name)
            active[0] = before

    def create(*, device):
        events.append("create")
        return Stream("shared", device)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: active[0])
    monkeypatch.setattr(torch.cuda, "stream", use)
    monkeypatch.setattr(torch.cuda, "Stream", create)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return events, main, shared, active


def test_backend_stream_prepares_once_and_declines_cpu_without_initialization(streams, monkeypatch):
    events, _, _, _ = streams
    backend = create_moe_backend("fused")
    backend.epilogue.prepare_output_stream(torch.empty(1))
    assert events == [] and not hasattr(backend, "_shared_stream")
    output = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    backend.epilogue.prepare_output_stream(output)
    backend.epilogue.prepare_output_stream(output)
    assert events == ["create"]
    with pytest.raises(ValueError, match="another device"):
        backend.epilogue.prepare_output_stream(
            SimpleNamespace(is_cuda=True, device=torch.device("cuda:1"))
        )
    other = create_moe_backend("fused")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before graph capture"):
        other.epilogue.prepare_output_stream(output)
    assert events == ["create"]


def test_front_fork_shared_pull_join_then_latent_norm_order(streams):
    events, _, side, active = streams
    backend = create_moe_backend("fused")
    backend.epilogue._shared_stream = side
    storage = torch.empty(12, dtype=torch.bfloat16)
    latent, shared = storage[:8].view(2, 4), storage[8:].view(2, 2)
    normalizer = TorchRMSNorm(4)
    normalizer.weight = torch.tensor([0.5, 1, 1.5, 2], dtype=torch.bfloat16)

    def experts(out):
        events.append("experts:" + active[0].name)
        return out.fill_(2)

    def shared_producer(out):
        events.append("shared:" + active[0].name)
        return out.fill_(3)

    class Input:
        is_cuda = True

        def record_stream(self, stream):
            events.append("record:" + stream.name)

    class ReturnedTensor(torch.Tensor):
        @property
        def is_cuda(self):
            return True

        def record_stream(self, stream):
            events.append("record-result:" + stream.name)

    def reduce(value, *, norm=None, output_key=None, overlap=False):
        assert output_key == "role"
        events.append(("shared_AR:" if overlap else "latent_AR:") + active[0].name)
        if overlap:
            assert norm is None
        result = value * 2
        if overlap:
            return result.as_subclass(ReturnedTensor)
        return norm(result) if norm is not None else result

    mixed, result = backend.epilogue.forward_outputs(
        SimpleNamespace(all_reduce_residual=reduce),
        experts,
        shared_producer,
        expert_out=latent,
        shared_out=shared,
        packed=storage,
        output_key="role",
        norm=normalizer,
        output_dtype=torch.bfloat16,
        shared_inputs=(Input(),),
    )
    assert events == [
        "wait:shared<-main",
        "record:shared",
        "experts:main",
        "enter:shared",
        "shared:shared",
        "shared_AR:shared",
        "exit:shared",
        "wait:main<-shared",
        "record-result:main",
        "latent_AR:main",
    ]
    expected = normalizer.forward(torch.full((2, 4), 4, dtype=torch.bfloat16))
    torch.testing.assert_close(mixed, expected, rtol=0, atol=0)
    assert result.eq(6).all()
    assert result.data_ptr() != shared.data_ptr()  # A generic AR result need not alias the MC tail.


def test_serial_fallback_preserves_fp32_partials_then_cast_norm_without_cuda():
    backend = create_moe_backend("fused")
    partial = torch.tensor([[1.003, 0.995, -0.7, 0.2]], dtype=torch.float32)
    normalizer = TorchRMSNorm(4)
    normalizer.weight = torch.ones(4, dtype=torch.bfloat16)
    events = []

    def experts(out):
        assert out is None
        events.append("expert")
        return partial.clone()

    def shared(out):
        assert out is None
        events.append("shared")
        return torch.tensor([[0.07, 0.01]], dtype=torch.bfloat16)

    def reduce(value):
        events.append("reduce:" + str(value.dtype))
        return value * 3

    actual, tail = backend.epilogue.forward_outputs(
        SimpleNamespace(all_reduce=reduce),
        experts,
        shared,
        norm=normalizer,
        output_dtype=torch.bfloat16,
    )
    assert events == ["expert", "shared", "reduce:torch.float32"]
    torch.testing.assert_close(actual, normalizer.forward((partial * 3).bfloat16()), rtol=0, atol=0)
    assert tail.dtype == torch.float32


@pytest.mark.parametrize("failed", ["expert", "shared"])
def test_producer_error_is_not_retried_or_routed_to_serial(streams, failed):
    events, _, side, _ = streams
    backend = create_moe_backend("fused")
    backend.epilogue._shared_stream = side
    storage = torch.empty(12, dtype=torch.bfloat16)
    counts = {"expert": 0, "shared": 0, "reduce": 0}

    def producer(name, value):
        counts[name] += 1
        if name == failed:
            raise RuntimeError("native CUDA failure")
        return value.fill_(1)

    def reduce(value, **kwargs):
        counts["reduce"] += 1
        return value

    with pytest.raises(RuntimeError, match="native CUDA failure"):
        backend.epilogue.forward_outputs(
            SimpleNamespace(all_reduce_residual=reduce),
            lambda out: producer("expert", out),
            lambda out: producer("shared", out),
            expert_out=storage[:8].view(2, 4),
            shared_out=storage[8:].view(2, 2),
            packed=storage,
            output_key="role",
            output_dtype=torch.bfloat16,
        )
    assert counts == {"expert": 1, "shared": int(failed == "shared"), "reduce": 0}


def test_generic_execution_hint_does_not_change_return_contract():
    value = torch.arange(4, dtype=torch.float32)
    transport = SimpleNamespace(all_reduce=lambda x: x * 2)
    result = DistributedImpl.all_reduce_residual(
        transport, value, torch.ones_like(value), norm=lambda x: x + 3, overlap=True
    )
    torch.testing.assert_close(result, value * 2 + 4, rtol=0, atol=0)


@pytest.fixture
def low_sm(host):
    provider, _, _ = host
    calls, tuning = [], []

    class Kernels:
        def push_res(self, *args):
            calls.append(("push", args))

        def pull_res(self, *args):
            calls.append(("pull", args))

    marker = object()

    def resolve(table, **kwargs):
        assert table is marker
        tuning.append(kwargs)
        return SimpleNamespace(num_blocks=4, unroll=8)

    provider.__dict__.update(
        _residual_kernel=Kernels(),
        _push_max_bytes=512 << 10,
        _residual_ops=SimpleNamespace(RES_TUNING=marker, _resolve_tuning=resolve),
    )
    output = provider.reserve_output("role", (64, 10752), torch.bfloat16)
    return provider, output.view(-1)[3584:10752], calls, tuning


def test_explicit_hint_forces_small_shared_slice_to_official_low_sm_pull(low_sm):
    provider, value, calls, tuning = low_sm
    assert value.nbytes < 512 << 10
    provider.all_reduce_residual(value, output_key="role")
    assert calls[0][0] == "push" and tuning == []
    provider.all_reduce_residual(value, output_key="role", overlap=True)
    assert calls[1][0] == "pull"
    assert calls[1][1][0] is provider.comm.obj
    assert calls[1][1][3:] == (provider._output_mc_ptr(value, key="role"), 4, 8)
    assert tuning == [{"nbytes": value.nbytes, "num_blocks": 4, "unroll": 8}]


def test_hint_passes_tp_communicator_composite_mnnvl_chain(low_sm, monkeypatch):
    provider, value, calls, _ = low_sm
    bridge = MNNVLDistributedImpl(provider)
    backend = FallbackDistributedImpl(bridge, bridge)
    monkeypatch.setattr(DistributedCommunicator, "_backend", lambda self: backend)
    facade = SimpleNamespace(size=2, _comm=DistributedCommunicator())
    TensorParallel.all_reduce_residual(facade, value, output_key="role", overlap=True)
    assert calls[0][0] == "pull" and calls[0][1][-2:] == (4, 8)


def test_overlap_requires_key_and_bad_local_mc_alias_raises_without_fallback(low_sm):
    provider, value, calls, _ = low_sm
    assert not provider.supports_all_reduce_residual(value, overlap=True)
    invalid = cuda_tensor(7168)
    assert provider.supports_all_reduce_residual(invalid, overlap=True, output_key="role")
    with pytest.raises(ValueError, match="reserved multicast"):
        provider.all_reduce_residual(invalid, overlap=True, output_key="role")
    assert calls == []


@pytest.mark.parametrize("invalid", [None, 0, 1, "", "yes"])
def test_invalid_hint_is_not_truthiness_coerced_by_any_entry(low_sm, monkeypatch, invalid):
    provider, value, calls, tuning = low_sm
    bridge = MNNVLDistributedImpl(provider)
    backend = FallbackDistributedImpl(bridge, bridge)
    monkeypatch.setattr(DistributedCommunicator, "_backend", lambda self: backend)
    facade = SimpleNamespace(size=2, _comm=DistributedCommunicator())
    with pytest.raises(ValueError, match="bool"):
        TensorParallel.all_reduce_residual(facade, value, output_key="role", overlap=invalid)
    with pytest.raises(ValueError, match="bool"):
        TensorParallel.all_reduce_residual(SimpleNamespace(size=1), value, overlap=invalid)
    with pytest.raises(ValueError, match="bool"):
        DistributedImpl.all_reduce_residual(
            SimpleNamespace(all_reduce=lambda x: pytest.fail("invalid hint launched reduction")),
            value,
            overlap=invalid,
        )
    assert calls == [] and tuning == []
