"""Host contracts for optional native AR+RMSNorm; no CUDA runtime or SDK kernels."""

import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.distributed.impl import (
    DistributedCommunicator,
    FallbackDistributedImpl,
    TorchDistributedImpl,
)
from minisgl.distributed.mnnvl import MNNVLDistributedImpl
from minisgl.distributed.parallel import TensorParallel
from minisgl.distributed.sgl import SGLangAllReduceProvider
from minisgl.moe import create_moe_backend


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("CPU contract tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)
    assert not torch.cuda.is_initialized()
    yield
    assert not torch.cuda.is_initialized()


class DeviceTensor:
    """CPU storage with CUDA metadata for checking the host adapter only."""

    device = torch.device("cuda:0")

    def __init__(self, value):
        self.value = value

    def __getattr__(self, name):
        return getattr(self.value, name)

    def clone(self, **kwargs):
        return type(self)(self.value.clone(**kwargs))

    def view(self, *shape):
        return type(self)(self.value.view(*shape))


@pytest.fixture
def native(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def push_norm(obj, flat, weight, eps, rows):
        assert obj is comm.obj and flat.ndim == 1 and rows * 3584 == flat.numel()
        assert flat.is_contiguous() and weight.is_contiguous()
        assert flat.data_ptr() % 32 == weight.data_ptr() % 32 == 0
        assert weight.dtype == flat.dtype == torch.bfloat16
        assert eps == 1e-5
        events.append(("norm", rows))
        flat.value.mul_(2)  # A spy result, not a simulation of the native arithmetic.

    def load(name):
        assert name == "sglang.kernels.ops.communication.all_reduce_residual"
        events.append("import")
        return SimpleNamespace(
            _jit_module=lambda world: (
                events.append(("compile", world)) or SimpleNamespace(push_norm=push_norm)
            )
        )

    @contextmanager
    def stream():
        events.append("stream-enter")
        yield
        events.append("stream-exit")

    comm = SimpleNamespace(
        obj=object(),
        disabled=False,
        has_multicast=True,
        world_size=8,
        max_push_size=768 * 1024,
        config=SimpleNamespace(num_push_blocks=148),
        close=lambda: events.append("close"),
    )
    module = load("sglang.kernels.ops.communication.all_reduce_residual")._jit_module(8)
    provider = SGLangAllReduceProvider(
        comm,
        device=torch.device("cuda:0"),
        norm_module=module,
        input_alignment=32,
        execution_context=stream,
    )
    assert events == ["import", ("compile", 8)]
    events.clear()
    return provider, comm, events


@pytest.mark.parametrize(
    "rows,accepted", [(1, True), (8, True), (48, True), (73, True), (74, False)]
)
def test_native_norm_uses_real_push_payload_limit(native, rows, accepted):
    provider, _, events = native
    x = DeviceTensor(torch.ones(rows, 3584, dtype=torch.bfloat16))
    weight = DeviceTensor(torch.ones(3584, dtype=torch.bfloat16))
    actual = provider.try_all_reduce_norm(x, weight, 1e-5)
    if accepted:
        assert actual is x and x.value.eq(2).all()
        assert events == ["stream-enter", ("norm", rows), "stream-exit"]
    else:
        assert actual is None and x.value.eq(1).all() and not events


@pytest.mark.parametrize("layout", ["aligned", "offset", "strided"])
def test_native_norm_layout_never_selects_a_different_collective(native, layout):
    provider, _, events = native
    values = torch.arange(2 * 3584 * 2, dtype=torch.bfloat16)
    if layout == "offset":
        x = DeviceTensor(values[1 : 1 + 2 * 3584].view(2, 3584))
    elif layout == "strided":
        x = DeviceTensor(values.view(2, 7168)[:, ::2])
    else:
        x = DeviceTensor(values[: 2 * 3584].view(2, 3584))
    weight = DeviceTensor(torch.ones(7168, dtype=torch.bfloat16)[::2])
    before = x.value.clone()
    actual = provider.try_all_reduce_norm(x, weight, 1e-5)
    assert actual is not None
    torch.testing.assert_close(actual.value, before * 2, rtol=0, atol=0)
    assert events == ["stream-enter", ("norm", 2), "stream-exit"]
    if layout != "aligned":
        assert actual.data_ptr() != x.data_ptr()
        torch.testing.assert_close(x.value, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "case", ["dtype", "rank", "width", "weight", "empty", "eps", "slots", "counters"]
)
def test_native_norm_declines_without_writes_or_collectives(native, case):
    provider, _, events = native
    value = torch.ones(48, 3584, dtype=torch.bfloat16)
    weight = torch.ones(3584, dtype=torch.bfloat16)
    eps = 1e-5
    if case == "dtype":
        value = value.float()
    elif case == "rank":
        value = value.flatten()
    elif case == "width":
        value = value[:, :128]
    elif case == "weight":
        weight = weight[:128]
    elif case == "empty":
        value = value[:0]
    elif case == "eps":
        eps = float("nan")
    elif case == "slots":
        provider._norm_push_bytes = value.numel() * value.element_size() - 1
    elif case == "counters":
        provider._norm_push_blocks = 48  # Kernel needs min(rows,96) < counters, not rows*7.
    before = value.clone()
    assert provider.try_all_reduce_norm(DeviceTensor(value), DeviceTensor(weight), eps) is None
    torch.testing.assert_close(value, before, rtol=0, atol=0)
    assert not events


def test_native_module_survives_communication_close(native):
    provider, comm, events = native
    module = provider._norm_module

    def close():
        assert provider._norm_module is module
        events.append("close")

    comm.close = close
    provider.close()
    assert events == ["close"] and provider._norm_module is None
    assert (
        provider.try_all_reduce_norm(
            DeviceTensor(torch.ones(1, 3584, dtype=torch.bfloat16)),
            DeviceTensor(torch.ones(3584, dtype=torch.bfloat16)),
            1e-5,
        )
        is None
    )
    assert events == ["close"]


@pytest.mark.parametrize("failure", ["compile", "capture"])
def test_factory_preparation_failure_creates_no_communicator(monkeypatch, failure):
    import minisgl.distributed.sgl as sgl

    events = []
    cpu, gpu = object(), object()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: failure == "capture")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 8)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: list(range(8)))

    def compile_module(world):
        events.append("compile")
        raise RuntimeError("JIT failed")

    def load(name):
        if name.endswith("all_reduce_residual"):
            return SimpleNamespace(_jit_module=compile_module)
        if name.endswith("custom_all_reduce_v2"):

            def unexpected(*args, **kwargs):
                raise AssertionError("Must not create a communicator on preparation failure")

            return SimpleNamespace(
                CustomAllReduceV2=unexpected, get_supported_world_sizes=lambda: (8,)
            )
        raise AssertionError(name)

    monkeypatch.setattr(sgl.importlib, "import_module", load)
    monkeypatch.setattr(sgl.TorchDistributedImpl, "warmup", lambda *args: events.append("warmup"))
    with pytest.raises(
        RuntimeError, match="JIT failed" if failure == "compile" else "before capture"
    ):
        sgl.create_provider(
            tp_info=DistributedInfo(0, 8),
            cpu_group=cpu,
            device_group=gpu,
            device=torch.device("cuda:0"),
            max_bytes=1024,
        )
    assert events == (["compile"] if failure == "compile" else [])


def test_optional_provider_and_fallback_decline_before_trying_next_backend():
    events = []
    x, weight = torch.ones(2, 3), torch.ones(3)
    first = SimpleNamespace(try_all_reduce_norm=lambda *args: events.append("decline"))
    second = MNNVLDistributedImpl(
        SimpleNamespace(try_all_reduce_norm=lambda *args: events.append("accept") or x + 4)
    )
    actual = FallbackDistributedImpl(first, second).try_all_reduce_norm(x, weight, 1e-5)
    assert torch.equal(actual, x + 4) and events == ["decline", "accept"]
    events.clear()
    assert FallbackDistributedImpl(first, first).try_all_reduce_norm(x, weight, 1e-5) is None
    assert events == ["decline"]
    assert MNNVLDistributedImpl(SimpleNamespace()).try_all_reduce_norm(x, weight, 1e-5) is None
    assert TorchDistributedImpl().try_all_reduce_norm(x, weight, 1e-5) is None


def test_tensor_parallel_uses_selected_group_and_tp1_does_nothing(monkeypatch):
    events = []
    group = SimpleNamespace(try_all_reduce_norm=lambda *args: events.append("selected") or args[0])
    other = SimpleNamespace(try_all_reduce_norm=lambda *args: events.append("other") or args[0])
    monkeypatch.setattr(DistributedCommunicator, "groups", {"test": group})
    monkeypatch.setattr(DistributedCommunicator, "plugins", [other])
    parallel = TensorParallel()
    parallel._comm = DistributedCommunicator("test")
    parallel.group = DistributedInfo(0, 1)
    x, weight = torch.ones(2, 3), torch.ones(3)
    assert parallel.try_all_reduce_norm(x, weight, 1e-5) is None and not events
    parallel.group = DistributedInfo(0, 8)
    assert parallel.try_all_reduce_norm(x, weight, 1e-5) is x
    assert events == ["selected"]


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("has_shared", [False, True])
def test_moe_completion_declines_before_shared_or_casts_then_reduces_once(accepted, has_shared):
    backend, events = create_moe_backend("fused"), []
    mixed = torch.tensor([[0.5, 1]], dtype=torch.bfloat16)
    shared = torch.tensor([[1.003, 2.007]], dtype=torch.float32) if has_shared else None
    before = mixed.clone()

    def norm(x, weight, eps):
        events.append("norm-try")
        assert x is mixed and weight.dtype == torch.bfloat16 and eps == 1e-5
        return x * 2 if accepted else None

    def reduce(x):
        events.append(("reduce", x.dtype, x.shape))
        return x * 3

    result = backend.try_reduce_norm_outputs(
        SimpleNamespace(try_all_reduce_norm=norm, all_reduce=reduce),
        mixed,
        shared,
        weight=torch.ones(2, dtype=torch.bfloat16),
        eps=1e-5,
    )
    assert torch.equal(mixed, before)
    if not accepted:
        assert result is None and events == ["norm-try"]
    else:
        assert torch.equal(result[0], mixed * 2)
        if has_shared:
            assert torch.equal(result[1], shared.to(mixed.dtype) * 3)
            assert events == ["norm-try", ("reduce", torch.bfloat16, shared.shape)]
        else:
            assert result[1] is None and events == ["norm-try"]


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize("mixture_dtype", [torch.bfloat16, torch.float32])
def test_latent_moe_applies_experts_reduction_and_norm_exactly_once(
    monkeypatch, accepted, mixture_dtype
):
    import minisgl.layers.latent_moe as module

    events = []
    backend = create_moe_backend("fused")
    mixed = torch.tensor([[0.2542, 0.5031], [0.7582, 1.011]], dtype=mixture_dtype)
    shared = torch.tensor([[1, 2], [3, 4]], dtype=torch.bfloat16)

    def experts(*args):
        events.append("experts")
        return mixed.clone()

    def norm_try(value, weight, eps):
        events.append("norm-try")
        return value * 7 if accepted else None

    def reduce(value):
        events.append(("reduce", value.numel()))
        return value * 2

    def normalize(value):
        events.append("ordinary-norm")
        assert value.dtype == torch.bfloat16
        return value * 3

    def up(value):
        events.append("up")
        return value

    model = module.LatentMoE.__new__(module.LatentMoE)
    model._front = None
    model._backend = backend
    model._expert_executor = SimpleNamespace(input_dtype=None)
    model._tp = SimpleNamespace(try_all_reduce_norm=norm_try, all_reduce=reduce)
    model.gate = SimpleNamespace(weight=None, routing=lambda logits: object())
    model.routed_expert_down_proj = None
    model.routed_expert_norm = SimpleNamespace(
        weight=torch.ones(2, dtype=torch.bfloat16), eps=1e-5, forward=normalize
    )
    model.routed_expert_up_proj = SimpleNamespace(forward=up)
    dense = SimpleNamespace(weight_format="dense")
    model.shared_experts = SimpleNamespace(
        gate_proj=dense, up_proj=dense, forward_local=lambda *args, **kwargs: shared.clone()
    )
    monkeypatch.setattr(module, "linear_fp32", lambda *args: torch.zeros(2, 1))
    monkeypatch.setattr(backend, "forward_experts", experts)
    monkeypatch.setattr(module, "use_strict_bf16", lambda: False)
    actual = model.forward(torch.ones_like(mixed, dtype=torch.bfloat16))
    fused = accepted and mixture_dtype == torch.bfloat16
    expected = (mixed * 7 if fused else (mixed * 2).to(torch.bfloat16) * 3) + shared * 2
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected_events = ["experts"]
    if mixture_dtype == torch.bfloat16:
        expected_events.append("norm-try")
    expected_events += [("reduce", 4), "up"] if fused else [("reduce", 8), "ordinary-norm", "up"]
    assert events == expected_events


def test_native_module_is_retained_if_communication_close_raises(native):
    provider, comm, _ = native
    module = provider._norm_module

    def fail():
        raise RuntimeError("communication still alive")

    comm.close = fail
    with pytest.raises(RuntimeError, match="still alive"):
        provider.close()
    assert provider._norm_module is module


def test_disabled_native_communicator_declines_without_collectives(native):
    provider, comm, events = native
    comm.disabled = True
    x = DeviceTensor(torch.ones(1, 3584, dtype=torch.bfloat16))
    weight = DeviceTensor(torch.ones(3584, dtype=torch.bfloat16))
    assert provider.try_all_reduce_norm(x, weight, 1e-5) is None
    assert x.value.eq(1).all() and not events


@pytest.mark.parametrize("world", [1, 9, 12])
def test_factory_unsupported_world_keeps_native_disabled_fallback(monkeypatch, world):
    import minisgl.distributed.sgl as sgl

    events = []
    cpu, gpu = object(), object()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: world)
    monkeypatch.setattr(
        torch.distributed, "get_process_group_ranks", lambda group: list(range(world))
    )
    monkeypatch.setattr(sgl.TorchDistributedImpl, "warmup", lambda *args: events.append("warmup"))

    class Disabled:
        disabled = True

        def __init__(self, group, device, *, max_size):
            assert group is cpu
            events.append("native-disabled")

    def supported():
        events.append("supported-sizes")
        return (2, 3, 4, 5, 6, 7, 8, 16)

    def load(name):
        assert name.endswith("custom_all_reduce_v2"), (
            "Must not import/JIT norm for unsupported world"
        )
        return SimpleNamespace(CustomAllReduceV2=Disabled, get_supported_world_sizes=supported)

    @contextmanager
    def stream():
        yield

    monkeypatch.setattr(sgl.importlib, "import_module", load)
    monkeypatch.setitem(
        sys.modules, "tvm_ffi", SimpleNamespace(use_torch_stream=stream)
    )

    def reduce(tensor, *, group):
        assert group is gpu
        events.append("fallback")
        tensor.mul_(2)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    provider = sgl.create_provider(
        tp_info=DistributedInfo(0, world),
        cpu_group=cpu,
        device_group=gpu,
        device=torch.device("cuda:0"),
        max_bytes=1024,
    )
    assert provider._norm_module is None
    value = torch.ones(2, 3)
    assert provider.try_all_reduce_norm(value, torch.ones(3), 1e-5) is None
    assert value.eq(1).all()
    assert provider.all_reduce(value).eq(2).all()
    assert events == (
        (["supported-sizes"] if world > 1 else []) + ["warmup", "native-disabled", "fallback"]
    )
