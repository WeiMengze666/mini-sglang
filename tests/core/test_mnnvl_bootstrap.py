"""Two-phase provider lifecycle and native SDK host adapter; no CUDA arithmetic."""

import importlib
import os
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch
from minisgl.distributed import DistributedInfo, sgl
from minisgl.distributed.mnnvl import prepare_mnnvl_provider


@pytest.fixture
def native_host(monkeypatch, tmp_path):
    calls = []

    def no_cuda(*a, **kw):
        raise AssertionError("Host fixtures must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 3))
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda i: SimpleNamespace(uuid=f"gpu-{i}")
    )
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(sgl.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "2")
    monkeypatch.delenv("SGLANG_SKIP_P2P_CHECK", raising=False)
    sdk = SimpleNamespace()

    def probe(src, dst):
        calls.append((src, dst, os.environ["SGLANG_CACHE_DIR"]))

    sdk.gpu_p2p_access_check = probe
    original = importlib.import_module

    def vendor(name, *a, **kw):
        if name.endswith("custom_all_reduce_utils"):
            return sdk
        return original(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", vendor)
    return sdk, calls


@pytest.mark.parametrize("preset", [False, True])
@pytest.mark.parametrize("fail", [False, True])
def test_native_prepare_scopes_cache_and_restores_environment(
    native_host, monkeypatch, preset, fail
):
    sdk, calls = native_host
    if preset:
        monkeypatch.setenv("SGLANG_CACHE_DIR", "caller-cache")
    else:
        monkeypatch.delenv("SGLANG_CACHE_DIR", raising=False)
    if fail:

        def probe(*a):
            raise RuntimeError("peer probe failed")

        sdk.gpu_p2p_access_check = probe
        with pytest.raises(RuntimeError, match="peer probe failed"):
            sgl.prepare(tp_info=DistributedInfo(0, 2), device=torch.device("cuda:0"))
    else:
        sgl.prepare(tp_info=DistributedInfo(0, 2), device=torch.device("cuda:0"))
        sgl.prepare(tp_info=DistributedInfo(1, 2), device=torch.device("cuda:1"))
        assert len(calls) == 2 and calls[0] == calls[1]
        assert calls[0][:2] == (0, 1) and calls[0][2] != "caller-cache"
    assert os.getenv("SGLANG_CACHE_DIR") == ("caller-cache" if preset else None)


@pytest.mark.parametrize("condition", ["initialized", "hidden_peers", "skip_probe", "no_cuda"])
def test_native_prepare_rejects_invalid_lifecycle_before_sdk(native_host, monkeypatch, condition):
    _, calls = native_host
    if condition == "initialized":
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    elif condition == "hidden_peers":
        monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    elif condition == "skip_probe":
        monkeypatch.setenv("SGLANG_SKIP_P2P_CHECK", "1")
    else:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises((ValueError, RuntimeError)):
        sgl.prepare(tp_info=DistributedInfo(0, 2), device=torch.device("cuda:0"))
    assert calls == []


@pytest.mark.parametrize("has_prepare", [False, True])
def test_engine_prepares_before_groups_and_creates_after_groups(monkeypatch, has_prepare):
    import minisgl.engine.engine as engine_module

    events = []
    cpu, gpu = object(), object()
    module = ModuleType("test_two_phase_mnnvl")
    if has_prepare:
        module.prepare = lambda **kw: events.append(("prepare", kw))
    provider = SimpleNamespace(all_reduce=lambda x: x, all_gather=lambda x: x)

    def create(**kw):
        assert kw["cpu_group"] is cpu and kw["device_group"] is gpu
        assert kw["max_bytes"] == 8 * 16 * 2
        events.append(("create", kw))
        return provider

    module.create_provider = create
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.distributed, "group", SimpleNamespace(WORLD=cpu))
    monkeypatch.setattr(
        torch.distributed, "init_process_group", lambda **kw: events.append(("gloo", kw))
    )
    monkeypatch.setattr(
        torch.distributed, "new_group", lambda **kw: events.append(("nccl", kw)) or gpu
    )
    monkeypatch.setattr(
        engine_module, "enable_mnnvl_distributed", lambda p: events.append(("enable", p))
    )
    cfg = SimpleNamespace(
        distributed_backend="mnnvl",
        distributed_provider=module.__name__,
        tp_info=DistributedInfo(0, 2),
        distributed_timeout=5,
        distributed_addr="unused",
        max_forward_len=8,
        model_config=SimpleNamespace(hidden_size=16),
    )
    engine = SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.bfloat16)
    assert engine_module.Engine._init_communication(engine, cfg) is cpu
    assert [e[0] for e in events] == (["prepare"] if has_prepare else []) + [
        "gloo",
        "nccl",
        "create",
        "enable",
    ]
    if has_prepare:
        assert events[0][1] == {"tp_info": cfg.tp_info, "device": engine.device}


def test_invalid_provider_factory_fails_before_prepare(monkeypatch):
    module = ModuleType("test_invalid_mnnvl_factory")
    module.prepare = lambda **kw: pytest.fail("invalid factory must not prepare")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(AttributeError, match="lacks create_provider"):
        prepare_mnnvl_provider(module.__name__)


@pytest.mark.parametrize("capability,alignment", [((9, 0), 16), ((10, 3), 32)])
def test_native_factory_delegates_collectives_capture_and_close(
    native_host, monkeypatch, capability, alignment
):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: capability)
    _, _ = native_host
    events = []
    cpu, gpu = object(), object()

    class Native:
        disabled = False

        def __init__(self, group, device, *, max_size):
            assert group is cpu and device == torch.device("cuda:0") and max_size == 4096

        def should_custom_ar(self, tensor):
            return tensor.numel() == 2

        def custom_all_reduce(self, tensor):
            events.append("native")
            return tensor + 2

        @contextmanager
        def capture(self):
            events.append("capture-enter")
            yield
            events.append("capture-exit")

        def close(self):
            events.append("close")

    monkeypatch.setattr(
        sgl.TorchDistributedImpl,
        "warmup",
        lambda self, device: events.append("fallback-warmup"),
    )
    original = importlib.import_module

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def vendor(name, *a, **kw):
        if name.endswith("custom_all_reduce_v2"):
            return SimpleNamespace(CustomAllReduceV2=Native, get_supported_world_sizes=lambda: (2,))
        if name.endswith("all_reduce_residual"):
            return SimpleNamespace(_jit_module=lambda world: object())
        return original(name, *a, **kw)

    monkeypatch.setattr(importlib, "import_module", vendor)

    def fallback(tensor, *, group):
        assert group is gpu
        events.append("fallback")
        tensor.add_(10)

    monkeypatch.setattr(torch.distributed, "all_reduce", fallback)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda group: [0, 1])
    ffi = ModuleType("tvm_ffi")

    @contextmanager
    def stream():
        events.append("stream-enter")
        yield
        events.append("stream-exit")

    ffi.use_torch_stream = stream
    monkeypatch.setitem(sys.modules, "tvm_ffi", ffi)

    def gather(output, tensor, *, group):
        assert group is gpu
        output.copy_(torch.cat([tensor, tensor]))

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", gather)
    provider = sgl.create_provider(
        tp_info=DistributedInfo(0, 2),
        cpu_group=cpu,
        device_group=gpu,
        device=torch.device("cuda:0"),
        max_bytes=4096,
    )
    assert provider.input_alignment == alignment
    assert provider.all_reduce(torch.zeros(2)).eq(2).all()
    assert provider.all_reduce(torch.zeros(1)).eq(10).all()
    assert provider.all_gather(torch.ones(2)).shape == (4,)
    with provider.capture():
        pass
    provider.close()
    assert events == [
        "fallback-warmup",
        "stream-enter",
        "stream-exit",
        "stream-enter",
        "native",
        "stream-exit",
        "fallback",
        "stream-enter",
        "capture-enter",
        "capture-exit",
        "stream-exit",
        "close",
    ]


@pytest.mark.parametrize("mismatch", ["rank", "size", "order"])
def test_native_factory_rejects_different_group_membership(native_host, monkeypatch, mismatch):
    cpu, gpu = object(), object()
    monkeypatch.setattr(
        torch.distributed, "get_rank", lambda group: int(mismatch == "rank" and group is gpu)
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_world_size",
        lambda group: 3 if mismatch == "size" and group is gpu else 2,
    )
    monkeypatch.setattr(
        torch.distributed,
        "get_process_group_ranks",
        lambda group: [1, 0] if mismatch == "order" and group is gpu else [0, 1],
    )
    with pytest.raises(ValueError, match="coordinates|same ordered ranks"):
        sgl.create_provider(
            tp_info=DistributedInfo(0, 2),
            cpu_group=cpu,
            device_group=gpu,
            device=torch.device("cuda:0"),
            max_bytes=4096,
        )


@pytest.mark.parametrize("layout", ["aligned", "offset", "transpose", "strided"])
@pytest.mark.parametrize("disabled", [False, True])
def test_native_payload_normalization_preserves_values_and_collective_choice(
    monkeypatch, layout, disabled
):
    from minisgl.distributed.mnnvl import MNNVLAllReduceProvider

    base = torch.arange(64, dtype=torch.bfloat16)
    if layout == "aligned":
        tensor = base[:32]
    elif layout == "offset":
        tensor = base[8:40]  # 16-byte view offset meets SDK entry but not its PDL copy.
    elif layout == "transpose":
        tensor = base.view(8, 8).T
    else:
        tensor = base[::2]
    expected = tensor.clone() * 2
    entered = []

    def reduce(payload):
        assert payload.is_contiguous() and payload.data_ptr() % 32 == 0
        payload.mul_(2)
        return payload

    def native(payload):
        entered.append("native")
        return reduce(payload)

    def fallback(payload, *, group):
        assert group == "same-ranks"
        entered.append("fallback")
        reduce(payload)

    monkeypatch.setattr(torch.distributed, "all_reduce", fallback)
    comm = SimpleNamespace(
        disabled=disabled, should_custom_ar=lambda x: True, custom_all_reduce=native
    )
    provider = MNNVLAllReduceProvider(comm, group="same-ranks", input_alignment=32)
    torch.testing.assert_close(provider.all_reduce(tensor), expected, rtol=0, atol=0)
    assert entered == ["fallback" if disabled else "native"]
