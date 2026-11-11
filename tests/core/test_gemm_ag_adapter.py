"""CPU contracts for an optional mature GEMM/all-gather epilogue; no native arithmetic claim."""

import gc
import importlib
import sys
import weakref
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedInfo
from minisgl.distributed.impl import (
    DistributedCommunicator,
    FallbackDistributedImpl,
    TorchDistributedImpl,
)
from minisgl.distributed.mnnvl import MNNVLDistributedImpl
from minisgl.distributed.parallel import TensorParallel
from minisgl.distributed.sgl import SGLangAllReduceProvider
from minisgl.layers.latent_moe import LatentMoE
from minisgl.layers.linear import Linear


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU contract tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)


def endpoint(result, events, label):
    class Endpoint(TorchDistributedImpl):
        def try_linear_all_gather_add(self, x, weight, addend):
            events.append(label)
            return result

    return Endpoint()


@pytest.mark.parametrize("success", [False, True])
def test_optional_fallback_exactly_once_and_group_selection(monkeypatch, success):
    x = torch.ones(1, 2)
    events = []
    primary = endpoint(x if success else None, events, "primary")
    fallback = endpoint(x, events, "fallback")
    mixed = FallbackDistributedImpl(primary, fallback)
    monkeypatch.setattr(DistributedCommunicator, "plugins", [mixed])
    monkeypatch.setattr(DistributedCommunicator, "groups", {"other": fallback})
    assert DistributedCommunicator().try_linear_all_gather_add(x, x, x) is x
    assert events == (["primary"] if success else ["primary", "fallback"])
    events.clear()
    assert DistributedCommunicator("other").try_linear_all_gather_add(x, x, x) is x
    assert events == ["fallback"]
    events.clear()
    same = endpoint(None, events, "same")
    assert FallbackDistributedImpl(same, same).try_linear_all_gather_add(x, x, x) is None
    assert events == ["same"]


def test_optional_default_provider_and_tp1_never_collect(monkeypatch):
    monkeypatch.setattr(
        torch.distributed, "all_reduce", lambda *a, **k: pytest.fail("unexpected collective")
    )
    x = torch.ones(1, 2)
    assert TorchDistributedImpl().try_linear_all_gather_add(x, x, x) is None
    assert MNNVLDistributedImpl(SimpleNamespace()).try_linear_all_gather_add(x, x, x) is None
    parallel = TensorParallel()
    parallel.group = DistributedInfo(0, 1)
    parallel._comm = SimpleNamespace(
        try_linear_all_gather_add=lambda *a: pytest.fail("TP1 delegated")
    )
    assert parallel.try_linear_all_gather_add(x, x, x) is None
    parallel.group = DistributedInfo(0, 8)
    calls = []
    parallel._comm.try_linear_all_gather_add = lambda *a: calls.append(a) or x
    assert parallel.try_linear_all_gather_add(x, x, x) is x and len(calls) == 1


class Kernel:
    def __init__(self):
        self.calls = []

    def run(self, comm, x, weight, addend, c, output):
        self.calls.append((comm, x, weight, addend, c, output))
        output.fill_(3)


@pytest.fixture(scope="module")
def weight():
    # Real canonical metadata/storage, without evaluating a substitute numeric kernel.
    return torch.empty(7168, 3584, dtype=torch.bfloat16)


def provider(kernel, *, blocks=128, slot=65536, disabled=False, multicast=True):
    comm = SimpleNamespace(
        disabled=disabled,
        has_multicast=multicast,
        world_size=8,
        max_push_size=slot,
        config=SimpleNamespace(num_push_blocks=blocks),
        obj=object(),
    )
    return SGLangAllReduceProvider(
        comm,
        device=torch.device("cpu"),
        gemm_module=kernel,
        gemm_shape=(3584, 7168, 12),
        input_alignment=32,
    )


@pytest.mark.parametrize("rows", range(1, 13))
def test_native_abi_uses_same_instance_and_current_full_weight(weight, rows):
    kernel = Kernel()
    backend = provider(kernel)
    x = torch.zeros(rows, 3584, dtype=torch.bfloat16)
    addend = torch.ones(rows, 7168, dtype=torch.bfloat16)
    result = backend.try_linear_all_gather_add(x, weight, addend)
    assert result.eq(3).all() and result.dtype == x.dtype and result.shape == addend.shape
    comm, actual_x, actual_w, actual_b, c, out = kernel.calls[0]
    assert comm is backend.comm.obj and actual_x is x and actual_w is weight and actual_b is addend
    assert c is None and out is result and result.data_ptr() != addend.data_ptr()
    assert x.count_nonzero() == 0 and addend.eq(1).all()


@pytest.mark.parametrize(
    "unsupported",
    [
        "empty",
        "rows13",
        "rows48",
        "dtype",
        "width",
        "addend",
        "weight",
        "grad",
        "device",
        "disabled",
        "multicast",
        "slot",
        "counter",
    ],
)
def test_unsupported_is_none_before_allocation_mutation_or_native_call(
    weight, unsupported, monkeypatch
):
    kernel = Kernel()
    options = {"disabled": unsupported == "disabled", "multicast": unsupported != "multicast"}
    if unsupported == "slot":
        options["slot"] = 1791
    if unsupported == "counter":
        options["blocks"] = 7
    backend = provider(kernel, **options)
    x = torch.zeros(1, 3584, dtype=torch.bfloat16)
    addend = torch.ones(1, 7168, dtype=torch.bfloat16)
    current = weight
    if unsupported in ("empty", "rows13", "rows48"):
        rows = {"empty": 0, "rows13": 13, "rows48": 48}[unsupported]
        x, addend = x.expand(rows, -1), addend.expand(rows, -1)
    elif unsupported == "dtype":
        x = x.float()
    elif unsupported == "width":
        x = x[:, :-1]
    elif unsupported == "addend":
        addend = addend[:, :-1]
    elif unsupported == "weight":
        current = current[:-1]
    elif unsupported == "grad":
        x.requires_grad_()
    elif unsupported == "device":
        current = torch.empty(weight.shape, device="meta", dtype=weight.dtype)
    original_x, original_b = x.detach().clone(), addend.clone()
    monkeypatch.setattr(
        torch.Tensor, "new_empty", lambda *a, **k: pytest.fail("unsupported allocated output")
    )
    assert backend.try_linear_all_gather_add(x, current, addend) is None and not kernel.calls
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(addend, original_b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "blocks,slot,success", [(85, 21504, True), (84, 21504, False), (85, 21503, False)]
)
def test_m12_actual_consumer_and_slot_boundary_precede_producer(weight, blocks, slot, success):
    kernel = Kernel()
    backend = provider(kernel, blocks=blocks, slot=slot)
    result = backend.try_linear_all_gather_add(
        torch.zeros(12, 3584).bfloat16(), weight, torch.zeros(12, 7168).bfloat16()
    )
    assert (result is not None) == success and len(kernel.calls) == int(success)


@pytest.mark.parametrize("layout", ["misaligned", "strided"])
def test_small_views_normalize_without_weight_copy_or_rank_local_fallback(weight, layout):
    if layout == "misaligned":
        x = torch.zeros(2 * 3584 + 1, dtype=torch.bfloat16)[1:].view(2, 3584)
        addend = torch.ones(2 * 7168 + 1, dtype=torch.bfloat16)[1:].view(2, 7168)
    else:
        x = torch.zeros(2, 7168, dtype=torch.bfloat16)[:, ::2]
        addend = torch.ones(2, 14336, dtype=torch.bfloat16)[:, ::2]
    kernel = Kernel()
    result = provider(kernel).try_linear_all_gather_add(x, weight, addend)
    _, actual_x, actual_w, actual_b, _, _ = kernel.calls[0]
    assert result is not None and actual_w is weight
    assert actual_x.is_contiguous() and actual_x.data_ptr() % 32 == 0
    assert actual_b.is_contiguous() and actual_b.data_ptr() % 16 == 0
    assert x.count_nonzero() == 0 and addend.eq(1).all()


def test_unprepared_weight_layout_is_explicit_error_never_fallback_or_native():
    kernel = Kernel()
    backend = provider(kernel)
    weight = torch.empty(7168 * 3584 + 1, dtype=torch.bfloat16)[1:].view(7168, 3584)
    with pytest.raises(ValueError, match="Prepare replicated dense weight"):
        backend.try_linear_all_gather_add(
            torch.zeros(1, 3584).bfloat16(), weight, torch.zeros(1, 7168).bfloat16()
        )
    assert not kernel.calls


def test_close_keeps_module_alive_through_native_cleanup():
    kernel = Kernel()
    backend = provider(kernel)

    def close():
        assert backend._gemm_module is kernel

    backend.comm.close = close
    backend.close()
    assert backend._gemm_module is None
    backend = provider(kernel)

    def fail():
        raise RuntimeError("close failed")

    backend.comm.close = fail
    with pytest.raises(RuntimeError, match="close failed"):
        backend.close()
    assert backend._gemm_module is kernel


@pytest.mark.parametrize("bias", [False, True])
def test_linear_optional_and_fallback_keep_bf16_projection_then_add(bias):
    layer = Linear(2, 1, has_bias=bias, dtype=torch.bfloat16)
    layer.weight.copy_(torch.tensor([[1.0, 1 / 256]]))
    if bias:
        layer.bias.zero_()
    x, addend = torch.ones(1, 2).bfloat16(), -torch.ones(1, 1).bfloat16()
    events = []
    parallel = SimpleNamespace(try_linear_all_gather_add=lambda *a: events.append("try") or None)
    result = layer.forward_replicated_add(x, addend, parallel=parallel)
    assert result.item() == 0 and events == ([] if bias else ["try"])
    assert ((x.float() @ layer.weight.float().T) + addend.float()).item() != 0
    if not bias:
        parallel.try_linear_all_gather_add = lambda *a: (
            events.append("native") or torch.full_like(addend, 7)
        )
        layer.forward = lambda x: pytest.fail("successful optional path repeated projection")
        assert layer.forward_replicated_add(x, addend, parallel=parallel).item() == 7


def test_prepared_weight_has_no_private_mirror_and_reload_uses_current_storage():
    layer = Linear(2, 3, dtype=torch.bfloat16)
    external = torch.arange(7).bfloat16()[1:].view(3, 2)
    old = weakref.ref(external)
    layer.weight = external
    layer.prepare_replicated_add()
    assert layer.weight.is_contiguous() and layer.weight.data_ptr() % 32 == 0
    assert layer.weight is not external and set(layer.state_dict()) == {"weight"}
    del external
    gc.collect()
    assert old() is None
    first = layer.weight
    layer.prepare_replicated_add()
    assert layer.weight is first
    layer.load_state_dict({"weight": torch.full((3, 2), 2, dtype=torch.bfloat16)})
    layer.prepare_replicated_add()
    seen = []
    parallel = SimpleNamespace(try_linear_all_gather_add=lambda x, w, b: seen.append(w) or None)
    result = layer.forward_replicated_add(
        torch.ones(1, 2).bfloat16(), torch.zeros(1, 3).bfloat16(), parallel=parallel
    )
    assert seen == [layer.weight] and layer.weight is not first and result.eq(4).all()


def model_config():
    return SimpleNamespace(
        hidden_size=4,
        intermediate_size=7,
        moe_intermediate_size=3,
        num_experts=4,
        num_experts_per_token=2,
        num_expert_group=2,
        topk_group=1,
        topk_method="noaux_tc",
        moe_router_activation_func="sigmoid",
        moe_renormalize=True,
        routed_scaling_factor=1.3,
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
        routed_expert_hidden_size=2,
        latent_moe_use_norm=True,
        rms_norm_eps=1e-5,
        num_shared_experts=1,
    )


@pytest.mark.parametrize("loader", ["state_dict", "stream"])
def test_moe_loading_prepares_up_once_after_children_without_canonical_extra_storage(
    monkeypatch, loader
):
    import minisgl.core as core

    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1, bf16_precision="default"))
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = LatentMoE(model_config())
    finally:
        torch.set_default_dtype(old)
    model._tp.group = DistributedInfo(0, 8)
    calls = []
    original = Linear.prepare_replicated_add

    def prepare(op):
        assert op is model.routed_expert_up_proj
        assert all(not t.is_meta for t in model.state_dict().values())
        before = op.weight
        original(op)
        calls.append((before, op.weight))

    monkeypatch.setattr(Linear, "prepare_replicated_add", prepare)
    for value in (1.0, 2.0):
        state = {k: torch.full_like(v, value) for k, v in model.state_dict().items()}
        if loader == "stream":
            model.load_weights(iter(state.items()), device=torch.device("cpu"))
        else:
            model.load_state_dict(state)
        assert calls[-1][0] is calls[-1][1]
        assert model.routed_expert_up_proj.weight.eq(value).all()
    assert len(calls) == 2 and calls[0][1] is not calls[1][1]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_actual_moe_completion_is_after_reduce_and_norm_and_adds_shared_once(monkeypatch, dtype):
    import minisgl.core as core

    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1, bf16_precision="default"))
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        model = LatentMoE(model_config())
    finally:
        torch.set_default_dtype(previous)
    generator = torch.Generator().manual_seed(920)
    for weight in model.state_dict().values():
        weight.copy_(torch.randn(weight.shape, generator=generator) * 0.2)
    x = torch.randn(2, 3, 4, generator=generator).to(dtype)
    expected = model.forward(x)
    events = []
    original_reduce = model._backend.reduce_outputs

    def reduce(*args):
        result = original_reduce(*args)
        events.append("reduced")
        return result

    monkeypatch.setattr(model._backend, "reduce_outputs", reduce)
    original_norm = model.routed_expert_norm.forward

    def norm(value):
        assert events == ["reduced"]
        events.append("normed")
        return original_norm(value)

    model.routed_expert_norm.forward = norm

    def optional(value, weight, addend):
        assert events == ["reduced", "normed"]
        events.append("tail")
        return F.linear(value, weight) + addend

    model._tp.try_linear_all_gather_add = optional
    model.routed_expert_up_proj.forward = lambda x: pytest.fail("native success also projected")
    torch.testing.assert_close(model.forward(x), expected, rtol=0, atol=0)
    assert events == ["reduced", "normed", "tail"]


@pytest.mark.parametrize("size,capability", [(1, (10, 3)), (2, (10, 3)), (8, (9, 0)), (8, (10, 3))])
def test_factory_compiles_before_comm_and_unsupported_worlds_keep_original_path(
    monkeypatch, size, capability
):
    import minisgl.distributed.sgl as module

    events = []
    cpu, gpu = object(), object()
    kernel = Kernel()

    class Native:
        disabled = False
        has_multicast = True
        world_size = size
        max_push_size = 65536
        config = SimpleNamespace(num_push_blocks=128)
        obj = object()

        def __init__(self, *a, **k):
            events.append("comm")

    def compile():
        events.append("jit")
        return kernel

    original = importlib.import_module

    def vendor(name, *a, **k):
        if name.endswith("custom_all_reduce_v2"):
            return SimpleNamespace(CustomAllReduceV2=Native)
        if name.endswith("gemm_ag"):
            return SimpleNamespace(_jit_module=compile, K=3584, N=7168, MAX_TOKENS=12)
        return original(name, *a, **k)

    monkeypatch.setattr(module.importlib, "import_module", vendor)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: capability)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda g: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda g: size)
    monkeypatch.setattr(torch.distributed, "get_process_group_ranks", lambda g: list(range(size)))
    monkeypatch.setattr(module.TorchDistributedImpl, "warmup", lambda *a: events.append("warm"))
    ffi = ModuleType("tvm_ffi")
    ffi.use_torch_stream = nullcontext
    monkeypatch.setitem(sys.modules, "tvm_ffi", ffi)
    result = module.create_provider(
        tp_info=DistributedInfo(0, size),
        cpu_group=cpu,
        device_group=gpu,
        device=torch.device("cuda:0"),
        max_bytes=65536,
    )
    eligible = size == 8 and capability == (10, 3)
    assert events == (["jit"] if eligible else []) + ["warm", "comm"]
    assert (result._gemm_module is kernel) == eligible
    if eligible:
        events.clear()

        def fail():
            events.append("jit")
            raise RuntimeError("compile failed")

        compile = fail
        with pytest.raises(RuntimeError, match="compile failed"):
            module.create_provider(
                tp_info=DistributedInfo(0, size),
                cpu_group=cpu,
                device_group=gpu,
                device=torch.device("cuda:0"),
                max_bytes=65536,
            )
        assert events == ["jit"]


@pytest.mark.parametrize("strict,size", [(True, 8), (False, 1)])
def test_strict_and_tp1_loading_preserve_old_storage_and_strict_forward(monkeypatch, strict, size):
    import minisgl.core as core

    monkeypatch.setattr(
        core,
        "_GLOBAL_CTX",
        core.Context(page_size=1, bf16_precision="strict" if strict else "default"),
    )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = LatentMoE(model_config())
    finally:
        torch.set_default_dtype(previous)
    model._tp.group = DistributedInfo(0, size)
    monkeypatch.setattr(
        Linear, "prepare_replicated_add", lambda op: pytest.fail("strict/TP1 storage prepared")
    )
    weights = {k: torch.full_like(v, 0.125) for k, v in model.state_dict().items()}
    model.load_state_dict(weights)
    if strict:
        model._tp.all_reduce = lambda x: x
        model._tp.try_linear_all_gather_add = lambda *a: pytest.fail(
            "strict reference entered new native path"
        )
        calls = []
        original = model.routed_expert_up_proj.forward
        model.routed_expert_up_proj.forward = lambda x: calls.append("linear") or original(x)
        result = model.forward(torch.ones(2, 4).bfloat16())
        assert calls == ["linear"] and torch.isfinite(result).all()
