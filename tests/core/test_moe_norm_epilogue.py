"""CPU semantics and native ABI selection, never CUDA execution."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed.impl import DistributedImpl
from minisgl.layers.norm import RMSNormEpilogue, TorchRMSNorm
from minisgl.moe import create_moe_backend
from test_collective_outputs import cuda_tensor
from test_collective_outputs import host as collective_host

# Register the already-reviewed CPU communicator fixture locally.
host = collective_host


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("CPU norm epilogue test initialized CUDA")

    monkeypatch.setattr(torch.cuda, "init", forbid)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)


def normalizer(width, dtype=torch.bfloat16):
    op = TorchRMSNorm(width, 1e-5)
    op.weight = torch.linspace(0.5, 1.5, width, dtype=dtype)
    return op


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_callable_fallback_norms_only_prefix_and_keeps_borrowed_storage(dtype):
    norm = normalizer(4, dtype)
    x = torch.linspace(-2, 3, 15, dtype=dtype)
    before = x.clone()
    expected = norm.forward(before[:8].view(2, 4))
    epilogue = RMSNormEpilogue(norm, 2)
    assert epilogue.weight is norm.weight and epilogue.eps == norm.eps and epilogue.width == 4
    assert epilogue(x) is x
    torch.testing.assert_close(x[:8].view(2, 4), expected, rtol=0, atol=0)
    assert torch.equal(x[8:].view(torch.uint8), before[8:].view(torch.uint8))
    # Width4 and arbitrary7-element tail demonstrate no SDK3584 rule in this abstraction.


def test_zero_norm_rows_is_a_noop_on_the_entire_borrowed_tail():
    norm = normalizer(4)
    norm.forward = lambda value: pytest.fail("zero rows invoked the normalizer")
    value = torch.arange(7, dtype=torch.bfloat16)
    before = value.clone()
    assert RMSNormEpilogue(norm, 0)(value) is value
    assert torch.equal(value.view(torch.uint8), before.view(torch.uint8))


def test_existing_distributed_callable_composition_needs_no_new_contract():
    transport = SimpleNamespace(all_reduce=lambda value: value * 3)
    x = torch.arange(12, dtype=torch.bfloat16) / 8 + 1
    norm = normalizer(4)
    expected = x * 3
    expected[:8].copy_(norm.forward(expected[:8].view(2, 4)).view(-1))
    result = DistributedImpl.all_reduce_residual(transport, x, norm=RMSNormEpilogue(norm, 2))
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("invalid", ["rows", "range", "stride", "dtype", "weight_alias"])
def test_callable_rejects_invalid_inplace_storage(invalid):
    norm = normalizer(4)
    x = torch.ones(12, dtype=torch.bfloat16)
    rows = -1 if invalid == "rows" else 4 if invalid == "range" else 2
    if invalid == "stride":
        x = torch.ones(24, dtype=torch.bfloat16)[::2]
    elif invalid == "dtype":
        x = x.float()
    elif invalid == "weight_alias":
        norm.weight = x[:4]
    with pytest.raises(ValueError):
        RMSNormEpilogue(norm, rows)(x)


@pytest.mark.parametrize("packed", [False, True])
def test_fp32_partials_reduce_before_cast_and_norm(packed):
    backend = create_moe_backend("fused")
    mixed = torch.tensor([[1.003, 0.995, -0.7, 0.2]], dtype=torch.float32)
    shared = torch.tensor([[0.017, -0.07]], dtype=torch.float32)
    storage = torch.cat((mixed.flatten(), shared.flatten()))
    if packed:
        mixed, shared = storage[:4].view_as(mixed), storage[4:].view_as(shared)
    before = mixed.clone()
    norm = normalizer(4)
    calls = []
    original = norm.forward

    def observe(value):
        calls.append(value.clone())
        return original(value)

    norm.forward = observe

    def reduce(value):
        assert value.dtype == torch.float32
        return value * 3

    def residual(value, *, norm=None, output_key=None):
        assert norm is None, "FP32 local mixtures must not enter a BF16 native epilogue"
        return reduce(value)

    tp = SimpleNamespace(all_reduce=reduce, all_reduce_residual=residual)
    out, tail = backend.reduce_outputs(
        tp,
        mixed,
        shared,
        packed=storage if packed else None,
        output_key="role" if packed else None,
        norm=norm,
        output_dtype=torch.bfloat16,
    )
    assert len(calls) == 1 and calls[0].dtype == torch.bfloat16
    torch.testing.assert_close(calls[0], (before * 3).bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(out, original((before * 3).bfloat16()), rtol=0, atol=0)
    torch.testing.assert_close(tail, shared * 3, rtol=0, atol=0)


def test_prepared_bf16_backend_builds_one_explicit_callable_not_a_global_handoff():
    backend = create_moe_backend("fused")
    storage = torch.arange(12, dtype=torch.bfloat16) / 8 + 0.25
    mixed, shared = storage[:8].view(2, 4), storage[8:].view(2, 2)
    norm = normalizer(4)
    expected = norm.forward((mixed * 2).clone())
    expected_shared = shared.clone() * 2
    calls, norm_calls = [], []
    original = norm.forward

    def observe(value):
        norm_calls.append(value.clone())
        return original(value)

    norm.forward = observe

    def reduce(value, *, norm=None, output_key=None):
        assert isinstance(norm, RMSNormEpilogue) and norm.num_norm_rows == 2
        assert norm.width == 4 and output_key == "role"
        calls.append(norm)
        return norm(value * 2)

    out, tail = backend.reduce_outputs(
        SimpleNamespace(all_reduce_residual=reduce),
        mixed,
        shared,
        packed=storage,
        output_key="role",
        norm=norm,
        output_dtype=torch.bfloat16,
    )
    assert len(calls) == len(norm_calls) == 1
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(tail, expected_shared, rtol=0, atol=0)


@pytest.fixture
def native_norm(host):
    provider, events, _ = host
    provider.comm.config = SimpleNamespace(num_push_blocks=128)
    calls, selections = [], []

    class Kernels:
        def push_norm(self, *args):
            calls.append(("push_norm", args))

        def pull_norm(self, *args):
            calls.append(("pull_norm", args))

    marker = object()

    def tuning(table, **kwargs):
        assert table is marker  # RES_TUNING cannot stand in for norm tuning.
        selections.append(kwargs)
        return SimpleNamespace(num_blocks=12, unroll=2)

    provider.__dict__.update(
        _residual_kernel=Kernels(),
        _push_max_bytes=512 << 10,
        _residual_ops=SimpleNamespace(
            NORM_TUNING=marker, RES_TUNING=object(), _resolve_tuning=tuning
        ),
    )
    norm = normalizer(3584)
    norm.weight = norm.weight.as_subclass(type(cuda_tensor(1)))
    return provider, RMSNormEpilogue(norm, 1), calls, selections


@pytest.mark.parametrize("rows,route", [(1, "push_norm"), (48, "pull_norm")])
def test_sdk_norm_dispatch_keeps_full_packed_input_and_exact_prefix(native_norm, rows, route):
    provider, epilogue, calls, selections = native_norm
    epilogue = RMSNormEpilogue(epilogue.normalizer, rows)
    full = provider.reserve_output("role", (64, 10752), torch.bfloat16)
    x = full[:rows].view(-1)
    assert provider.supports_all_reduce_residual(x, norm=epilogue, output_key="role")
    assert provider.all_reduce_residual(x, norm=epilogue, output_key="role") is x
    name, args = calls[0]
    assert name == route and args[0] is provider.comm.obj and args[1].data_ptr() == x.data_ptr()
    assert args[1].numel() == rows * 10752 and args[2] is epilogue.weight
    assert args[3:5] == (epilogue.eps, rows)
    if route == "pull_norm":
        assert args[5:] == (provider._output_mc_ptr(x, key="role"), 12, 2)
        assert selections == [{"nbytes": x.nbytes, "num_blocks": None, "unroll": None}]
    else:
        assert selections == []


def test_zero_norm_rows_still_reduces_the_whole_nonempty_native_buffer(native_norm):
    provider, epilogue, calls, _ = native_norm
    value = cuda_tensor(10752)
    epilogue = RMSNormEpilogue(epilogue.normalizer, 0)
    assert provider.supports_all_reduce_residual(value, norm=epilogue)
    provider.all_reduce_residual(value, norm=epilogue)
    assert calls[0][0] == "push_norm" and calls[0][1][4] == 0
    assert calls[0][1][1].numel() == value.numel()


@pytest.mark.parametrize("invalid", ["device", "stride", "alignment", "alias", "mc_alias"])
def test_local_errors_raise_after_uniform_capability_selection(native_norm, invalid):
    provider, epilogue, calls, _ = native_norm
    rows = 48 if invalid == "mc_alias" else 1
    x = cuda_tensor(rows * 10752)
    epilogue = RMSNormEpilogue(epilogue.normalizer, rows)
    if invalid == "device":
        epilogue.normalizer.weight = torch.ones(3584, dtype=torch.bfloat16)
    elif invalid == "stride":
        epilogue.normalizer.weight = cuda_tensor(7168)[::2]
    elif invalid == "alignment":
        epilogue.normalizer.weight = cuda_tensor(3585)[1:]
    elif invalid == "alias":
        epilogue.normalizer.weight = x[:3584]
    assert provider.supports_all_reduce_residual(x, norm=epilogue, output_key="role")
    with pytest.raises(ValueError, match="weights|reserved multicast"):
        provider.all_reduce_residual(x, norm=epilogue, output_key="role")
    assert calls == []


@pytest.mark.parametrize(
    "decline", ["width", "fp32", "residual", "callable", "range", "push_counters"]
)
def test_semantic_constraints_decline_uniformly(native_norm, decline):
    provider, epilogue, calls, _ = native_norm
    x, residual = cuda_tensor(10752), None
    if decline == "width":
        epilogue = RMSNormEpilogue(normalizer(256), 1)
    elif decline == "fp32":
        epilogue = RMSNormEpilogue(normalizer(3584, torch.float32), 1)
    elif decline == "residual":
        residual = x.clone()
    elif decline == "callable":
        epilogue = lambda value: value
    elif decline == "range":
        epilogue = RMSNormEpilogue(epilogue.normalizer, 4)
    else:
        provider.comm.config.num_push_blocks = 3
    assert not provider.supports_all_reduce_residual(x, residual, norm=epilogue, output_key="role")
    assert calls == []
