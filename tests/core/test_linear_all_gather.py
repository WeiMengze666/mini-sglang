"""CPU AG capability/ABI/ownership checks; no native CUDA or performance evidence."""

from contextlib import contextmanager
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
from minisgl.distributed.multimem.provider import MultimemProvider
from minisgl.distributed.parallel import TensorParallel
from minisgl.layers.linear import Linear
from minisgl.moe import create_moe_backend
from test_collective_outputs import cuda_tensor


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("CPU AG check initialized CUDA")

    monkeypatch.setattr(torch.cuda, "init", forbid)
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


@pytest.fixture(scope="module")
def full_weight():
    return cuda_tensor(7168, 3584)


@pytest.fixture
def native(full_weight):
    calls = []
    comm = SimpleNamespace(
        obj=object(),
        disabled=False,
        has_multicast=True,
        world_size=8,
        max_push_size=512 * 1024,
        config=SimpleNamespace(num_push_blocks=96),
    )

    @contextmanager
    def scope():
        calls.append("enter")
        yield
        calls.append("exit")

    provider = MultimemProvider(comm, device=torch.device("cuda:0"), execution_context=scope)

    class Kernel:
        fail = False

        def run(self, *args):
            calls.append(args)
            if self.fail:
                raise RuntimeError("unknown native failure")
            args[-1].fill_(9)

    kernel = Kernel()

    def jit():
        calls.append("jit")
        return kernel

    provider._linear_all_gather_ops = SimpleNamespace(
        K=3584, N=7168, MAX_TOKENS=12, _jit_module=jit
    )
    return provider, cuda_tensor(2, 3584), full_weight, cuda_tensor(2, 7168), calls, kernel


@pytest.mark.parametrize(
    "rows,blocks,window,expected",
    [
        (1, 8, 1792, True),
        (12, 85, 21504, True),
        (0, 96, 512 * 1024, False),
        (13, 96, 512 * 1024, False),
        (12, 84, 21504, False),
        (12, 85, 21503, False),
        (1, 96, 1 << 29, False),
    ],
)
def test_sdk_row_counter_and_phase_window_limits(native, rows, blocks, window, expected):
    p, _, w, _, calls, _ = native
    p.comm.config.num_push_blocks = blocks
    p.comm.max_push_size = window
    assert (
        p.supports_linear_all_gather(cuda_tensor(rows, 3584), w, cuda_tensor(rows, 7168))
        is expected
    )
    assert calls == []


@pytest.mark.parametrize(
    "case", ["cpu", "dtype", "sharded_weight", "addend_shape", "world", "multicast", "disabled"]
)
def test_metadata_declines_before_jit(native, case):
    p, x, w, b, calls, _ = native
    if case == "cpu":
        x = torch.ones((2, 3584), dtype=torch.bfloat16)
    elif case == "dtype":
        b = b.float()
    elif case == "sharded_weight":
        w = w[:896]
    elif case == "addend_shape":
        b = b[:1]
    elif case == "world":
        p.comm.world_size = 4
    elif case == "multicast":
        p.comm.has_multicast = False
    else:
        p.comm.disabled = True
    assert not p.supports_linear_all_gather(x, w, b)
    assert calls == []


def test_exact_instance_abi_owned_result_and_read_only_operands(native):
    p, x, w, b, calls, _ = native
    result = p.linear_all_gather(x, w, b)
    assert calls[0:2] == ["jit", "enter"] and calls[-1] == "exit"
    args = calls[2]
    assert len(args) == 6 and args[0] is p.comm.obj
    assert args[1] is x and args[2] is w and args[3] is b and args[4] is None and args[5] is result
    assert result.shape == b.shape and result.eq(9).all()
    assert all(not torch._C._overlaps(result, a) for a in (x, w, b))
    assert x.eq(1).all() and b.eq(1).all()
    saved = result.clone()
    x.zero_()
    b.zero_()
    assert torch.equal(saved, result)


@pytest.mark.parametrize("case", ["strided", "unaligned", "device", "grad"])
def test_local_storage_violation_keeps_uniform_capability_and_raises(native, case):
    p, x, w, b, calls, _ = native
    if case == "strided":
        x = cuda_tensor(3584, 2).t()
    elif case == "unaligned":
        x = cuda_tensor(2 * 3584 + 1)[1:].view(2, 3584)
    elif case == "device":
        p.device = torch.device("cuda:1")
    else:
        x.requires_grad_()
    assert p.supports_linear_all_gather(x, w, b)
    with pytest.raises(ValueError, match="aligned contiguous local inference"):
        p.linear_all_gather(x, w, b)
    assert calls == []


def test_warmup_guard_and_selected_error_never_retry(native, monkeypatch):
    p, x, w, b, calls, kernel = native
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="Warm native"):
        p.linear_all_gather(x, w, b)
    assert calls == []
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    p.linear_all_gather(x, w, b)
    assert calls.count("jit") == 1
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    p.linear_all_gather(x, w, b)
    assert calls.count("jit") == 1
    calls.clear()
    kernel.fail = True

    class Other(DistributedImpl):
        def all_reduce(self, x):
            return x

        def all_gather(self, x):
            return x

        def supports_linear_all_gather(self, *args):
            return True

        def linear_all_gather(self, *args):
            pytest.fail("retried after native error")

    backend = FallbackDistributedImpl(MNNVLDistributedImpl(p), Other())
    with pytest.raises(RuntimeError, match="unknown native failure"):
        backend.linear_all_gather(x, w, b)
    assert len([c for c in calls if isinstance(c, tuple)]) == 1


def test_base_old_provider_and_singleton_decline_without_cuda():
    class Base(DistributedImpl):
        def all_reduce(self, x):
            return x

        def all_gather(self, x):
            return x

    x, w, b = torch.ones(1, 2), torch.ones(3, 2), torch.ones(1, 3)
    for impl in (Base(), MNNVLDistributedImpl(SimpleNamespace())):
        assert not impl.supports_linear_all_gather(x, w, b)
        with pytest.raises(NotImplementedError):
            impl.linear_all_gather(x, w, b)
    tp = TensorParallel()
    tp.group = DistributedInfo(0, 1)
    assert not tp.supports_linear_all_gather(x, w, b)
    with pytest.raises(NotImplementedError):
        tp.linear_all_gather(x, w, b)


def test_generic_facades_forward_all_three_operands(native, monkeypatch):
    p, x, w, b, calls, _ = native
    backend = MNNVLDistributedImpl(p)
    monkeypatch.setattr(DistributedCommunicator, "plugins", [backend])
    tp = TensorParallel()
    tp.group = DistributedInfo(0, 8)
    assert tp.supports_linear_all_gather(x, w, b)
    result = tp.linear_all_gather(x, w, b)
    assert result.eq(9).all() and calls[2][2] is w


def test_prepared_projection_uses_native_and_keeps_owned_output(monkeypatch):
    import minisgl.distributed.layout as layout

    monkeypatch.setattr(layout, "get_parallel_layout", lambda: SimpleNamespace(ep_size=1))
    op = Linear(2, 4)
    op.weight = cuda_tensor(4, 2)
    x, b = cuda_tensor(2, 2), cuda_tensor(2, 4)
    calls = []

    class Parallel:
        def supports_linear_all_gather(self, a, w, c):
            calls.append(("supports", a, w, c))
            return True

        def linear_all_gather(self, a, w, c):
            calls.append(("native", a, w, c))
            return a @ w.t() + c

    monkeypatch.setattr(op, "forward", lambda x: pytest.fail("replicated GEMM ran"))
    result = create_moe_backend("fused").epilogue.project_outputs(
        Parallel(), x, b, op, output_dtype=torch.bfloat16, prepared=True
    )
    assert [c[0] for c in calls] == ["supports", "native"]
    assert calls[1][2] is op.weight and result.eq(3).all()
    assert not torch._C._overlaps(result, b)
    prefix = torch.full_like(result, 7)
    result_with_prefix = create_moe_backend("fused").epilogue.project_outputs(
        Parallel(), x, b, op, output_dtype=torch.bfloat16, prepared=True, residual=prefix
    )
    assert result_with_prefix.eq(10).all() and prefix.eq(7).all() and b.eq(1).all()
    assert [c[0] for c in calls] == ["supports", "native", "supports", "native"]


@pytest.mark.parametrize(
    "case", ["unprepared", "cpu", "ep", "bias", "quant", "fp32", "subclass", "decline"]
)
def test_noneligible_projection_keeps_original_forward_and_shared_cast(monkeypatch, case):
    import minisgl.distributed.layout as layout

    monkeypatch.setattr(
        layout, "get_parallel_layout", lambda: SimpleNamespace(ep_size=2 if case == "ep" else 1)
    )

    class Custom(Linear):
        pass

    width = 32 if case == "quant" else 2
    op = (Custom if case == "subclass" else Linear)(
        width,
        4,
        has_bias=case == "bias",
        weight_format="mxfp4" if case == "quant" else "dense",
    )
    if case != "quant":
        op.weight = cuda_tensor(4, 2)
    x = cuda_tensor(2, width)
    shared = cuda_tensor(2, 4)
    if case == "cpu":
        x = torch.ones((2, 2), dtype=torch.bfloat16)
    if case == "fp32":
        x = x.float()
    calls = []

    def forward(value):
        calls.append("forward")
        return torch.full((2, 4), 1.003, dtype=torch.float32)

    monkeypatch.setattr(op, "forward", forward)

    class Parallel:
        def supports_linear_all_gather(self, *args):
            assert case == "decline"
            calls.append("supports")
            return False

        def linear_all_gather(self, *args):
            pytest.fail("native called")

    # Preserve the original cast target even when custom forward returns FP32.
    shared = shared.float() * 1.003
    if case == "decline":
        shared = shared.bfloat16()
    if case not in ("cpu", "fp32", "decline"):
        shared = shared.bfloat16()
    result = create_moe_backend("fused").epilogue.project_outputs(
        Parallel(), x, shared, op, output_dtype=x.dtype, prepared=case != "unprepared"
    )
    expected = forward(x) + shared.to(x.dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert calls.count("forward") == 2


def test_projection_without_up_or_shared_preserves_existing_identity():
    backend = create_moe_backend("fused")
    x = torch.ones((2, 3), dtype=torch.bfloat16)
    assert backend.epilogue.project_outputs(SimpleNamespace(), x, None) is x
