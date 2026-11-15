"""CPU protocol checks for caller-owned local MoE partials; no CUDA kernels."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.core import Context
from minisgl.layers.latent_moe import LatentMoE
from minisgl.layers.linear import Linear
from minisgl.moe import create_moe_backend


class OutputExecutor:
    supports_cuda_graph = True
    input_dtype = torch.bfloat16

    def __init__(self):
        self.outputs = []

    def supports_out(self, dtype):
        return dtype == torch.bfloat16

    def forward(self, x, weights, ids, *, out=None):
        result = x.to(torch.bfloat16) * 2 + 1
        self.outputs.append(out)
        if out is not None:
            out.copy_(result)
            return out
        return result


class Parallel:
    def __init__(self, *, inplace=False):
        self.outputs, self.events = {}, []
        self.inplace = inplace

    def reserve_output(self, key, shape, dtype, *, enabled=True):
        self.events.append(("reserve", key, shape, dtype, enabled))
        if not enabled:
            return None
        if key not in self.outputs:
            self.outputs[key] = torch.empty(shape, dtype=dtype)
        return self.outputs[key]

    def all_reduce(self, x):
        self.events.append(("generic",))
        return x * 2

    def all_reduce_residual(self, x, residual=None, *, norm=None, output_key=None):
        assert residual is None and output_key in self.outputs
        assert x.is_contiguous()
        self.events.append(("prepared", output_key))
        reduced = x.mul_(2) if self.inplace else x * 2
        return norm(reduced) if norm is not None else reduced


@pytest.fixture(autouse=True)
def no_cuda(monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("CPU MoE output tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbid)
    monkeypatch.setattr(torch.cuda, "init", forbid)


@pytest.fixture
def prepared(monkeypatch):
    import minisgl.core as core

    ctx = Context(page_size=1, max_forward_len=6)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    return ctx


def layer():
    config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=6,
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
    with torch.device("cpu"):
        old = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            op = LatentMoE(config)
        finally:
            torch.set_default_dtype(old)
    generator = torch.Generator().manual_seed(13)
    for name, tensor in op.state_dict().items():
        tensor.copy_(torch.randn(tensor.shape, generator=generator).to(tensor.dtype) * 0.25)
        if name.endswith("norm.weight"):
            tensor.fill_(1)
    op._expert_executor = OutputExecutor()
    return op


@pytest.mark.parametrize("inplace", [False, True])
def test_packed_blocks_match_original_reduction_and_keep_owned_result(prepared, inplace):
    op = layer()
    op._tp = Parallel(inplace=inplace)
    x = torch.arange(12, dtype=torch.bfloat16).view(3, 4) / 8
    expected = op.forward(x)
    op.prepare_inference()
    buffer = op._output
    assert buffer.shape == (6, 6)
    out = op.forward(x)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    partial = op._expert_executor.outputs[-1]
    assert partial.shape == (3, 2) and partial.is_contiguous()
    assert partial.data_ptr() == buffer.data_ptr()
    # The shared down GEMM writes the block immediately after all latent rows.
    assert partial.untyped_storage().data_ptr() == buffer.untyped_storage().data_ptr()
    assert op._tp.events[-1][0] == "prepared"
    assert len([event for event in op._tp.events if event[0] == "prepared"]) == 1
    saved = out.clone()
    op.forward(x + 1)
    assert op._output is buffer
    assert torch.equal(out, saved)  # Final norm/up/add owns storage across role reuse.
    assert not torch._C._overlaps(out, buffer)


def test_same_geometry_layers_share_one_full_capacity_role(prepared):
    first, second = layer(), layer()
    tp = Parallel()
    first._tp = second._tp = tp
    first.prepare_inference()
    second.prepare_inference()
    assert first._output is second._output
    assert len(tp.outputs) == 1 and tp.events[0][2] == (6, 6)


@pytest.mark.parametrize("decline", ["executor", "shared", "ep"])
def test_static_decline_still_joins_reservation_before_any_forward(prepared, monkeypatch, decline):
    import minisgl.distributed.layout as layout

    op = layer()
    op._tp = Parallel()
    if decline == "executor":
        op._expert_executor = SimpleNamespace()
    elif decline == "shared":
        op.shared_experts.down_proj = Linear(32, 4, weight_format="mxfp4")
    else:
        monkeypatch.setattr(layout, "get_parallel_layout", lambda: SimpleNamespace(ep_size=2))
    op.prepare_inference()
    assert op._output is None and op._reserved_key is None
    assert op._tp.events == [
        ("reserve", "moe.output:2:4:torch.bfloat16", (6, 6), torch.bfloat16, False)
    ]


@pytest.mark.parametrize("reason", ["unprepared", "oversize", "dtype"])
def test_runtime_unsupported_uses_original_path_without_touching_slot(prepared, reason):
    op = layer()
    op._tp = Parallel()
    if reason != "unprepared":
        op.prepare_inference()
        op._output.fill_(-17)
    if reason == "dtype":
        x = torch.ones(2, 4, dtype=torch.float32)
        # A different model precision is represented by its matching parameters.
        for tensor in op.state_dict().values():
            tensor.data = tensor.data.float()
        op._expert_executor.input_dtype = torch.float32
        assert op.gate.weight.dtype == torch.float32
    else:
        x = torch.ones(7 if reason == "oversize" else 2, 4, dtype=torch.bfloat16)
    op.forward(x)
    assert op._expert_executor.outputs[-1] is None
    assert op._tp.events[-1] == ("generic",)
    if op._output is not None:
        assert op._output.eq(-17).all()


@pytest.mark.parametrize("ep,capability", [(2, True), (1, False)])
def test_explicit_unsupported_out_rejects_before_routes(ep, capability, monkeypatch):
    import minisgl.distributed.layout as layout

    monkeypatch.setattr(layout, "get_parallel_layout", lambda: SimpleNamespace(ep_size=ep))
    executor = OutputExecutor() if capability else SimpleNamespace()
    routing = SimpleNamespace(select=lambda backend: pytest.fail("routed before eligibility"))
    with pytest.raises(NotImplementedError, match="plain TP"):
        create_moe_backend("fused").forward_experts(
            executor, torch.zeros(2, 2), routing, out=torch.empty(2, 2, dtype=torch.bfloat16)
        )
    if capability:
        assert executor.outputs == []


def test_reload_discards_prepared_destination(prepared):
    op = layer()
    op._tp = Parallel()
    op.prepare_inference()
    assert op._output is not None
    op.after_load()
    assert op._output is None and op._reserved_key is None


@pytest.mark.parametrize("invalid", ["length", "offset", "strided", "dtype"])
def test_backend_rejects_inconsistent_packed_blocks_before_collective(invalid):
    backend = create_moe_backend("fused")
    tp = Parallel()
    packed = torch.arange(18, dtype=torch.bfloat16)
    latent, shared = packed[:6].view(3, 2), packed[6:].view(3, 4)
    if invalid == "length":
        packed = packed[:-1]
    elif invalid == "offset":
        shared = shared.clone()
    elif invalid == "strided":
        latent = latent.T
    else:
        shared = shared.float()
    with pytest.raises(ValueError, match="packed|consecutive"):
        backend.reduce_outputs(tp, latent, shared, packed=packed, output_key="owned")
    assert tp.events == []
