"""Merged projection semantics and pre-load storage ownership."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.layers.base import BaseOP
from minisgl.layers.linear import Linear, MergedLinear
from minisgl.layers.weight import allocate_weight_views


@pytest.mark.parametrize("tokens", [0, 5])
@pytest.mark.parametrize("alignment", [1, 8])
def test_merged_linear_matches_fp64_with_strided_inputs(tokens, alignment):
    generator = torch.Generator().manual_seed(170)
    ops = {
        name: Linear(7, width, dtype=torch.bfloat16)
        for name, width in [("gate", 3), ("router", 5), ("latent", 2)]
    }
    original = {}
    for name, op in ops.items():
        op.weight.copy_(torch.randn(op.weight.shape, generator=generator).bfloat16())
        original[name] = op.weight.clone()
    plan = MergedLinear.try_create(ops, pad_rows_to=alignment)
    assert plan is not None
    assert plan.weight.shape[0] % alignment == 0
    assert torch.count_nonzero(plan.weight[10:]) == 0
    x = torch.randn((tokens * 2, 14), generator=generator).bfloat16()[::2, ::2]
    actual = plan.forward(x)
    assert tuple(actual) == tuple(ops)
    for name in ops:
        expected = (x.double() @ original[name].double().T).float()
        torch.testing.assert_close(actual[name], expected, rtol=1e-6, atol=1e-6)
        assert actual[name].dtype == torch.float32


@pytest.mark.parametrize("invalid", ["bias", "packed", "width", "dtype", "grad"])
def test_merged_linear_rejects_unsupported_groups_without_mutation(invalid):
    a = SimpleNamespace(weight=torch.zeros(3, 7), bias=None, weight_format="dense")
    b = SimpleNamespace(weight=torch.zeros(5, 7), bias=None, weight_format="dense")
    if invalid == "bias":
        b.bias = torch.zeros(5)
    elif invalid == "packed":
        b.weight_format = "mxfp4"
    elif invalid == "width":
        b.weight = torch.zeros(5, 8)
    elif invalid == "dtype":
        b.weight = b.weight.bfloat16()
    else:
        b.weight.requires_grad_()
    before = [a.weight, b.weight]
    assert MergedLinear.try_create({"a": a, "b": b}) is None
    assert a.weight is before[0] and b.weight is before[1]


class ProjectionModel(BaseOP):
    def __init__(self):
        self.router = Linear(7, 5, dtype=torch.bfloat16)
        self.other = Linear(7, 2, dtype=torch.bfloat16)
        self.gate = Linear(7, 3, dtype=torch.bfloat16)
        self.latent = Linear(7, 2, dtype=torch.bfloat16)
        self.up = Linear(7, 3, dtype=torch.bfloat16)
        self._plan = MergedLinear.try_create(self.projections())

    def projections(self):
        return {"gate": self.gate, "up": self.up, "router": self.router, "latent": self.latent}

    def after_load(self):
        self._plan = MergedLinear.try_create(self.projections())

    def forward(self, x):
        return self._plan.forward(x)


def test_declared_meta_group_survives_stream_loading_without_weight_copy():
    with torch.device("meta"):
        model = ProjectionModel()
    expected = model.state_dict()
    generator = torch.Generator().manual_seed(171)
    payload = {
        name: torch.randn(value.shape, generator=generator).to(value.dtype)
        for name, value in expected.items()
    }
    model.load_weights(payload.items(), device=torch.device("cpu"))
    assert model.state_dict().keys() == payload.keys()
    plan = model._plan
    start = plan.weight.data_ptr()
    offset = 0
    for name, op in model.projections().items():
        assert op.weight.data_ptr() == start + offset * op.weight.element_size()
        offset += op.weight.numel()
        torch.testing.assert_close(op.weight, payload[name + ".weight"], rtol=0, atol=0)
    assert not start <= model.other.weight.data_ptr() < start + plan.weight.numel() * 2
    assert start % 256 == 0
    assert plan.weight.untyped_storage()._cdata == model.other.weight.untyped_storage()._cdata
    model.after_load()
    assert model._plan.weight.data_ptr() == start
    assert sum(x.numel() for x in model.state_dict().values()) == sum(
        x.numel() for x in payload.values()
    )


def test_weight_arena_preserves_overlapping_contiguous_meta_views():
    storage = torch.empty(20, device="meta", dtype=torch.bfloat16)
    expected = {"a": storage[2:12], "separate": torch.empty(3, device="meta"), "b": storage[8:16]}
    views = allocate_weight_views(expected, torch.device("cpu"))
    assert views["b"].data_ptr() - views["a"].data_ptr() == 6 * views["a"].element_size()
    views["a"].fill_(7)
    torch.testing.assert_close(views["b"][:4], torch.full((4,), 7, dtype=torch.bfloat16))
    assert views["a"].data_ptr() % 256 == views["separate"].data_ptr() % 256 == 0


def test_latent_moe_fused_front_matches_independent_cpu_path(monkeypatch):
    import minisgl.core as core
    from minisgl.layers.latent_moe import LatentMoE

    monkeypatch.setattr(core, "_GLOBAL_CTX", core.Context(page_size=1, bf16_precision="strict"))
    config = SimpleNamespace(
        hidden_size=4,
        intermediate_size=4,
        moe_intermediate_size=4,
        num_experts=4,
        num_experts_per_token=2,
        num_expert_group=1,
        topk_group=1,
        topk_method="noaux_tc",
        moe_router_activation_func="sigmoid",
        moe_renormalize=True,
        routed_scaling_factor=1.5,
        routed_expert_hidden_size=3,
        latent_moe_use_norm=True,
        num_shared_experts=1,
        rms_norm_eps=1e-5,
        hidden_act="situ",
        activation_situ_beta=2.0,
        activation_situ_linear_beta=3.0,
    )
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = LatentMoE(config)
    finally:
        torch.set_default_dtype(previous)
    generator = torch.Generator().manual_seed(172)
    for weight in model.state_dict().values():
        weight.copy_((torch.randn(weight.shape, generator=generator) * 0.2).to(weight.dtype))
    model.apply(lambda op: op.after_load())
    assert model._front is None  # CPU keeps the portable automatic dispatch.
    x = torch.randn((5, 4), generator=generator).bfloat16()
    expected = model.forward(x)
    model._front = MergedLinear.try_create(model._front_projections())
    assert model._front is not None
    width = model.shared_experts.gate_proj.weight.shape[0] * 2
    model.shared_experts._gate_up_weight = model._front.weight[:width]
    torch.testing.assert_close(model.forward(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("alignment", [0, -1])
def test_merged_linear_rejects_invalid_alignment(alignment):
    with pytest.raises(ValueError, match="alignment"):
        MergedLinear.try_create({}, pad_rows_to=alignment)
