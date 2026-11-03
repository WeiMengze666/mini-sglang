"""Precision boundaries of generic gated projections and distributed mixtures."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.layers.latent_moe import GatedMLP, LatentMoE
from minisgl.layers.linear import Linear


def config(**changes):
    values = {
        "hidden_size": 4,
        "intermediate_size": 7,
        "moe_intermediate_size": 3,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "num_expert_group": 2,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "moe_router_activation_func": "sigmoid",
        "moe_renormalize": True,
        "routed_scaling_factor": 1.3,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "routed_expert_hidden_size": 2,
        "latent_moe_use_norm": True,
        "rms_norm_eps": 1e-05,
        "num_shared_experts": 1,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def make(op, dtype, config_):
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        model = op(config_)
    finally:
        torch.set_default_dtype(old)
    generator = torch.Generator().manual_seed(13)
    for name, tensor in model.state_dict().items():
        tensor.copy_(torch.randn(tensor.shape, generator=generator).to(tensor.dtype) * 0.4)
        if name.endswith("norm.weight"):
            tensor.fill_(1)
    return model


@pytest.mark.parametrize("strided", [False, True])
def test_fp32_gate_up_retained_until_activation_rounding(strided):
    layer = make(GatedMLP, torch.bfloat16, config())
    generator = torch.Generator().manual_seed(47)
    x = torch.randn(27, 8, generator=generator).to(torch.bfloat16)
    x = x[:, ::2] if strided else x[:, :4].contiguous()
    # BF16 operands have exact products here; FP64 independently computes the
    # unrounded projections before the FP32 nonlinearity and BF16 storage.
    gate = (x.double() @ layer.gate_proj.weight.double().T).float()
    up = (x.double() @ layer.up_proj.weight.double().T).float()
    activation = 4 * torch.tanh(gate / 4) * torch.sigmoid(gate) * 25 * torch.tanh(up / 25)
    hidden = activation.to(x.dtype)
    expected = (hidden.double() @ layer.down_proj.weight.double().T).to(x.dtype)
    actual = layer.forward_local(x, projection_dtype=torch.float32)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert (actual != layer.forward_local(x)).any(), "fixture must distinguish early rounding"
    assert layer.forward_local(x[:0], projection_dtype=torch.float32).shape == (0, 4)


def test_fp32_gate_up_rejects_unsupported_packed_projection():
    layer = make(GatedMLP, torch.bfloat16, config(hidden_size=32, intermediate_size=32))
    layer.gate_proj = Linear(32, 32, weight_format="mxfp4")
    with pytest.raises(ValueError, match="dense, bias-free"):
        layer.forward_local(
            torch.empty(0, 32, dtype=torch.bfloat16), projection_dtype=torch.float32
        )
    with pytest.raises(ValueError, match="None or float32"):
        layer.forward_local(
            torch.empty(0, 32, dtype=torch.bfloat16), projection_dtype=torch.float16
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("latent,shared", [(2, 1), (None, 0), (None, 1)])
def test_mixture_collective_respects_storage_precision(monkeypatch, dtype, strict, latent, shared):
    import minisgl.layers.latent_moe as module

    monkeypatch.setattr(module, "use_strict_bf16", lambda: strict)
    layer = make(
        LatentMoE, dtype, config(routed_expert_hidden_size=latent, num_shared_experts=shared)
    )
    seen = []
    original = layer._tp.all_reduce

    def observe(x):
        seen.append((x.dtype, x.shape))
        return original(x)

    layer._tp.all_reduce = observe
    x = torch.randn(2, 3, 4).to(dtype)
    actual = layer.forward(x)
    assert seen == [
        (
            torch.bfloat16 if strict and dtype == torch.bfloat16 else torch.float32,
            # Expert and shared blocks retain their own contiguous row layout.
            torch.Size([6 * ((latent or 4) + 4)]) if shared else torch.Size([6, latent or 4]),
        )
    ]
    assert actual.shape == x.shape and actual.dtype == dtype and torch.isfinite(actual).all()


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("required_dtype", [torch.float32, torch.bfloat16])
def test_expert_input_precision_is_negotiated_before_latent_rounding(fused, required_dtype):
    from minisgl.layers.linear import linear_fp32

    layer = make(LatentMoE, torch.bfloat16, config())
    x = torch.randn(7, 4, generator=torch.Generator().manual_seed(198)).bfloat16()
    unrounded = (x.double() @ layer.routed_expert_down_proj.weight.double().T).float()
    assert (unrounded != unrounded.bfloat16().float()).any()
    seen = []

    def expert_forward(routed, weights, ids):
        seen.append(routed.clone())
        return torch.zeros_like(routed, dtype=torch.float32)

    layer._expert_executor = SimpleNamespace(input_dtype=required_dtype, forward=expert_forward)
    if fused:
        # A fused front supplies unrounded slices; its GEMM/storage contract has
        # independent MergedLinear tests. Here the consumer must retain precision.
        projections = layer._front_projections()
        front = {name: linear_fp32(x, op.weight) for name, op in projections.items()}
        layer._front = SimpleNamespace(forward=lambda _: front)
    actual = layer.forward(x)
    assert len(seen) == 1 and seen[0].dtype == required_dtype
    torch.testing.assert_close(seen[0], unrounded.to(required_dtype), rtol=0, atol=0)
    assert actual.dtype == x.dtype and torch.isfinite(actual).all()


def test_router_preserves_backend_slot_order_and_policy():
    layer = make(LatentMoE, torch.float32, config())
    logits = torch.randn(3, 4, generator=torch.Generator().manual_seed(203))
    weights, ids = layer.gate.route(logits)
    expected = weights.flip(-1), ids.flip(-1)
    seen = []

    def route(value, **options):
        assert value is logits
        seen.append(options)
        return expected

    layer.gate._backend = SimpleNamespace(route=route)
    actual = layer.gate.route(logits)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    assert len(seen) == 1
    assert seen[0]["bias"] is layer.gate.e_score_correction_bias
    assert seen[0]["num_groups"] == 2 and seen[0]["topk_groups"] == 1
    assert seen[0]["scoring_func"] == "sigmoid" and seen[0]["scale"] == 1.3
