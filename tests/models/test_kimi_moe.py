from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.models.kimi_moe import KimiK3MLP, KimiK3MoE, KimiRouter, situ_and_mul


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
        "rms_norm_eps": 1e-5,
        "num_shared_experts": 1,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def initialize(op):
    torch.manual_seed(13)
    for name, tensor in op.state_dict().items():
        tensor.copy_(torch.randn_like(tensor) * 0.4)
        if name.endswith("norm.weight"):
            tensor.fill_(1.0)


def test_situ_reference_and_saturation():
    gate = torch.tensor([-100.0, -4.0, 0.0, 4.0, 100.0])
    up = torch.tensor([100.0, -25.0, 1.0, 25.0, 100.0])
    expected = 4 * torch.tanh(gate / 4) * gate.sigmoid() * 25 * torch.tanh(up / 25)
    torch.testing.assert_close(situ_and_mul(gate, up, 4.0, 25.0), expected)
    assert situ_and_mul(gate, up, 4.0, 25.0)[-1].item() < 100.0
    torch.testing.assert_close(
        situ_and_mul(gate, up, 4.0), 4 * torch.tanh(gate / 4) * gate.sigmoid() * up
    )
    with pytest.raises(ValueError):
        situ_and_mul(gate, up, 0.0)


def test_router_bias_selects_but_does_not_weight():
    c = config(routed_scaling_factor=2.0)
    router = KimiRouter(c)
    router.weight.zero_()
    router.weight[:, 0] = torch.logit(torch.tensor([0.9, 0.8, 0.1, 0.2]))
    router.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, 2.0, 2.0]))
    weights, ids = router.forward(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert set(ids[0].tolist()) == {2, 3}
    expected = torch.tensor([[0.9, 0.8, 0.1, 0.2]]).gather(1, ids)
    torch.testing.assert_close(weights, expected / expected.sum(-1, keepdim=True) * 2)


def test_latent_moe_full_manual_reference_and_token_independence():
    c = config()
    model = KimiK3MoE(c)
    initialize(model)
    x = torch.randn(3, 4)
    weights, ids = model.gate.forward(x)
    routed = F.linear(x, model.routed_expert_down_proj.weight)
    expected_rows = []
    for row in range(len(x)):
        mixture = torch.zeros(2)
        for slot in range(c.num_experts_per_token):
            expert = model.experts.op_list[ids[row, slot]]
            g, u = F.linear(routed[row], expert.w1.weight), F.linear(routed[row], expert.w3.weight)
            activation = 4 * torch.tanh(g / 4) * torch.sigmoid(g) * 25 * torch.tanh(u / 25)
            mixture += weights[row, slot] * F.linear(activation, expert.w2.weight)
        mixture = mixture * torch.rsqrt(mixture.square().mean() + c.rms_norm_eps)
        mixture *= model.routed_expert_norm.weight
        expected_rows.append(F.linear(mixture, model.routed_expert_up_proj.weight))
    expected = torch.stack(expected_rows) + model.shared_experts.forward(x)
    torch.testing.assert_close(model.forward(x), expected)
    torch.testing.assert_close(model.forward(x), torch.cat([model.forward(row[None]) for row in x]))
    assert model.forward(x[:0]).shape == (0, 4)


def test_dense_and_expert_checkpoint_names_are_unfused():
    mlp = KimiK3MLP(config())
    assert set(mlp.state_dict()) == {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    model = KimiK3MoE(config())
    names = model.state_dict()
    assert "experts.0.w1.weight" in names
    assert "experts.3.w2.weight" in names
    assert "experts.1.w3.weight" in names
    assert names["gate.e_score_correction_bias"].dtype == torch.float32
    assert "routed_expert_norm.weight" in names
    clone = KimiK3MoE(config())
    clone.load_state_dict({k: torch.zeros_like(v) for k, v in names.items()})
    assert all(torch.count_nonzero(v) == 0 for v in clone.state_dict().values())


@pytest.mark.parametrize("latent,shared", [(None, 0), (2, 0), (None, 2)])
def test_optional_latent_and_shared_branches(latent, shared):
    model = KimiK3MoE(config(routed_expert_hidden_size=latent, num_shared_experts=shared))
    initialize(model)
    output = model.forward(torch.randn(2, 3, 4))
    assert output.shape == (2, 3, 4)
    assert torch.isfinite(output).all()
