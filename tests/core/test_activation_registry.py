"""Shared gated activation contracts."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from minisgl.layers.activation import GatedActivation, resolve_gated_activation


def test_resolve_gated_activation_preserves_configured_math():
    gate = torch.tensor([[-2.0, 0.5, 3.0]], dtype=torch.float32)
    up = torch.tensor([[1.5, -2.0, 0.25]], dtype=torch.float32)

    assert torch.equal(resolve_gated_activation("silu").apply(gate, up), F.silu(gate) * up)
    assert torch.equal(resolve_gated_activation("gelu").apply(gate, up), F.gelu(gate) * up)

    config = SimpleNamespace(
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )
    actual = GatedActivation.from_config(config).apply(gate, up)
    expected_gate = 4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)
    expected_up = 25.0 * torch.tanh(up / 25.0)
    assert torch.equal(actual, expected_gate * expected_up)


@pytest.mark.parametrize(
    "name,beta,linear_beta",
    [("unknown", 1.0, None), ("situ", 0.0, None), ("situ", 1.0, 0.0)],
)
def test_resolve_gated_activation_rejects_invalid_contract(name, beta, linear_beta):
    with pytest.raises(ValueError):
        resolve_gated_activation(name, beta=beta, linear_beta=linear_beta)
