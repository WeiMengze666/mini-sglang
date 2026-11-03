"""The packed executor must round route scalars and products at their boundary."""

import os
from types import SimpleNamespace

import pytest
import torch
from minisgl.layers.linear import Linear

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1" or not torch.cuda.is_available(),
    reason="explicit GPU precision gate only",
)


def expert(down_code):
    projections = {}
    for name, code in (("w1", 2), ("w3", 2), ("w2", down_code)):
        op = Linear(128, 128, weight_format="mxfp4", device="cuda")
        op.weight_packed.zero_()
        # E2M1 low nibble: 2 = 1, 1 = 0.5. Every output reads input column 0.
        op.weight_packed[:, 0] = code
        op.weight_scale.fill_(127)  # E8M0 scale 2**0.
        projections[name] = op
    return SimpleNamespace(**projections)


@pytest.mark.parametrize("tokens", [1, 3, 17])
@pytest.mark.parametrize("strict", [False, True])
def test_route_scalar_product_and_sum_storage(monkeypatch, tokens, strict):
    import minisgl.moe.mxfp4 as module
    from minisgl.moe.routing import RoutingPolicy

    monkeypatch.setattr(module, "use_strict_bf16", lambda: strict)
    executor = module.MXFP4Experts(
        [(0, expert(2)), (1, expert(1))],
        activation="silu",
        routing_policy=RoutingPolicy(stable=True),
    )
    x = torch.zeros(tokens, 128, device="cuda", dtype=torch.bfloat16)
    x[:, 0] = 1
    ids = torch.tensor([[0, 1], [1, 0]], device="cuda", dtype=torch.int32).repeat(
        (tokens + 1) // 2, 1
    )[:tokens]
    weights = (
        torch.tensor([[0.2999, 0.7001]], device="cuda", dtype=torch.float32)
        .expand(tokens, -1)
        .contiguous()
    )
    # All GEMMs are exact sparse copies, so this oracle isolates routing
    # precision from tile/accumulation choices and dequantization algorithms.
    activated = torch.nn.functional.silu(torch.tensor(1.0)).to(torch.bfloat16).float()
    values = (activated * torch.tensor([1.0, 0.5], device="cuda"))[ids.long()]
    if strict:
        expected = (values * weights.to(torch.bfloat16).float()).to(torch.bfloat16).float().sum(1)
    else:
        expected = (values * weights).sum(1)
    expected = expected[:, None].expand(-1, 128)
    actual = executor.forward(x, weights, ids)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == torch.float32
    default = (values * weights).sum(1)
    strict_value = (values * weights.to(torch.bfloat16).float()).to(torch.bfloat16).float().sum(1)
    assert (default != strict_value).any(), "fixture must detect missing BF16 rounding"
    assert torch.equal(
        ids,
        torch.tensor([[0, 1], [1, 0]], device="cuda", dtype=torch.int32).repeat(
            (tokens + 1) // 2, 1
        )[:tokens],
    )
