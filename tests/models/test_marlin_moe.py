"""Optional mature-kernel adapter against independently generated dense experts."""

import os
from types import SimpleNamespace

import pytest
import torch
from minisgl.layers.linear import Linear
from minisgl.moe.marlin import MarlinMXFP4Experts
from minisgl.moe.routing import RoutingPolicy

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate"
)


def projection(inputs, outputs):
    codes = torch.randint(16, (outputs, inputs), device="cuda", dtype=torch.uint8)
    scales = torch.randint(122, 127, (outputs, inputs // 32), device="cuda", dtype=torch.uint8)
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    dense = table[codes.long()] * (2.0 ** (scales.float() - 127)).repeat_interleave(32, -1)
    op = Linear(inputs, outputs, device="cuda", dtype=torch.bfloat16, weight_format="mxfp4")
    op.weight_packed.copy_(codes[:, ::2] | codes[:, 1::2] << 4)
    op.weight_scale.copy_(scales)
    return op, dense.double()


@pytest.mark.parametrize("hidden,intermediate", [(64, 96), (96, 96), (128, 128)])
@pytest.mark.parametrize("tokens", [1, 27, 65])
@pytest.mark.parametrize("activation", ["silu", "situ"])
@pytest.mark.parametrize("selection", ["mixed", "unowned"])
def test_mature_marlin_experts_and_padding(
    hidden, intermediate, tokens, activation, selection, request
):
    pytest.importorskip("sglang")
    torch.manual_seed(199)
    items, dense = [], []
    for index in (4, 5):
        w1, gate = projection(hidden, intermediate)
        w3, up = projection(hidden, intermediate)
        w2, down = projection(intermediate, hidden)
        items.append((index, SimpleNamespace(w1=w1, w3=w3, w2=w2)))
        dense.append((gate, up, down))
    executor = MarlinMXFP4Experts(
        items,
        activation=activation,
        beta=4.0,
        linear_beta=25.0,
        num_global_experts=8,
        routing_policy=RoutingPolicy(stable=True),
    )
    x = (torch.randn(tokens, hidden, device="cuda") * 0.12).bfloat16()
    ids = torch.randint(3, 7, (tokens, 3), device="cuda")
    if selection == "unowned":
        ids.fill_(0)
    else:
        ids[0] = torch.tensor([0, 4, 5], device="cuda")
    weights = torch.randn(tokens, 3, device="cuda").softmax(-1) * 1.7
    expected = torch.zeros(tokens, hidden, device="cuda", dtype=torch.float32)
    for index, (gate, up, down) in enumerate(dense, 4):
        rows, slots = torch.where(ids == index)
        g = (x[rows].double() @ gate.T).bfloat16().float()
        u = (x[rows].double() @ up.T).bfloat16().float()
        activated = (
            4 * torch.tanh(g / 4) * g.sigmoid() * 25 * torch.tanh(u / 25)
            if activation == "situ"
            else g * g.sigmoid() * u
        ).bfloat16()
        values = (activated.double() @ down.T).bfloat16()
        weighted = (values * weights[rows, slots, None].bfloat16()).bfloat16()
        expected.index_add_(0, rows, weighted.float())
    actual = executor.forward(x, weights, ids)
    torch.testing.assert_close(actual.float(), expected.bfloat16().float(), rtol=0.015, atol=0.015)
    if selection == "unowned":
        assert not actual.count_nonzero()
    assert executor.forward(x[:0], weights[:0], ids[:0]).shape == (0, hidden)
    for _ in range(2):
        executor.forward(x, weights, ids)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = executor.forward(x, weights, ids)
    for _ in range(3):
        x.neg_()
        ids.copy_(ids.roll(1, -1))
        weights.copy_(weights.flip(-1))
        expected_eager = executor.forward(x, weights, ids)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, expected_eager, rtol=0, atol=0)
    graph.reset()
    request.node.user_properties.append(("graph_replays", 3))
