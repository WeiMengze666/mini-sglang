"""Queued grouped-expert checks against independent, exactly generated weights."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.layers.activation import situ_and_mul
from minisgl.quantization.mxfp4 import MXFP4Linear

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate"
)


def _projection(inputs, outputs, dtype):
    codes = torch.randint(16, (outputs, inputs), device="cuda", dtype=torch.uint8)
    scales = torch.randint(122, 127, (outputs, inputs // 32), device="cuda", dtype=torch.uint8)
    codebook = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    dense = (codebook[codes.long()] * (2.0 ** (scales.float() - 127)).repeat_interleave(32, -1)).to(
        dtype
    )
    op = MXFP4Linear(inputs, outputs, device="cuda", dtype=dtype)
    op.weight_packed.copy_(codes[:, ::2] | codes[:, 1::2] << 4)
    op.weight_scale.copy_(scales)
    return op, dense


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("activation", ["silu", "situ"])
@pytest.mark.parametrize(
    "tokens,selection", [(1, "remote"), (1, "mixed"), (7, "mixed"), (65, "owned")]
)
def test_grouped_packed_experts(dtype, activation, tokens, selection):
    from minisgl.quantization.mxfp4_moe import MXFP4Experts

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(103)
    hidden, intermediate = 96, 96  # exercise the last 64-column K tile
    items, dense = [], []
    for index in range(4, 8):
        w1, gate = _projection(hidden, intermediate, dtype)
        w3, up = _projection(hidden, intermediate, dtype)
        w2, down = _projection(intermediate, hidden, dtype)
        items.append((index, SimpleNamespace(w1=w1, w3=w3, w2=w2)))
        dense.append((gate, up, down))
    executor = MXFP4Experts(items, activation=activation, beta=4.0, linear_beta=25.0)
    x = torch.randn(tokens, hidden, device="cuda", dtype=dtype) * 0.12
    ids = torch.randint(4 if selection == "owned" else 0, 8, (tokens, 3), device="cuda")
    if selection == "remote":
        ids.fill_(0)
    elif selection == "mixed":
        ids[0] = torch.tensor([0, 4, 7], device="cuda")
    weights = torch.randn(tokens, 3, device="cuda").softmax(-1) * 1.7
    expected = torch.zeros(tokens, hidden, device="cuda", dtype=torch.float32)
    for index, (gate, up, down) in enumerate(dense, 4):
        rows, slots = torch.where(ids == index)
        g = F.linear(x[rows].float(), gate.float()).to(dtype)
        u = F.linear(x[rows].float(), up.float()).to(dtype)
        activated = situ_and_mul(g, u, 4.0, 25.0) if activation == "situ" else F.silu(g) * u
        values = (
            F.linear(activated.float(), down.float()).to(dtype).float() * weights[rows, slots, None]
        )
        expected.index_add_(0, rows, values)
    actual = executor.forward(x, weights, ids)
    tolerance = 3e-5 if dtype == torch.float32 else 0.015
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    if selection == "remote":
        assert actual.eq(0).all()


@pytest.mark.parametrize("tokens", [2, 8])
def test_large_experts_preserve_duplicate_rows(tokens):
    from minisgl.quantization.mxfp4_moe import MXFP4Experts

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(113)
    dtype = torch.bfloat16
    hidden, intermediate = 3584, 3072
    items, dense = [], []
    for index in (4, 5):
        w1, gate = _projection(hidden, intermediate, dtype)
        w3, up = _projection(hidden, intermediate, dtype)
        w2, down = _projection(intermediate, hidden, dtype)
        items.append((index, SimpleNamespace(w1=w1, w3=w3, w2=w2)))
        dense.append((gate, up, down))
    x = (torch.randn(1, hidden, device="cuda", dtype=dtype) * 0.02).expand(tokens, -1)
    ids = torch.tensor([[0, 4, 5]], device="cuda").expand(tokens, -1)
    weights = torch.tensor([[0.2, 0.3, 0.5]], device="cuda").expand(tokens, -1)
    expected = torch.zeros(tokens, hidden, device="cuda", dtype=torch.float32)
    for slot, (gate, up, down) in enumerate(dense, 1):
        g = F.linear(x.float(), gate.float()).to(dtype)
        u = F.linear(x.float(), up.float()).to(dtype)
        a = situ_and_mul(g, u, 4.0, 25.0)
        expected += F.linear(a.float(), down.float()).to(dtype).float() * weights[:, slot, None]
    actual = MXFP4Experts(items, activation="situ", beta=4.0, linear_beta=25.0).forward(
        x, weights, ids
    )
    torch.testing.assert_close(actual, expected, rtol=0.015, atol=0.015)
    torch.testing.assert_close(actual, actual[:1].expand_as(actual), rtol=0, atol=0)
