import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.layers.linear import Linear
from minisgl.layers.weight import decode_mxfp4

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate"
)


def test_flashinfer_executor_imports_without_cuda_side_effect():
    from minisgl.moe.flashinfer import FlashInferMXFP4Experts

    assert FlashInferMXFP4Experts is not None


def test_flashinfer_executor_rejects_situ_bf16_policy():
    from minisgl.moe.flashinfer import FlashInferMXFP4Experts

    with pytest.raises(ValueError, match="SiTU.*mxfp8"):
        FlashInferMXFP4Experts(
            [],
            activation="situ",
            beta=4.0,
            linear_beta=25.0,
            num_global_experts=0,
            activation_precision="bf16",
        )


def test_flashinfer_executor_rejects_non_cuda_weights():
    from minisgl.moe.flashinfer import FlashInferMXFP4Experts

    with pytest.raises(ValueError, match="at least one local expert"):
        FlashInferMXFP4Experts(
            [],
            activation="silu",
            beta=1.0,
            linear_beta=None,
            num_global_experts=0,
            activation_precision="bf16",
        )


def _expert(hidden=256, intermediate=256, dtype=torch.bfloat16):
    ops = []
    for inputs, outputs in ((hidden, intermediate), (hidden, intermediate), (intermediate, hidden)):
        op = Linear(inputs, outputs, device="cuda", dtype=dtype, weight_format="mxfp4")
        op.weight_packed.copy_(
            torch.randint(0, 256, op.weight_packed.shape, device="cuda", dtype=torch.uint8)
        )
        op.weight_scale.copy_(
            torch.randint(122, 127, op.weight_scale.shape, device="cuda", dtype=torch.uint8)
        )
        ops.append(op)
    return SimpleNamespace(w1=ops[0], w3=ops[1], w2=ops[2])


def test_flashinfer_silu_bf16_matches_decoded_reference():
    from minisgl.moe.flashinfer import FlashInferMXFP4Experts

    torch.manual_seed(7)
    expert = _expert()
    raw = tuple(
        (op.weight_packed.clone(), op.weight_scale.clone())
        for op in (expert.w1, expert.w3, expert.w2)
    )
    executor = FlashInferMXFP4Experts(
        [(4, expert)],
        activation="silu",
        beta=1.0,
        linear_beta=None,
        num_global_experts=8,
        activation_precision="bf16",
    )
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16) * 0.1
    ids = torch.tensor([[4], [0], [4]], device="cuda", dtype=torch.int32)
    weights = torch.ones_like(ids, dtype=torch.float32)
    actual = executor.forward(x, weights, ids)
    gate = F.linear(x.float(), decode_mxfp4(raw[0][0], raw[0][1]).float()).to(x.dtype)
    up = F.linear(x.float(), decode_mxfp4(raw[1][0], raw[1][1]).float()).to(x.dtype)
    hidden = (F.silu(gate) * up).float()
    expected = F.linear(hidden, decode_mxfp4(raw[2][0], raw[2][1]).float()).to(x.dtype).float()
    torch.testing.assert_close(actual[0], expected[0], rtol=0.02, atol=0.2)
    torch.testing.assert_close(actual[2], expected[2], rtol=0.02, atol=0.2)
    assert actual[1].eq(0).all()


def test_flashinfer_situ_mxfp8_is_finite_and_repeatable():
    from minisgl.moe.flashinfer import FlashInferMXFP4Experts

    torch.manual_seed(11)
    expert = _expert()
    executor = FlashInferMXFP4Experts(
        [(4, expert)],
        activation="situ",
        beta=4.0,
        linear_beta=25.0,
        num_global_experts=8,
        activation_precision="mxfp8",
    )
    row = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16) * 0.05
    x = row.expand(4, -1)
    ids = torch.full((4, 1), 4, device="cuda", dtype=torch.int32)
    weights = torch.ones((4, 1), device="cuda", dtype=torch.float32)
    actual = executor.forward(x, weights, ids)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, actual[:1].expand_as(actual), rtol=0, atol=0)
