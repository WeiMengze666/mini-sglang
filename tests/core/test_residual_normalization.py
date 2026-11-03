import os
from types import SimpleNamespace

import pytest
import torch
from minisgl.layers.attn_residual import attention_residual
from minisgl.layers.norm import TorchRMSNorm


def _oracle(head, bank, norm, proj, output_norm):
    rows = torch.stack([*bank, head], dim=1).double()
    query = (norm.weight.float() * proj.weight.float()).to(head.dtype).double()
    scores = (rows * query).sum(-1) * torch.rsqrt(rows.square().mean(-1) + norm.eps)
    mixed = (scores.softmax(1).unsqueeze(-1) * rows).sum(1)
    normalized = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + output_norm.eps)
    return (normalized * output_norm.weight.double()).to(head.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tokens,count", [(0, 1), (3, 1), (3, 4)])
def test_normalized_residual_matches_high_precision_bank(dtype, tokens, count):
    torch.manual_seed(73)
    hidden = 17
    head = torch.randn(tokens, hidden).to(dtype)
    bank = [torch.randn_like(head) for _ in range(count)]
    norm = SimpleNamespace(weight=torch.randn(hidden).to(dtype), eps=1e-5)
    proj = SimpleNamespace(weight=(torch.randn(1, hidden) * 0.1).to(dtype))
    output_norm = TorchRMSNorm(hidden, 1e-3)
    output_norm.weight = torch.randn(hidden).to(dtype)
    original = [x.clone() for x in [head, *bank]]
    actual = attention_residual(head, bank, norm, proj, output_norm)
    expected = _oracle(head, bank, norm, proj, output_norm)
    torch.testing.assert_close(
        actual, expected, rtol=0.008 if dtype == torch.bfloat16 else 1e-5, atol=1e-6
    )
    for actual_input, saved in zip([head, *bank], original):
        torch.testing.assert_close(actual_input, saved, rtol=0, atol=0)


def _rounding_case(device):
    dtype = torch.bfloat16
    head = torch.tensor(
        [[0.0272216796875, -0.30859375, 0.83984375, -1.2890625]], dtype=dtype, device=device
    )
    bank = [
        torch.tensor(
            [[-0.007293701171875, -0.6953125, 0.373046875, 0.482421875]], dtype=dtype, device=device
        )
    ]
    norm = SimpleNamespace(weight=torch.ones(4, dtype=dtype, device=device), eps=1e-5)
    proj = SimpleNamespace(weight=torch.zeros(1, 4, dtype=dtype, device=device))
    output_norm = TorchRMSNorm(4, 1e-5)
    output_norm.weight = torch.tensor(
        [2.265625, -1.8359375, 0.0035247802734375, -0.35546875], dtype=dtype, device=device
    )
    return head, bank, norm, proj, output_norm


def test_normalized_residual_retains_mixture_before_final_norm():
    head, bank, norm, proj, output_norm = _rounding_case("cpu")
    expected = _oracle(head, bank, norm, proj, output_norm)
    early = output_norm.forward(attention_residual(head, bank, norm, proj))
    assert not torch.equal(early, expected)
    actual = attention_residual(head, bank, norm, proj, output_norm)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        attention_residual(head, [], norm, proj, output_norm),
        output_norm.forward(head),
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate")
def test_cuda_normalized_residual_rounding_and_chunk_invariance():
    args = _rounding_case("cuda")
    expected = _oracle(*_rounding_case("cpu"))
    actual = attention_residual(*args)
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    torch.manual_seed(74)
    for tokens, hidden, count in [
        (0, 129, 2),
        (1, 129, 4),
        (27, 129, 4),
        (27, 7168, 1),
        (3, 31, 0),
        (3, 1025, 2),
        (3, 1536, 8),
        (2, 257, 16),
    ]:
        head = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
        bank = [torch.randn_like(head) for _ in range(count)]
        norm = SimpleNamespace(weight=torch.ones(hidden, device="cuda", dtype=head.dtype), eps=1e-5)
        proj = SimpleNamespace(
            weight=torch.randn(1, hidden, device="cuda", dtype=head.dtype) * 0.01
        )
        output_norm = TorchRMSNorm(hidden, 1e-3)
        output_norm.weight = torch.randn(hidden, device="cuda", dtype=head.dtype)
        actual = attention_residual(head, bank, norm, proj, output_norm)
        expected = _oracle(head, bank, norm, proj, output_norm)
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-5)
        if tokens:
            chunks = [
                attention_residual(
                    head[i : i + 7], [x[i : i + 7] for x in bank], norm, proj, output_norm
                )
                for i in range(0, tokens, 7)
            ]
            torch.testing.assert_close(actual, torch.cat(chunks), rtol=0, atol=0)


@pytest.mark.skipif(os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued CUDA gate")
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("weight_stride", [1, 2])
def test_cuda_normalized_residual_output_weight_fallback(weight_dtype, weight_stride):
    torch.manual_seed(75)
    head = torch.randn(3, 17, device="cuda", dtype=torch.bfloat16)
    bank = [torch.randn_like(head)]
    norm = SimpleNamespace(weight=torch.ones(17, device="cuda", dtype=head.dtype), eps=1e-5)
    proj = SimpleNamespace(weight=torch.randn(1, 17, device="cuda", dtype=head.dtype) * 0.01)
    output_norm = TorchRMSNorm(17, 1e-3)
    output_norm.weight = torch.randn(17 * weight_stride, device="cuda", dtype=weight_dtype)[
        ::weight_stride
    ]
    actual = attention_residual(head, bank, norm, proj, output_norm)
    expected = _oracle(head, bank, norm, proj, output_norm)
    torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-5)
