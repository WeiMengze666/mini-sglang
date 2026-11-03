"""Queued native expanded MLA prefill and independent mathematical checks."""

import os

import pytest
import torch
import torch.nn.functional as F
from minisgl.attention.mla import CausalLayout, MLALatentOutput, MLAPagedMetadata
from minisgl.attention.mla.flashinfer import FlashInferMLAAdapter

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="CUDA work must be explicitly queued"
)


def oracle(q, latent, key, weight, start, nope, value):
    heads = q.shape[1]
    expanded = F.linear(latent, weight).reshape(len(latent), heads, nope + value)
    keys, values = expanded.split((nope, value), -1)
    keys = torch.cat((keys, key[:, None].expand(-1, heads, -1)), -1)
    score = torch.einsum("thd,shd->hts", q.double(), keys.double()) / q.shape[-1] ** 0.5
    mask = (
        torch.arange(len(latent), device=q.device)[None, :]
        > (torch.arange(len(q), device=q.device) + start)[:, None]
    )
    score.masked_fill_(mask[None], -torch.inf)
    return torch.einsum("hts,shv->thv", score.softmax(-1), values.double()).to(q.dtype)


@pytest.mark.parametrize("head_dim,value_dim", [(128, 128), (192, 128), (256, 256)])
@pytest.mark.parametrize("heads", [12, 24])
@pytest.mark.parametrize("length,tokens", [(1, 1), (17, 7), (65, 65), (65, 1), (129, 17)])
def test_native_expanded_prefill(head_dim, value_dim, heads, length, tokens):
    torch.manual_seed(401)
    nope = head_dim - 64
    start = length - tokens
    q = torch.randn(tokens, heads, head_dim, dtype=torch.bfloat16, device="cuda") * 0.1
    latent = torch.randn(length, 512, dtype=torch.bfloat16, device="cuda") * 0.1
    key = torch.randn(length, 64, dtype=torch.bfloat16, device="cuda") * 0.1
    weight = (
        torch.randn(heads * (nope + value_dim), 512, dtype=torch.bfloat16, device="cuda") * 0.05
    )
    expected = oracle(q, latent, key, weight, start, nope, value_dim)
    native = FlashInferMLAAdapter()
    actual = native(
        q,
        latent,
        key,
        weight,
        torch.arange(start, length, device=q.device),
        torch.arange(length, device=q.device),
        nope_dim=nope,
        value_dim=value_dim,
        causal_layout=CausalLayout(start),
    )
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.002)
    assert native._expanded_workspace is not None
    again = native(
        q,
        latent,
        key,
        weight,
        torch.arange(start, length, device=q.device),
        torch.arange(length, device=q.device),
        nope_dim=nope,
        value_dim=value_dim,
        causal_layout=CausalLayout(start),
    )
    torch.testing.assert_close(again, actual, rtol=0, atol=0)


def test_prefill_after_paged_decode_keeps_separate_workspace():
    torch.manual_seed(409)
    heads, length, tokens = 12, 17, 8
    q = torch.randn(tokens, heads, 192, device="cuda", dtype=torch.bfloat16) * 0.1
    latent = torch.randn(length, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    key = torch.randn(length, 64, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(heads * 256, 512, device="cuda", dtype=torch.bfloat16) * 0.05
    expected = oracle(q, latent, key, weight, 9, 128, 128)
    native = FlashInferMLAAdapter()
    pages = torch.randn(5, 4, 576, device="cuda", dtype=torch.bfloat16)
    metadata = MLAPagedMetadata(
        torch.tensor([0, 5], dtype=torch.int32),
        torch.arange(5, device="cuda", dtype=torch.int32),
        torch.tensor([17], dtype=torch.int32),
        4,
    )
    for _ in range(3):
        decoded = native.decode(
            q[:1],
            (pages[..., :512], pages[..., 512:]),
            weight,
            metadata,
            nope_dim=128,
            value_dim=128,
        )
        assert isinstance(decoded, MLALatentOutput)
        assert torch.isfinite(decoded.values).all() and torch.isfinite(decoded.lse).all()
        actual = native(
            q,
            latent,
            key,
            weight,
            torch.arange(9, 17, device="cuda"),
            torch.arange(17, device="cuda"),
            nope_dim=128,
            value_dim=128,
            causal_layout=CausalLayout(9),
        )
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.002)
    assert native._decode_plan is not None and native._expanded_workspace is not None


def test_native_prefill_uses_the_producer_stream():
    import tvm_ffi

    producer = torch.cuda.Stream()
    ffi_device = tvm_ffi.device(f"cuda:{torch.cuda.current_device()}")
    previous = tvm_ffi.get_raw_stream(ffi_device)
    with torch.cuda.stream(producer):
        torch.cuda._sleep(1_000_000)
        heads, tokens, length = 12, 13, 17
        q = torch.randn(tokens, heads, 192, device="cuda", dtype=torch.bfloat16) * 0.1
        latent = torch.randn(length, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        key = torch.randn(length, 64, device="cuda", dtype=torch.bfloat16) * 0.1
        weight = torch.randn(heads * 256, 512, device="cuda", dtype=torch.bfloat16) * 0.05
        expected = oracle(q, latent, key, weight, 4, 128, 128)
        native = FlashInferMLAAdapter()
        for _ in range(3):
            actual = native(
                q,
                latent,
                key,
                weight,
                torch.arange(4, 17, device="cuda"),
                torch.arange(17, device="cuda"),
                nope_dim=128,
                value_dim=128,
                causal_layout=CausalLayout(4),
            )
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.002)
        assert native._expanded_workspace is not None
    assert tvm_ffi.get_raw_stream(ffi_device) == previous
    torch.cuda.current_stream().wait_stream(producer)
