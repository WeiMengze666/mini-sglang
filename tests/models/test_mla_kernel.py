"""Queued absorbed MLA/LSE checks, including padded TP head counts."""

import os

import pytest
import torch
from minisgl.attention.mla import CausalLayout, MLAPagedMetadata, mla_attention
from minisgl.attention.mla_fi import FlashInferMLA

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="CUDA work must be explicitly queued"
)


@pytest.mark.parametrize(
    "heads,length,tokens,start,offset",
    [
        (12, 1, 1, 1, 0),
        (12, 17, 1, 33, 0),
        (24, 127, 1, 253, 0),
        (24, 0, 1, 0, 0),
        (24, 10, 7, 13, 0),
        (24, 4, 7, 0, 0),
        (24, 3, 7, 0, 1),
        (12, 17, 1, 0, 1),
    ],
)
@pytest.mark.parametrize("layout_known", [False, True])
def test_absorbed_decode_values_and_natural_lse(heads, length, tokens, start, offset, layout_known):
    torch.manual_seed(107)
    q = torch.randn(tokens, heads, 256, dtype=torch.bfloat16, device="cuda") * 0.1
    latent = torch.randn(length, 512, dtype=torch.bfloat16, device="cuda") * 0.1
    key = torch.randn(length, 64, dtype=torch.bfloat16, device="cuda") * 0.1
    weight = torch.randn(heads * 320, 512, dtype=torch.bfloat16, device="cuda") * 0.05
    positions = torch.arange(length, device="cuda") * 2 + offset
    query_pos = torch.arange(start, start + tokens, device="cuda")
    expected, expected_lse = mla_attention(
        q, latent, key, weight, query_pos, positions, nope_dim=192, value_dim=128, return_lse=True
    )
    native = FlashInferMLA()
    for _ in range(2):
        actual, lse = native(
            q,
            latent,
            key,
            weight,
            query_pos,
            positions,
            nope_dim=192,
            value_dim=128,
            return_lse=True,
            causal_layout=CausalLayout(start, offset, 2) if layout_known else None,
        )
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.002)
        torch.testing.assert_close(lse, expected_lse, rtol=0.001, atol=0.001)
    assert (native._prefill_plan is not None) == bool(
        (positions[None, :] <= query_pos[:, None]).any()
    )


def test_prefill_plan_after_paged_decode_keeps_its_own_query_groups():
    torch.manual_seed(307)
    heads = 2
    q = torch.randn(8, heads, 256, device="cuda", dtype=torch.bfloat16) * 0.1
    latent = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    key = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16) * 0.1
    weight = torch.randn(heads * 320, 512, device="cuda", dtype=torch.bfloat16) * 0.05
    positions = torch.arange(8, device="cuda")
    history = torch.arange(1, 8, 2, device="cuda")
    expected, expected_lse = mla_attention(
        q, latent, key, weight, positions, history, nope_dim=192, value_dim=128, return_lse=True
    )
    native = FlashInferMLA()
    pages = torch.randn(5, 4, 576, device="cuda", dtype=torch.bfloat16)
    metadata = MLAPagedMetadata(
        torch.tensor([0, 5], dtype=torch.int32),
        torch.arange(5, device="cuda", dtype=torch.int32),
        torch.tensor([17], dtype=torch.int32),
        4,
    )
    for _ in range(3):
        actual, lse = native(
            q,
            latent,
            key,
            weight,
            positions,
            history,
            nope_dim=192,
            value_dim=128,
            return_lse=True,
            causal_layout=CausalLayout(0, 1, 2),
        )
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.002)
        torch.testing.assert_close(lse, expected_lse, rtol=0.001, atol=0.001)
        decoded = native.decode(
            q[:1],
            (pages[..., :512], pages[..., 512:]),
            weight,
            metadata,
            nope_dim=192,
            value_dim=128,
        )
        assert torch.isfinite(decoded).all()


def test_native_mla_uses_the_producer_stream():
    import tvm_ffi

    producer = torch.cuda.Stream()
    ffi_device = tvm_ffi.device("cuda:0")
    previous_stream = tvm_ffi.get_raw_stream(ffi_device)
    with torch.cuda.stream(producer):
        # Inputs and planning metadata are produced after work on this stream.
        torch.cuda._sleep(1_000_000)
        heads, tokens = 2, 13
        q = torch.randn(tokens, heads, 256, device="cuda", dtype=torch.bfloat16) * 0.1
        latent = torch.randn(6, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        key = torch.randn(6, 64, device="cuda", dtype=torch.bfloat16) * 0.1
        weight = torch.randn(heads * 320, 512, device="cuda", dtype=torch.bfloat16) * 0.05
        positions = torch.arange(tokens, device="cuda")
        history = torch.arange(1, tokens, 2, device="cuda")
        expected, expected_lse = mla_attention(
            q,
            latent,
            key,
            weight,
            positions,
            history,
            nope_dim=192,
            value_dim=128,
            return_lse=True,
        )
        native = FlashInferMLA()
        for _ in range(3):
            actual, lse = native(
                q,
                latent,
                key,
                weight,
                positions,
                history,
                nope_dim=192,
                value_dim=128,
                return_lse=True,
                causal_layout=CausalLayout(0, 1, 2),
            )
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.002)
            torch.testing.assert_close(lse, expected_lse, rtol=0.001, atol=0.001)
    assert tvm_ffi.get_raw_stream(ffi_device) == previous_stream
