"""Backend selection, precision and collective contracts without optional SDKs."""

import builtins
from types import SimpleNamespace

import pytest
import torch
from minisgl.moe import TorchExpertExecutor, create_moe_backend, get_moe_backend


@pytest.mark.parametrize("name", ["fused", "marlin"])
def test_backend_selection_and_cpu_fallback_do_not_load_optional_kernels(monkeypatch, name):
    import minisgl.core as core

    original = builtins.__import__

    def without_sdk(module, *args, **kwargs):
        if module.split(".")[0] in ("sglang", "sgl_kernel", "tvm_ffi", "flashinfer"):
            raise AssertionError(f"CPU backend selection imported {module}")
        return original(module, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sdk)
    backend = create_moe_backend(name)
    monkeypatch.setattr(core, "_GLOBAL_CTX", SimpleNamespace(moe_backend=backend))
    assert get_moe_backend() is backend
    assert (
        backend.route(
            torch.zeros(1, 8),
            bias=None,
            topk=1,
            num_groups=1,
            topk_groups=1,
            scoring_func="softmax",
            renormalize=True,
            scale=1.0,
        )
        is None
    )
    executor = backend.prepare_experts([], activation="silu")
    assert isinstance(executor, TorchExpertExecutor)
    monkeypatch.setattr(core, "_GLOBAL_CTX", None)
    assert type(get_moe_backend()) is type(create_moe_backend("fused"))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_backend_reduces_independent_output_shapes_through_one_collective(dtype):
    backend = create_moe_backend("fused")
    parts = (
        torch.arange(12, dtype=dtype).reshape(3, 4).T,
        torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        torch.tensor(2.0, dtype=dtype),
    )
    seen = []

    def all_reduce(value):
        seen.append(value)
        return value * 3

    actual = backend.reduce_outputs(SimpleNamespace(all_reduce=all_reduce), *parts)
    assert len(seen) == 1 and seen[0].is_contiguous() and seen[0].dtype == dtype
    assert seen[0].numel() == 19
    for value, original in zip(actual, parts):
        assert value.shape == original.shape and value.dtype == dtype
        torch.testing.assert_close(value, original.to(dtype) * 3, rtol=0, atol=0)


def test_single_output_keeps_layout_and_uses_selected_transport():
    value = torch.arange(12.0).reshape(3, 4).T
    seen = []

    def all_reduce(partial):
        assert partial is value
        seen.append(partial)
        return partial + 1

    (result,) = create_moe_backend("fused").reduce_outputs(
        SimpleNamespace(all_reduce=all_reduce), value
    )
    assert len(seen) == 1 and result.shape == value.shape
    torch.testing.assert_close(result, value + 1, rtol=0, atol=0)
    with pytest.raises(ValueError, match="at least one"):
        create_moe_backend("fused").reduce_outputs(SimpleNamespace(all_reduce=all_reduce))


def test_marlin_precision_contract_is_explicit():
    backend = create_moe_backend("marlin")
    backend.validate_precision("bf16")
    with pytest.raises(ValueError, match="bf16"):
        backend.validate_precision("mxfp8")


@pytest.mark.parametrize("hidden", [64, 96, 128, 3584])
def test_marlin_hidden_padding_preserves_checkpoint_codes_and_zero_features(hidden):
    from minisgl.moe.marlin import MarlinMXFP4Experts

    experts, intermediate = 2, 96
    w13 = torch.arange(experts * 2 * intermediate * (hidden // 2)).remainder(251).byte()
    w13 = w13.reshape(experts, 2 * intermediate, hidden // 2)
    w2 = torch.arange(experts * hidden * (intermediate // 2)).remainder(251).byte()
    w2 = w2.reshape(experts, hidden, intermediate // 2)
    s13 = torch.full((experts, 2 * intermediate, hidden // 32), 123, dtype=torch.uint8)
    s2 = torch.full((experts, hidden, intermediate // 32), 126, dtype=torch.uint8)
    original = (w13, w2, s13, s2)
    padded = MarlinMXFP4Experts._pad_hidden(*original)
    if hidden % 128 == 0:
        assert all(a is b for a, b in zip(padded, original))
    a, b, c, d = padded
    assert a.shape[-1] * 2 == b.shape[1] == ((hidden + 127) // 128) * 128
    torch.testing.assert_close(a[..., : hidden // 2], w13, rtol=0, atol=0)
    torch.testing.assert_close(b[:, :hidden], w2, rtol=0, atol=0)
    torch.testing.assert_close(c[..., : hidden // 32], s13, rtol=0, atol=0)
    torch.testing.assert_close(d[:, :hidden], s2, rtol=0, atol=0)
    assert not a[..., hidden // 2 :].count_nonzero() and not b[:, hidden:].count_nonzero()
    assert (c[..., hidden // 32 :] == 127).all() and (d[:, hidden:] == 127).all()


@pytest.mark.parametrize(
    "quantization,dtype,precision,expected",
    [
        ({"format": "mxfp4-pack-quantized"}, torch.bfloat16, "bf16", "marlin"),
        ({"format": "mxfp4-pack-quantized"}, torch.bfloat16, "mxfp8", "fused"),
        ({"format": "mxfp4-pack-quantized"}, torch.float16, "bf16", "fused"),
        ({"format": "mxfp4-pack-quantized"}, torch.float32, "bf16", "fused"),
        (None, torch.bfloat16, "bf16", "fused"),
        ({"format": "another-format"}, torch.bfloat16, "bf16", "fused"),
    ],
)
def test_auto_expert_backend_follows_weight_and_activation_contract(
    quantization, dtype, precision, expected
):
    from minisgl.moe import select_moe_backend

    assert select_moe_backend(quantization, dtype=dtype, activation_precision=precision) == expected
