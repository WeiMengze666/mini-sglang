"""Real CPU destination ABI checks; these do not qualify CUDA arithmetic."""

import pytest
import torch
import torch.nn.functional as F
from minisgl.layers.linear import Linear, MergedLinear


@pytest.fixture(autouse=True)
def forbid_cuda(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("CPU destination test initialized CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", fail)


def projection(dtype=torch.bfloat16, *, bias=False, input_width=16, output_width=32):
    op = Linear(input_width, output_width, has_bias=bias, dtype=dtype)
    op.weight.copy_(torch.linspace(-0.25, 0.25, op.weight.numel()).view_as(op.weight))
    if op.bias is not None:
        op.bias.copy_(torch.linspace(-0.125, 0.125, output_width))
    x = torch.linspace(-0.5, 0.5, 48 * input_width, dtype=dtype).view(48, input_width)
    return op, x


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("bias", [False, True])
def test_direct_output_is_the_reserved_view_without_resizing(dtype, bias):
    op, x = projection(dtype, bias=bias)
    expected = F.linear(x, op.weight, op.bias)
    storage = torch.full((52, 32), -17, dtype=dtype)
    output = storage[2:50]
    pointer = output.data_ptr()
    saved_x, saved_weight = x.clone(), op.weight.clone()
    saved_bias = op.bias.clone() if bias else None
    assert op.supports_out(dtype)
    result = op.forward(x, out=output)
    assert result is output and result.data_ptr() == pointer
    assert output.shape == (48, 32) and output.stride() == (32, 1)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert storage[:2].eq(-17).all() and storage[50:].eq(-17).all()
    assert torch.equal(x, saved_x) and torch.equal(op.weight, saved_weight)
    if bias:
        assert torch.equal(op.bias, saved_bias)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "invalid",
    ["shape", "dtype", "device", "stride", "input_width", "input_rank", "overlap", "input_grad"],
)
def test_invalid_destination_fails_before_the_kernel_or_resize(monkeypatch, invalid):
    op, x = projection(output_width=16)
    output = torch.full((48, 16), -17, dtype=torch.bfloat16)
    if invalid == "shape":
        output = torch.full((47, 16), -17, dtype=torch.bfloat16)
    elif invalid == "dtype":
        output = output.float()
    elif invalid == "device":
        output = torch.empty(48, 16, dtype=torch.bfloat16, device="meta")
    elif invalid == "stride":
        output = torch.full((48, 32), -17, dtype=torch.bfloat16)[:, ::2]
    elif invalid == "input_width":
        x = x[:, :-1]
    elif invalid == "input_rank":
        x = x.unsqueeze(0)
    elif invalid == "overlap":
        output = x
    elif invalid == "input_grad":
        x.requires_grad_()
    original_shape = output.shape
    original_stride = output.stride()

    def fail(*args, **kwargs):
        pytest.fail("invalid destination reached GEMM")

    monkeypatch.setattr(F, "linear", fail)
    with pytest.raises(ValueError):
        op.forward(x, out=output)
    assert output.shape == original_shape and output.stride() == original_stride


def test_quantized_output_is_declined_without_allocating_or_decoding(monkeypatch):
    op = Linear(32, 16, weight_format="mxfp4")
    x = torch.ones(4, 32, dtype=torch.bfloat16)
    output = torch.empty(4, 16, dtype=torch.bfloat16)
    assert not op.supports_out(torch.bfloat16)
    marker = object()
    monkeypatch.setattr(op, "_forward_mxfp4", lambda candidate: marker)
    with pytest.raises(NotImplementedError, match="dense"):
        op.forward(x, out=output)
    assert op.forward(x) is marker  # Existing no-out quantized route is unchanged.


@pytest.mark.parametrize(
    "unsupported", ["dtype", "unloaded", "weight_shape", "bias_shape", "weight_grad", "bias_grad"]
)
def test_output_capability_only_advertises_supported_weights(unsupported):
    op, x = projection(bias=True)
    dtype = torch.float32 if unsupported == "dtype" else x.dtype
    if unsupported == "unloaded":
        op.weight = torch.empty_like(op.weight, device="meta")
    elif unsupported == "weight_shape":
        op.weight = op.weight[:-1]
    elif unsupported == "bias_shape":
        op.bias = op.bias[:-1]
    elif unsupported == "weight_grad":
        op.weight.requires_grad_()
    elif unsupported == "bias_grad":
        op.bias.requires_grad_()
    assert not op.supports_out(dtype)


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_existing_merged_projection_precision_and_shapes_are_unchanged(output_dtype):
    first, x = projection(output_width=8)
    second, _ = projection(output_width=16)
    merged = MergedLinear.try_create({"first": first, "second": second})
    assert merged is not None
    results = merged.forward(x, output_dtype=output_dtype)
    for name, op in (("first", first), ("second", second)):
        expected = (
            F.linear(x, op.weight)
            if output_dtype == x.dtype
            else F.linear(x.float(), op.weight.float())
        )
        assert results[name].dtype == output_dtype
        torch.testing.assert_close(results[name], expected, rtol=0, atol=0)


def test_allocating_dense_route_uses_the_unchanged_dispatcher():
    op, x = projection(bias=True)
    expected = F.linear(x, op.weight, op.bias)
    actual = op.forward(x)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_missing_runtime_out_abi_declines_without_changing_normal_linear(monkeypatch):
    from minisgl.layers import linear

    op, x = projection(bias=True)
    expected = F.linear(x, op.weight, op.bias)
    output = torch.full((48, 32), -17, dtype=torch.bfloat16)
    monkeypatch.setattr(linear, "_LINEAR_OUT", None)
    assert not op.supports_out(x.dtype)
    with pytest.raises(NotImplementedError, match="dense inference weights"):
        op.forward(x, out=output)
    assert output.eq(-17).all()
    torch.testing.assert_close(op.forward(x), expected, rtol=0, atol=0)
