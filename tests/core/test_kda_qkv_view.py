"""CPU layout/lifetime checks of the indexed-decode QKV input seam."""

import pytest
import torch
from minisgl.attention.kda import sgl


@pytest.mark.parametrize("batch", [1, 48])
def test_adjacent_qkvg_prefix_is_readonly_row_gapped_view(monkeypatch, batch):
    width = 1536  # Canonical rank-local H12 * D128.
    merged = torch.arange(batch * 4 * width).remainder(31).bfloat16().reshape(batch, 4 * width)
    q, k, v, gate = merged.split(width, dim=-1)
    expected, snapshot = torch.cat((q, k, v), -1), merged.clone()
    monkeypatch.setattr(sgl.torch, "cat", lambda *args, **kwargs: pytest.fail("QKV copied"))
    result = sgl._conv_qkv_input(q, k, v)
    assert result.shape == (batch, 3 * width) and result.stride() == (4 * width, 1)
    assert torch._C._is_alias_of(result, merged)
    assert result.storage_offset() == q.storage_offset()
    assert torch.equal(result, expected) and torch.equal(merged, snapshot)
    assert torch.equal(gate, snapshot[:, 3 * width :])
    del q, k, v, gate, merged
    assert torch.equal(result, expected)  # The view retains its storage owner.


@pytest.mark.parametrize(
    "case", ["independent", "gap", "row_stride", "channel_stride", "grad", "dtype", "logical_view"]
)
def test_noneligible_layout_keeps_original_cat(case):
    batch, width = 3, 4
    merged = torch.arange(batch * 4 * width).float().reshape(batch, 4 * width)
    parts = merged.split(width, -1)[:3]
    if case == "independent":
        parts = tuple(part.clone() for part in parts)
    elif case == "gap":
        merged = torch.arange(batch * 5 * width).float().reshape(batch, 5 * width)
        parts = (
            merged[:, :width],
            merged[:, 2 * width : 3 * width],
            merged[:, 3 * width : 4 * width],
        )
    elif case == "row_stride":
        parts = parts[0], parts[1][:1].expand(batch, -1), parts[2]
    elif case == "channel_stride":
        merged = torch.arange(batch * 8 * width).float().reshape(batch, 8 * width)
        parts = tuple(
            merged[:, start : start + 2 * width : 2] for start in (0, 2 * width, 4 * width)
        )
    elif case == "grad":
        merged.requires_grad_()
        parts = merged.split(width, -1)[:3]
    elif case == "logical_view":
        parts = torch._neg_view(parts[0]), parts[1], parts[2]
    else:
        parts = parts[0].double(), parts[1], parts[2]
    expected = torch.cat(parts, -1)
    result = sgl._conv_qkv_input(*parts)
    assert torch.equal(result, expected)
    assert all(not torch._C._is_alias_of(result, part) for part in parts)
    assert result.requires_grad == expected.requires_grad


def test_selected_view_failure_propagates_without_cat_retry(monkeypatch):
    q, k, v, _ = torch.zeros(2, 16).split(4, -1)
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("selected view failed")

    monkeypatch.setattr(torch.Tensor, "as_strided", fail)
    monkeypatch.setattr(
        sgl.torch, "cat", lambda *args, **kwargs: pytest.fail("view retried as cat")
    )
    with pytest.raises(RuntimeError, match="selected view failed"):
        sgl._conv_qkv_input(q, k, v)
    assert calls == [1]
