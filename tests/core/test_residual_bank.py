"""Host ownership and fallback arithmetic of a forward-local residual bank."""

from contextlib import nullcontext
from types import SimpleNamespace

import minisgl.layers.attn_residual as residual
import pytest
import torch
from minisgl.layers.norm import TorchRMSNorm


def parameters(hidden, dtype):
    norm = TorchRMSNorm(hidden, 1e-5)
    target = TorchRMSNorm(hidden, 1e-5)
    norm.weight = torch.randn(hidden, dtype=dtype)
    target.weight = torch.randn(hidden, dtype=dtype)
    proj = SimpleNamespace(weight=torch.randn(1, hidden, dtype=dtype))
    return norm, proj, target


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("normalized", [False, True])
@pytest.mark.parametrize("tokens,hidden", [(1, 4), (7, 16)])
def test_snapshots_are_owned_and_cpu_reductions_are_unchanged(dtype, normalized, tokens, hidden):
    torch.manual_seed(2026)
    head = torch.randn(tokens, hidden * 2, dtype=dtype)[:, ::2]
    bank = residual.ResidualBank(head, 5)
    bank._storage.fill_(float("nan"))
    norm, proj, target = parameters(hidden, dtype)
    target = target if normalized else None
    expected = []
    for _ in range(3):
        raw = torch.randn_like(head)
        expected.append(raw.clone())
        bank.append(raw)
        raw.zero_()
        for got, want in zip(bank, expected):
            torch.testing.assert_close(got, want, rtol=0, atol=0)
        actual = residual.attention_residual(head, bank, norm, proj, target)
        legacy = residual.attention_residual(head, expected, norm, proj, target)
        torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
        assert torch.isfinite(actual).all()
    assert bank._storage[:, 3:].isnan().all()


def test_empty_bank_and_failed_append_preserve_ownership():
    head = torch.randn(3, 8)
    bank = residual.ResidualBank(head, 1)
    norm, proj, target = parameters(8, torch.float32)
    assert residual.attention_residual(head, bank, norm, proj) is head
    torch.testing.assert_close(
        residual.attention_residual(head, bank, norm, proj, target),
        target.forward(head),
        rtol=0,
        atol=0,
    )
    with pytest.raises(ValueError):
        bank.append(torch.randn(3, 8, dtype=torch.float64))
    assert len(bank) == 0
    bank.append(head)
    expected = list(bank)[0].clone()
    with pytest.raises(ValueError):
        bank.append(head)
    assert len(bank) == 1
    torch.testing.assert_close(list(bank)[0], expected, rtol=0, atol=0)
    with pytest.raises(ValueError):
        residual.ResidualBank(head, 0)


def test_native_metadata_passes_valid_count_separately_from_capacity(monkeypatch):
    seen = []

    def kernel(head, rows, count, *weights):
        seen.append((rows, count))
        return head

    monkeypatch.setattr(residual, "_native_kernels", lambda: (kernel, kernel, nullcontext))
    head = torch.randn(2, 8)
    bank = residual.ResidualBank(head, 8)
    bank.append(head)
    bank.append(head + 1)
    norm, proj, target = parameters(8, torch.float32)
    actual = residual._read_native(head, bank._storage, norm, proj, target, valid_blocks=len(bank))
    assert actual is head and seen[-1][0] is bank._storage and seen[-1][1] == 2
