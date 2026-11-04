"""Host ABI and error semantics for optional native attention combination."""

import pytest
import torch
from minisgl.kernel import dcp


class DeviceInput(torch.Tensor):
    @property
    def is_cuda(self):
        return True


@pytest.mark.parametrize("dimension", [3, 1025])
def test_unsupported_dimensions_do_not_load_native_kernel(monkeypatch, dimension):
    monkeypatch.setattr(dcp, "_native_lse_combine", lambda: pytest.fail("outside native coverage"))
    x = torch.empty(2, 1, 1, dimension).as_subclass(DeviceInput)
    assert dcp.combine_partial_attention(x, torch.zeros(2, 1, 1)) is None


def test_native_adapter_preserves_empty_and_corrupt_rows_without_modifying_producers(monkeypatch):
    def no_cuda(*a, **kw):
        pytest.fail("Host ABI fixture must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", no_cuda)
    x = torch.ones(2, 1, 3, 8).as_subclass(DeviceInput)
    x[0, 0, 0] = torch.nan
    lse = torch.zeros(2, 1, 3)
    lse[0, 0] = torch.tensor([-torch.inf, torch.nan, torch.inf])
    calls = []

    def native(values, actual_lse, *, is_lse_base_on_e):
        assert values.dtype == torch.float32 and actual_lse is lse and is_lse_base_on_e
        assert values[0, 0, 0].eq(0).all()
        calls.append(1)
        return torch.ones(1, 3, 8), None

    monkeypatch.setattr(dcp, "_native_lse_combine", lambda: native)
    result = dcp.combine_partial_attention(x, lse)
    assert calls == [1]
    assert result[0, 0].eq(1).all() and result[0, 1:].isnan().all()
    assert x[0, 0, 0].isnan().all()


def test_unavailable_native_kernel_defers_to_portable_combine(monkeypatch):
    monkeypatch.setattr(dcp, "_native_lse_combine", lambda: None)
    x = torch.empty(2, 1, 1, 8).as_subclass(DeviceInput)
    assert dcp.combine_partial_attention(x, torch.zeros(2, 1, 1)) is None
