"""CPU protocol only: original storage/dtype and native preselection/errors."""

import pytest
import torch
from minisgl.kernel import dcp


class DeviceInput(torch.Tensor):
    @property
    def is_cuda(self):
        return True


def device(value):
    return value.as_subclass(DeviceInput)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_receives_original_strided_storage_and_returns_owned_sdk_result(monkeypatch, dtype):
    x = device(torch.arange(2 * 1 * 3 * 16).reshape(2, 1, 3, 16).to(dtype)[..., ::2])
    lse = device(torch.zeros(2, 1, 3, 2)[..., 0])
    x[0, 0, 0] = torch.nan
    lse[0, 0] = torch.tensor([-torch.inf, torch.nan, torch.inf])
    saved = x.clone()
    expected = torch.ones(1, 3, 8, dtype=dtype)
    calls = []

    def native(values, statistics, *, is_lse_base_on_e):
        assert values is x and statistics is lse and is_lse_base_on_e is False
        assert values.stride(-1) == 2 and statistics.stride(-1) == 2
        assert values[0, 0, 0].isnan().all()
        calls.append(1)
        return expected, None

    monkeypatch.setattr(dcp, "_native_lse_combine", lambda: native)
    actual = dcp.combine_partial_attention(x, lse, is_lse_base_on_e=False)
    assert calls == [1] and actual is expected and actual.dtype == dtype
    torch.testing.assert_close(x, saved, equal_nan=True, rtol=0, atol=0)


@pytest.mark.parametrize(
    "violation",
    [
        "cpu_output",
        "cpu_lse",
        "device",
        "dtype",
        "lse_dtype",
        "rank",
        "shape",
        "empty_shard",
        "empty_batch",
        "empty_head",
        "dimension",
        "large_dimension",
        "output_grad",
        "lse_grad",
        "output_neg",
        "lse_neg",
        "conj",
        "layout",
    ],
)
def test_unsupported_contract_declines_before_optional_import(monkeypatch, violation):
    monkeypatch.setattr(
        dcp, "_native_lse_combine", lambda: pytest.fail("unsupported input loaded SDK")
    )
    x, lse = device(torch.zeros(2, 1, 1, 8)), device(torch.zeros(2, 1, 1))
    if violation == "cpu_output":
        x = torch.zeros(2, 1, 1, 8)
    elif violation == "cpu_lse":
        lse = torch.zeros(2, 1, 1)
    elif violation == "device":

        class OtherDevice(DeviceInput):
            @property
            def device(self):
                return torch.device("cuda:1")

        lse = lse.as_subclass(OtherDevice)
    elif violation == "dtype":
        x = x.double()
    elif violation == "lse_dtype":
        lse = lse.double()
    elif violation == "rank":
        x = x[0]
    elif violation == "shape":
        lse = device(torch.zeros(2, 1, 2))
    elif violation.startswith("empty_"):
        shape = {
            "empty_shard": (0, 1, 1, 8),
            "empty_batch": (2, 0, 1, 8),
            "empty_head": (2, 1, 0, 8),
        }[violation]
        x, lse = device(torch.empty(shape)), device(torch.empty(shape[:-1]))
    elif violation in ("dimension", "large_dimension"):
        dim = 3 if violation == "dimension" else 2048
        x = device(torch.zeros(2, 1, 1, dim))
    elif violation == "output_grad":
        x.requires_grad_()
    elif violation == "lse_grad":
        lse.requires_grad_()
    elif violation == "output_neg":
        x = x._neg_view()
    elif violation == "lse_neg":
        lse = lse._neg_view()
    elif violation == "conj":

        class Conjugate(DeviceInput):
            def is_conj(self):
                return True

        x = x.as_subclass(Conjugate)
    else:

        class NonStrided(DeviceInput):
            @property
            def layout(self):
                return torch.sparse_coo

        x = x.as_subclass(NonStrided)
    assert dcp.combine_partial_attention(x, lse) is None


def test_unavailable_native_kernel_declines(monkeypatch):
    monkeypatch.setattr(dcp, "_native_lse_combine", lambda: None)
    assert (
        dcp.combine_partial_attention(device(torch.zeros(2, 1, 1, 8)), device(torch.zeros(2, 1, 1)))
        is None
    )


def test_selected_native_error_propagates_without_retry(monkeypatch):
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("selected DCP failure")

    monkeypatch.setattr(dcp, "_native_lse_combine", lambda: fail)
    with pytest.raises(RuntimeError, match="selected DCP failure"):
        dcp.combine_partial_attention(device(torch.zeros(2, 1, 1, 8)), device(torch.zeros(2, 1, 1)))
    assert calls == [1]
