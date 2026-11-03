"""Ownership and failure cleanup for optional SDK precision state."""

import builtins
import sys
from types import SimpleNamespace

import pytest
import torch
from minisgl.precision import native_precision_scope


@pytest.mark.parametrize(
    "policy,device,dtype",
    [
        ("default", "cuda:0", torch.bfloat16),
        ("strict", "cpu", torch.bfloat16),
        ("strict", "cuda:0", torch.float32),
    ],
)
def test_portable_precision_does_not_import_optional_sdk(monkeypatch, policy, device, dtype):
    original = builtins.__import__

    def without_sdk(name, *args, **kwargs):
        if name.startswith("sglang"):
            raise AssertionError("Portable precision imported the optional SDK")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_sdk)
    with native_precision_scope(policy, device=torch.device(device), dtype=dtype):
        pass


@pytest.mark.parametrize("external", [False, True])
def test_native_precision_scope_preserves_external_ownership_and_restores_once(
    monkeypatch, external
):
    state = {"enabled": external, "entered": 0, "closed": 0}

    def enable():
        state["enabled"] = True
        state["entered"] += 1

    def disable():
        state["enabled"] = False
        state["closed"] += 1

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.batch_invariant_ops",
        SimpleNamespace(
            enable_batch_invariant_mode=enable,
            disable_batch_invariant_mode=disable,
            is_batch_invariant_mode_enabled=lambda: state["enabled"],
        ),
    )
    scope = native_precision_scope("strict", device=torch.device("cuda:0"), dtype=torch.bfloat16)
    assert state["enabled"]
    with native_precision_scope("strict", device=torch.device("cuda:0"), dtype=torch.bfloat16):
        assert state["enabled"]
    assert state["enabled"]
    scope.close()
    scope.close()
    assert state == {"enabled": external, "entered": int(not external), "closed": int(not external)}


def test_native_precision_partial_enable_failure_is_cleaned_up(monkeypatch):
    state = {"enabled": False}

    def enable():
        state["enabled"] = True
        raise RuntimeError("SDK enable failed")

    def disable():
        state["enabled"] = False

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.batch_invariant_ops",
        SimpleNamespace(
            enable_batch_invariant_mode=enable,
            disable_batch_invariant_mode=disable,
            is_batch_invariant_mode_enabled=lambda: state["enabled"],
        ),
    )
    with pytest.raises(RuntimeError, match="SDK enable failed"):
        native_precision_scope("strict", device=torch.device("cuda:0"), dtype=torch.bfloat16)
    assert not state["enabled"]
