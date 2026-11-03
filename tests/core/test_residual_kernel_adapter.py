"""Library parameter views, callable fallback norm, and cached-weight ownership."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.layers import attn_residual as module


@pytest.mark.parametrize("normalize", [False, True])
def test_native_residual_adapter_keeps_sdk_state_out_of_layer_interfaces(monkeypatch, normalize):
    head = torch.arange(8.0).reshape(2, 4)
    rows = torch.stack((head + 1, head), dim=1)
    proj = SimpleNamespace(weight=torch.ones(1, 4))
    norm = SimpleNamespace(weight=torch.ones(4), eps=1e-5, forward=lambda x: x)
    output_norm = SimpleNamespace(weight=torch.ones(4), eps=1e-5, forward=lambda x: x + 5)
    views = []

    def mixture(prefix, bank, nvb, score, scoring_norm):
        assert prefix is head and bank is rows and nvb == 1
        assert scoring_norm.weight is norm.weight and scoring_norm.variance_epsilon == norm.eps
        assert score.weight is proj.weight
        views.append(score)
        # The real SDK owns this metadata on its parameter holder.
        score.sdk_metadata = "prepared"
        return prefix + bank[:, 0]

    def aggregate(prefix, bank, nvb, score, scoring_norm, out_norm):
        assert out_norm.weight is output_norm.weight
        assert out_norm.variance_epsilon == output_norm.eps
        return out_norm(mixture(prefix, bank, nvb, score, scoring_norm))

    monkeypatch.setattr(module, "_native_kernels", lambda: (aggregate, mixture, nullcontext))
    target = output_norm if normalize else None
    for _ in range(2):
        actual = module._read_native(head, rows, norm, proj, target)
        torch.testing.assert_close(actual, head + rows[:, 0] + (5 if normalize else 0))
    assert views[0] is views[1] and not hasattr(proj, "sdk_metadata")
    norm.weight = norm.weight.clone()
    module._read_native(head, rows, norm, proj, target)
    assert views[2] is not views[1]
