"""KV storage capability must be independent of the recurrence implementation."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.recurrent import forward_layer


@pytest.mark.parametrize("recurrence", ["torch", "flashinfer", "triton"])
@pytest.mark.parametrize("native_mla", [False, True])
def test_mla_store_policy_follows_its_executor(monkeypatch, recurrence, native_mla):
    import minisgl.attention.recurrent as module

    monkeypatch.setattr(module, "ContextParallel", lambda: SimpleNamespace(rank=0, size=1))
    stored = []

    class Layer:
        state_kind = "mla"

        def project(self, x, positions):
            return x, x, x[:, :1]

        def attend_partial(self, q, *args, **kwargs):
            return q

        def finish(self, x, partial):
            return partial

    class Native:
        def __call__(self, *args, **kwargs):
            raise AssertionError("The fake layer owns this test's arithmetic")

        def decode(self, *args, **kwargs):
            raise AssertionError("No paged decode metadata was prepared")

    x = torch.randn(1, 4)
    cache = SimpleNamespace(
        store_mla=lambda *args, **kwargs: stored.append(kwargs["implementation"]),
        gather_mla=lambda *args: (x, x[:, :1]),
    )
    req = SimpleNamespace(extend_len=1, device_len=1, cached_len=0, table_idx=0)
    batch = SimpleNamespace(reqs=[req], positions=torch.tensor([0]), out_loc=torch.tensor([0]))
    executor = Native() if native_mla else lambda *args: None
    output = forward_layer(
        Layer(),
        0,
        x,
        batch,
        cache,
        torch.tensor([[0]]),
        kda_implementation=recurrence,
        mla_attention=executor,
    )
    assert stored == ["triton" if native_mla else "torch"]
    torch.testing.assert_close(output, x, rtol=0, atol=0)
