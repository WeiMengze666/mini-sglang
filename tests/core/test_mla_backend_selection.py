"""MLA cache storage follows its own backend, without a KDA selector."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.mla.backend import MLAAttentionBackend, MLAMetadata


@pytest.mark.parametrize("native_mla", [False, True])
def test_mla_store_policy_follows_its_executor(monkeypatch, native_mla):
    import minisgl.attention.mla.backend as module

    monkeypatch.setattr(module, "ContextParallel", lambda: SimpleNamespace(rank=0, size=1))
    stored = []

    class Layer:
        def project(self, x, positions):
            return x, x, x[:, :1]

        def attend_partial(self, q, *args, **kwargs):
            return q

        def finish(self, x, partial):
            return partial

    class NativeBackend(MLAAttentionBackend):
        cache_implementation = "triton"

    x = torch.randn(1, 4)
    cache = SimpleNamespace(
        device=x.device,
        store_mla=lambda *args, **kwargs: stored.append(kwargs["implementation"]),
        gather_mla=lambda *args: (x, x[:, :1]),
    )
    req = SimpleNamespace(extend_len=1, device_len=1, cached_len=0, table_idx=0)
    batch = SimpleNamespace(
        reqs=[req],
        positions=torch.tensor([0]),
        out_loc=torch.tensor([0]),
        attn_metadata=MLAMetadata(torch.tensor([0, 1])),
    )
    cls = NativeBackend if native_mla else MLAAttentionBackend
    backend = cls(None, cache=cache)
    output = backend.forward_layer(Layer(), 0, x, batch, cache, torch.tensor([[0]]))
    assert stored == ["triton" if native_mla else "torch"]
    torch.testing.assert_close(output, x, rtol=0, atol=0)
