"""MLA cache storage follows its own backend, without a KDA selector."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.mla import MLAOutput
from minisgl.attention.mla.backend import MLAAttentionBackend, MLAMetadata


@pytest.mark.parametrize("native_mla", [False, True])
def test_mla_store_policy_follows_its_executor(monkeypatch, native_mla):
    import minisgl.attention.mla.backend as module

    monkeypatch.setattr(
        module,
        "ContextParallel",
        lambda: SimpleNamespace(rank=0, size=1, gather_interleaved=lambda x, total: x),
    )
    stored = []

    class Layer:
        def project(self, x, positions, *, is_decode=False):
            return x, x, x[:, :1]

        def attend_partial(self, q, *args, **kwargs):
            return MLAOutput(q)

        def finish(self, x, partial):
            return partial.values

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
        is_decode=False,
        positions=torch.tensor([0]),
        out_loc=torch.tensor([0]),
        attn_metadata=MLAMetadata(torch.tensor([0, 1])),
    )
    cls = NativeBackend if native_mla else MLAAttentionBackend
    backend = cls(None, cache=cache)
    output = backend.forward_layer(Layer(), 0, x, batch, cache, torch.tensor([[0]]))
    assert stored == ["triton" if native_mla else "torch"]
    torch.testing.assert_close(output, x, rtol=0, atol=0)


@pytest.mark.parametrize("phase", ["prefill", "decode"])
@pytest.mark.parametrize("extend_len", [1, 2])
def test_paged_decode_follows_batch_phase(monkeypatch, phase, extend_len):
    import minisgl.attention.mla.backend as module
    from minisgl.core import Batch

    calls = []
    sentinel = object()
    table = torch.tensor([[0, 1, 2, 3]])
    monkeypatch.setattr(module, "get_global_ctx", lambda: SimpleNamespace(page_table=table))

    def decode_metadata(*args):
        calls.append(args)
        return sentinel

    cache = SimpleNamespace(
        device=torch.device("cpu"), mla=SimpleNamespace(decode_metadata=decode_metadata)
    )
    attention = SimpleNamespace(decode=lambda *args, **kwargs: None)
    backend = MLAAttentionBackend(None, attention=attention, cache=cache)
    batch = Batch([SimpleNamespace(extend_len=extend_len, device_len=4, table_idx=0)], phase)
    if phase == "decode" and extend_len != 1:
        with pytest.raises(ValueError, match="one new token"):
            backend.prepare_metadata(batch)
        assert not calls
        return
    backend.prepare_metadata(batch)
    assert (batch.attn_metadata.decode is sentinel) == (phase == "decode")
    assert len(calls) == int(phase == "decode")


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("first_key", [0, 1])
def test_portable_mla_fallback_needs_no_flashinfer(monkeypatch, dtype, return_lse, first_key):
    import builtins

    from minisgl.attention.mla import CausalLayout
    from minisgl.attention.mla.flashinfer import FlashInferMLAAdapter

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("flashinfer"):
            raise AssertionError("CPU fallback must not import optional CUDA code")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    torch.manual_seed(419)
    q = (torch.randn(3, 2, 5) * 0.2).to(dtype)
    latent = (torch.randn(6, 4) * 0.2).to(dtype)
    key = (torch.randn(6, 2) * 0.2).to(dtype)
    weight = (torch.randn(12, 4) * 0.2).to(dtype)
    positions, history = torch.tensor([0, 4, 12]), torch.arange(6) * 2 + first_key
    result = FlashInferMLAAdapter()(
        q,
        latent,
        key,
        weight,
        positions,
        history,
        nope_dim=3,
        value_dim=3,
        return_lse=return_lse,
        causal_layout=CausalLayout(0, first_key, 2),
    )
    expanded = torch.nn.functional.linear(latent, weight).reshape(6, 2, 6)
    keys, values = expanded.split(3, -1)
    keys = torch.cat((keys, key[:, None].expand(-1, 2, -1)), -1)
    scores = torch.einsum("thd,shd->hts", q.double(), keys.double()) / 5**0.5
    scores.masked_fill_((history[None, :] > positions[:, None])[None], -torch.inf)
    probabilities = torch.nan_to_num(scores.softmax(-1))
    expected = torch.einsum("hts,shv->thv", probabilities, values.double())
    if return_lse:
        actual, lse = result
        torch.testing.assert_close(lse, scores.logsumexp(-1).T.float(), rtol=1e-5, atol=1e-6)
    else:
        actual = result
    torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0.01, atol=1e-5)


def test_prefill_uses_unquantized_current_rows_and_dequantized_cached_prefix():
    from minisgl.attention.mla import MLAOutput
    from minisgl.attention.mla.backend import MLAMetadata

    q = torch.randn(5, 1, 6).bfloat16()
    fresh = torch.linspace(-0.37, 0.81, 20).reshape(5, 4).bfloat16()
    suffix = fresh[:, :2].clone()
    history = torch.linspace(-0.41, 0.73, 12).reshape(3, 4).to(torch.float8_e4m3fn)
    history_key = history[:, :2]
    readouts, writes, gathers = [], [], []

    class Layer:
        def project(self, x, positions, *, is_decode=False):
            return q, fresh, suffix

        def attend_partial(self, query, latent, key, positions, history_positions, **options):
            readouts.append((latent.clone(), key.clone(), history_positions.clone()))
            assert latent.dtype == key.dtype == query.dtype == torch.bfloat16
            assert not options["context_sharded"]
            return MLAOutput(query)

        def finish(self, x, partial):
            return partial.values

    def gather(layer_id, locations):
        gathers.append(locations.clone())
        return history, history_key

    cache = SimpleNamespace(
        device=torch.device("cpu"),
        store_mla=lambda layer, latent, key, *args, **kwargs: writes.append((latent, key)),
        gather_mla=gather,
    )
    batch = SimpleNamespace(
        positions=torch.tensor([0, 1, 2, 3, 4]),
        out_loc=torch.arange(5),
        is_decode=False,
        reqs=[
            SimpleNamespace(extend_len=3, cached_len=0, device_len=3, table_idx=0),
            SimpleNamespace(extend_len=2, cached_len=3, device_len=5, table_idx=1),
        ],
        attn_metadata=MLAMetadata(torch.tensor([0, 3, 5])),
    )
    backend = MLAAttentionBackend(None, cache=cache)
    result = backend.forward_layer(
        Layer(), 0, torch.zeros(5, 4), batch, cache, torch.arange(12).reshape(2, 6)
    )
    torch.testing.assert_close(result, q, rtol=0, atol=0)
    assert len(writes) == 1 and writes[0][0] is fresh and writes[0][1] is suffix
    assert len(gathers) == 1 and gathers[0].tolist() == [6, 7, 8]
    torch.testing.assert_close(readouts[0][0], fresh[:3], rtol=0, atol=0)
    torch.testing.assert_close(readouts[0][1], suffix[:3], rtol=0, atol=0)
    torch.testing.assert_close(
        readouts[1][0], torch.cat((history.bfloat16(), fresh[3:])), rtol=0, atol=0
    )
    torch.testing.assert_close(
        readouts[1][1], torch.cat((history_key.bfloat16(), suffix[3:])), rtol=0, atol=0
    )
    assert readouts[1][2].tolist() == list(range(5))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_flashinfer_prefill_precision_follows_cache_format(dtype):
    from minisgl.attention.fi import FlashInferMLABackend

    cache = SimpleNamespace(dtype=dtype)
    backend = FlashInferMLABackend(None, cache=cache)
    assert backend.kvcache is cache
    assert backend.attention.kv_dtype == dtype
