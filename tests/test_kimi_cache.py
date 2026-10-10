from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.kimi import KimiBackend
from minisgl.kvcache.kimi_pool import (
    KimiHybridCache,
    kimi_cache_bytes_per_token,
    kimi_state_bytes_per_request,
)


def config():
    return SimpleNamespace(
        num_hidden_layers=3,
        kv_lora_rank=5,
        qk_rope_head_dim=2,
        linear_attn_config={
            "full_attn_layers": [2],
            "kda_layers": [1, 3],
            "num_heads": 2,
            "head_dim": 3,
            "short_conv_kernel_size": 4,
        },
    )


def pool():
    return KimiHybridCache(config(), 3, 4, 2, torch.float32, torch.device("cpu"))


def test_fixed_memory_and_mla_page_locations():
    cache = pool()
    expected = 12 * kimi_cache_bytes_per_token(config(), torch.float32)
    expected += 2 * kimi_state_bytes_per_request(config(), torch.float32)
    assert cache.allocated_bytes == expected
    loc = torch.tensor([7, 2, 9], dtype=torch.int32)
    latent, suffix = torch.randn(3, 5), torch.randn(3, 2)
    cache.store_mla(1, latent, suffix, loc)
    got_l, got_s = cache.gather_mla(1, loc.flip(0))
    torch.testing.assert_close(got_l, latent.flip(0))
    torch.testing.assert_close(got_s, suffix.flip(0))
    with pytest.raises(ValueError, match="padded"):
        cache.store_mla(1, torch.randn(4, 5), suffix, loc)
    with pytest.raises(IndexError):
        cache.gather_mla(1, torch.tensor([12]))
    with pytest.raises(ValueError):
        cache.gather_mla(1, torch.tensor([1.0]))


def test_request_continuation_isolation_and_reuse():
    cache = pool()
    cache.begin_request(0, 100, 0)
    a = cache.get_kda_state(0, 0)
    a.conv.fill_(3)
    a.recurrent.fill_(4)
    cache.begin_request(1, 200, 0)
    b = cache.get_kda_state(0, 1)
    assert not b.conv.any() and not b.recurrent.any()
    assert not cache.get_kda_state(2, 0).recurrent.any()
    cache.finish_request(0, 100, 5)
    cache.finish_request(1, 200, 8)
    cache.begin_request(0, 100, 5)
    torch.testing.assert_close(cache.get_kda_state(0, 0).recurrent, torch.full((2, 3, 3), 4.0))
    cache.finish_request(0, 100, 6)
    with pytest.raises(RuntimeError, match="cached_len"):
        cache.begin_request(0, 100, 5)
    with pytest.raises(RuntimeError, match="prefix"):
        cache.begin_request(0, 101, 6)
    cache.begin_request(0, 101, 0)
    assert not cache.get_kda_state(0, 0).conv.any()
    assert not cache.get_kda_state(0, 0).recurrent.any()
    cache.release_request(0, 100)  # late release from old owner cannot touch new owner
    cache.finish_request(0, 101, 1)
    cache.release_request(0, 101)
    assert cache._owners == [None, 200]
    cache.reset()
    assert cache._owners == [None, None]


def test_state_copy_shape_release_and_bounds():
    cache = pool()
    cache.begin_request(0, 1, 0)
    state = cache.get_kda_state(0, 0)
    cache.set_kda_state(
        0, 0, type(state)(torch.ones_like(state.conv), torch.ones_like(state.recurrent))
    )
    assert state.recurrent.dtype == torch.float32
    assert state.conv.sum().item() == state.conv.numel()
    with pytest.raises(ValueError):
        cache.set_kda_state(0, 0, type(state)(state.conv[:1], state.recurrent))
    with pytest.raises(RuntimeError, match="unfinished"):
        cache.begin_request(0, 2, 0)
    cache.release_request(0, 1)  # abort in the middle of a forward
    cache.begin_request(0, 2, 0)
    assert not cache.get_kda_state(0, 0).recurrent.any()
    with pytest.raises(IndexError):
        cache.begin_request(2, 3, 0)


def test_metadata_only_uses_real_requests_and_graphs_rejected():
    backend = KimiBackend.__new__(KimiBackend)
    backend.kvcache = pool()
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(extend_len=3), SimpleNamespace(extend_len=2)],
        padded_reqs=[SimpleNamespace(extend_len=999)],
    )
    backend.prepare_metadata(batch)
    assert batch.attn_metadata.get_last_indices(2).tolist() == [2, 4]
    assert batch.attn_metadata.get_last_indices(0).tolist() == []
    with pytest.raises(ValueError):
        batch.attn_metadata.get_last_indices(3)
    backend.init_capture_graph(100, [])
    with pytest.raises(ValueError, match="eager"):
        backend.init_capture_graph(100, [1])
    with pytest.raises(ValueError):
        backend.prepare_for_capture(batch)
    with pytest.raises(ValueError):
        backend.prepare_for_replay(batch)
