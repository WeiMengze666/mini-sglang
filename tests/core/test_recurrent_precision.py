"""State storage precision is independent of activations and token-cache precision."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.kda import KDAState
from minisgl.attention.kda.backend import KDAAttentionBackend
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import _adjust_config
from minisgl.kvcache import create_kvcache_pool
from minisgl.kvcache.config import HybridCacheConfig
from minisgl.kvcache.recurrent_pool import RecurrentStatePool


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_precision_follows_working_state_checkpoint_and_reordered_batch(
    unified, state_dtype, kv_dtype
):
    layout = HybridCacheConfig(2, (1,), (0,), 5, 2, 2, 3, 4)
    cache = create_kvcache_pool(
        SimpleNamespace(cache_config=layout),
        64,
        2,
        torch.bfloat16,
        torch.device("cpu"),
        max_running_req=3,
        max_cached_states=1,
        unified_memory=unified,
        state_dtype=state_dtype,
        kv_dtype=kv_dtype,
    )
    pool = cache.recurrent
    assert cache.dtype == kv_dtype and cache.mla.dtype == kv_dtype
    expected_bytes = 2 * 3 * 3 * state_dtype.itemsize + 3 * 2 * 3 * 3 * 2
    assert (
        layout.state_bytes_per_request(torch.bfloat16, 1, state_dtype=state_dtype) == expected_bytes
    )
    if unified:
        assert cache.arena.state_bytes >= expected_bytes
    else:
        assert cache.allocated_bytes == 128 * 7 * kv_dtype.itemsize + 3 * expected_bytes
    assert pool.conv_dtype == torch.bfloat16 and pool.state_dtype == state_dtype

    cache.begin_request(0, 10, 0)
    target = cache.get_kda_state(0, 0)
    values = KDAState(
        torch.linspace(-0.3, 0.7, target.conv.numel()).reshape_as(target.conv),
        torch.linspace(-0.8, 0.6, target.recurrent.numel()).reshape_as(target.recurrent),
    )
    expected = KDAState(values.conv.bfloat16(), values.recurrent.to(state_dtype))
    cache.set_kda_state(0, 0, values)
    cache.finish_request(0, 10, 4)
    checkpoint = pool.snapshot(0, 10)
    pool.restore(1, 11, checkpoint)
    cache.begin_request(0, 10, 4)
    cache.begin_request(1, 11, 4)
    for actual, reference in zip(cache.get_kda_state(0, 1), expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    # Kernel working values are FP32; storage conversion happens at commit.
    order = [1, 0]
    indices = pool.batch_indices(order)
    updated = KDAState(
        torch.stack([values.conv + 0.003, values.conv + 0.007]),
        torch.stack([values.recurrent + 0.003, values.recurrent + 0.007]),
    )
    pool.set_kda_batch(0, indices, updated)
    for row, slot in enumerate(order):
        actual = cache.get_kda_state(0, slot)
        for a, b in zip(actual, updated):
            torch.testing.assert_close(a, b[row].to(a.dtype), rtol=0, atol=0)

    # A checkpoint is immutable, and a physical arena move keeps the new dtype.
    cache.release_request(0, 10)
    pool.restore(2, 12, checkpoint)
    cache.begin_request(2, 12, 4)
    for a, b in zip(cache.get_kda_state(0, 2), expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    remapped = pool.get_kda_batch(0, pool.batch_indices([1, 2]))
    for row, slot in enumerate([1, 2]):
        for a, b in zip(cache.get_kda_state(0, slot), remapped):
            torch.testing.assert_close(a, b[row], rtol=0, atol=0)
    cache.reset()
    if unified:
        cache.arena.check_integrity()
        assert not cache.arena._states


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_graph_eligibility_uses_convolution_dtype_not_mla_storage(state_dtype):
    pool = RecurrentStatePool((0,), 1, 1, 128, 4, torch.bfloat16, "cpu", state_dtype=state_dtype)
    cache = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float8_e4m3fn, recurrent=pool)
    config = SimpleNamespace(cache_config=SimpleNamespace(head_dim=128))
    backend = KDAAttentionBackend(config, cache=cache)
    backend.init_capture_graph(4096, [1, 2])
    assert backend._graph_sizes == {1, 2}
    unsupported = SimpleNamespace(conv_dtype=torch.float32)
    cache.recurrent = unsupported
    with pytest.raises(ValueError, match="BF16"):
        backend.init_capture_graph(4096, [1])


@pytest.mark.parametrize("name,dtype", [("float32", torch.float32), ("bfloat16", torch.bfloat16)])
def test_server_resolves_independent_recurrent_storage(name, dtype):
    from minisgl.server.args import parse_args

    config, _ = parse_args(
        ["--model", "unused", "--dtype", "bfloat16", "--recurrent-state-dtype", name]
    )
    assert config.dtype == torch.bfloat16
    assert config.recurrent_state_dtype == dtype


def test_unsupported_state_precision_rejected_before_model_loading():
    with pytest.raises(ValueError, match="Recurrent state storage"):
        RecurrentStatePool((0,), 1, 1, 4, 2, torch.bfloat16, "cpu", state_dtype=torch.float8_e4m3fn)
    config = EngineConfig(
        "unused", DistributedInfo(0, 1), torch.bfloat16, recurrent_state_dtype=torch.float16
    )
    with pytest.raises(ValueError, match="Recurrent state storage"):
        _adjust_config(config)


def test_flashinfer_adapter_promotes_storage_before_working_state(monkeypatch):
    import sys

    from minisgl.attention.kda.flashinfer import recurrent_kda

    torch.manual_seed(71)
    q, k, v, forget = [torch.randn(3, 2, 4).bfloat16() for _ in range(4)]
    stored = torch.randn(2, 4, 4).bfloat16()
    original = stored.clone()

    def run(q, k, v, forget, beta, **kwargs):
        working = kwargs["initial_state"]
        assert working.dtype == torch.float32
        torch.testing.assert_close(working[0], stored.float().transpose(-1, -2), rtol=0, atol=0)
        return v, working + 0.125

    monkeypatch.setitem(sys.modules, "flashinfer", SimpleNamespace(recurrent_kda=run))
    output, final = recurrent_kda(
        q, k, v, forget, torch.randn(3, 2), torch.zeros(2), torch.zeros(8), stored, -5.0
    )
    assert final.dtype == torch.float32
    torch.testing.assert_close(output, v, rtol=0, atol=0)
    torch.testing.assert_close(final, stored.float() + 0.125, rtol=0, atol=0)
    torch.testing.assert_close(stored, original, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name,expected",
    [("auto", None), ("bfloat16", torch.bfloat16), ("fp8_e4m3", torch.float8_e4m3fn)],
)
def test_server_resolves_independent_token_cache_storage(name, expected):
    from minisgl.server.args import parse_args

    config, _ = parse_args(
        [
            "--model",
            "unused",
            "--dtype",
            "bfloat16",
            "--kv-cache-dtype",
            name,
            "--recurrent-state-dtype",
            "bfloat16",
        ]
    )
    assert config.dtype == torch.bfloat16
    assert config.kv_cache_dtype == expected
    assert config.cache_dtype == (expected or torch.bfloat16)
    assert config.recurrent_state_dtype == torch.bfloat16


def test_unsupported_token_cache_precision_rejected_before_loading():
    config = EngineConfig(
        "unused", DistributedInfo(0, 1), torch.bfloat16, kv_cache_dtype=torch.int8
    )
    with pytest.raises(ValueError, match="KV cache storage"):
        _adjust_config(config)
