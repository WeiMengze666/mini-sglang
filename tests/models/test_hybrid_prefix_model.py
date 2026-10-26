"""A restored recurrent prefix and shared MLA pages must equal recomputation."""

import pytest
import torch
from minisgl.kvcache.hybrid_pool import HybridCache
from minisgl.models import create_model
from minisgl.models.config import ModelConfig
from run_kimi_runtime_gate import batch_for, tiny_config


@pytest.mark.parametrize("unified", [False, True])
def test_restored_prefix_logits_and_compaction(unified):
    torch.manual_seed(89)
    config = ModelConfig.from_hf(tiny_config())
    model = create_model(config)
    for name, tensor in model.state_dict().items():
        tensor.normal_(std=0.06)
        if "norm.weight" in name:
            tensor.add_(1)
    cache = HybridCache(
        config.cache_config,
        128,
        1,
        2,
        torch.float32,
        torch.device("cpu"),
        max_cached_states=2,
        unified_memory=unified,
    )
    if unified:
        cache.arena.allocate_pages(list(range(16)) + list(range(64, 80)))
    table = torch.arange(128).reshape(2, 64)
    prefix, suffix = [1, 2, 3, 4, 5], [9, 12]
    model.forward_batch(batch_for(prefix, "cpu"), cache, table)
    checkpoint = cache.recurrent.snapshot(0, 10)
    cache.release_request(0, 10)  # moves the snapshot into the gap in unified mode
    cache.recurrent.restore(1, 20, checkpoint)
    table[1, : len(prefix)] = table[0, : len(prefix)]
    actual = model.forward_batch(
        batch_for(suffix, "cpu", start=len(prefix), slot=1, uid=20), cache, table, all_logits=True
    )
    expected = model.forward_batch(
        batch_for(prefix + suffix, "cpu", uid=30), cache, table, all_logits=True
    )[len(prefix) :]
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    cache.release_request(0, 30)
    cache.release_request(1, 20)
    cache.recurrent.restore(0, 40, checkpoint)
    replay = model.forward_batch(
        batch_for(suffix, "cpu", start=len(prefix), uid=40), cache, table, all_logits=True
    )
    torch.testing.assert_close(replay, expected, rtol=3e-5, atol=3e-6)
    if unified:
        cache.arena.check_integrity()
        before = cache.arena.free_pages
        cache.recurrent.drop_checkpoint(checkpoint)
        assert cache.arena.free_pages > before
