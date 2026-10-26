import pytest
import torch
from minisgl.kvcache.unified import UnifiedMemory


def test_both_allocation_directions_and_compaction_preserve_logical_handles():
    arena = UnifiedMemory(12, 256, 768, "cpu")
    arena.allocate_state("a").fill_(7)
    arena.allocate_state("b").fill_(9)
    arena.allocate_pages([3, 8, 11])
    storage = arena.page_storage(torch.uint8)
    storage[arena.page_map[3]].fill_(13)
    storage[arena.page_map[8]].fill_(18)
    storage[arena.page_map[11]].fill_(21)
    assert arena.free_pages == 3
    arena.free_state("a")
    assert arena.state_view("b").eq(9).all()
    assert arena.free_pages == 6
    arena.free_pages_by_id([8])
    assert storage[arena.page_map[11]].eq(21).all()
    assert storage[arena.page_map[3]].eq(13).all()
    arena.allocate_state("c").fill_(17)
    assert arena.free_pages == 4
    arena.allocate_pages([0, 1, 2, 4])
    with pytest.raises(MemoryError):
        arena.allocate_state("full")
    arena.check_integrity()
    assert arena.state_view("b").eq(9).all() and arena.state_view("c").eq(17).all()


def test_allocation_failure_is_atomic_and_byte_capacity_changes_with_workload():
    arena = UnifiedMemory(8, 256, 1024, "cpu")
    arena.allocate_state(0)
    with pytest.raises(MemoryError):
        arena.allocate_pages([0, 1, 2, 3, 4])
    assert not arena._pages and arena.free_pages == 4
    arena.free_state(0)
    arena.allocate_pages(list(range(8)))
    assert arena.free_bytes == 0
    arena.free_pages_by_id([0, 1, 2, 3])
    arena.allocate_state(1)
    arena.check_integrity()


def test_scheduler_reclaims_optional_checkpoint_before_protected_kv(monkeypatch):
    import minisgl.core as core
    from minisgl.kvcache.config import HybridCacheConfig
    from minisgl.kvcache.hybrid_pool import HybridCache
    from minisgl.scheduler.cache import CacheManager

    ctx = core.Context(page_size=2)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    layout = HybridCacheConfig(2, (1,), (0,), 3, 2, 1, 8, 2)
    ctx.kv_cache = pool = HybridCache(
        layout,
        8,
        2,
        2,
        torch.float32,
        torch.device("cpu"),
        max_cached_states=2,
        unified_memory=True,
    )
    table = torch.empty(2, 16, dtype=torch.int32)
    cm = CacheManager(7, 2, table, "hybrid", pool=pool)
    handle = cm.prefix_cache.match_prefix(torch.arange(6, dtype=torch.int32)).cuda_handle
    cm.restore_state(0, 100, handle)
    pages = cm._allocate(3)
    table[0, :6] = cm._page_to_token(pages)
    pool.begin_request(0, 100, 0)
    pool.finish_request(0, 100, 6)
    req = core.Req(torch.arange(6, dtype=torch.int32), 0, 0, 4, 100, core.SamplingParams(), handle)
    req.cached_len = 6
    cm.cache_req(req, finished=False)
    assert len(pool.recurrent._checkpoints) == 1
    # Only one physical KV page is free. The snapshot occupies two pages' worth
    # of bytes; it can be discarded because the running request has private state.
    assert pool.arena.free_pages == 1
    more = cm._allocate(2)
    assert not pool.recurrent._checkpoints
    assert cm.prefix_cache.size_info.protected_size == 6
    torch.testing.assert_close(table[0, :6], cm._page_to_token(pages))
    cm._free(cm._page_to_token(more))
    pool.release_request(0, 100)
    cm.unlock(req.cache_handle)
    cm._free(cm.prefix_cache.evict(6))
    cm.check_integrity()
    pool.arena.check_integrity()
