"""Physical kernel pages and DCP-widened scheduler/allocator locations."""

import pytest
import torch
from minisgl.distributed.info import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.scheduler.config import SchedulerConfig


@pytest.mark.parametrize("context_size", [1, 2, 4, 8])
@pytest.mark.parametrize("physical_page", [1, 64])
def test_dcp_widens_allocator_pages_without_changing_physical_page(context_size, physical_page):
    config = EngineConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 8),
        dtype=torch.bfloat16,
        dcp_size=context_size,
        page_size=physical_page,
    )
    assert config.page_size == physical_page
    assert config.logical_page_size == physical_page * context_size
    # Each request's first global allocation covers one complete physical page
    # on every context rank, including page_size=1 under DCP.
    for rank in range(context_size):
        local_positions = list(range(rank, config.logical_page_size, context_size))
        assert len(local_positions) == physical_page
        assert [position // context_size for position in local_positions] == list(
            range(physical_page)
        )


def test_scheduler_and_engine_share_the_same_page_contract():
    config = SchedulerConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 8),
        dtype=torch.bfloat16,
        dcp_size=8,
        page_size=64,
        max_prefill_tokens=32768,
    )
    assert config.page_size == 64 and config.logical_page_size == 512
    assert config.max_prefill_tokens % config.logical_page_size == 0


@pytest.mark.parametrize("kv_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("context_size", [2, 8])
def test_engine_capacity_counts_complete_physical_pages_on_each_rank(context_size, kv_dtype):
    from types import SimpleNamespace

    from minisgl.engine.engine import Engine

    config = EngineConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 8),
        dtype=torch.bfloat16,
        dcp_size=context_size,
        page_size=64,
        max_running_req=1,
        memory_ratio=0.5,
        kv_cache_dtype=kv_dtype,
    )
    seen = []

    def page_bytes(dtype, span):
        seen.append(span)
        return (span // context_size) * 6 * dtype.itemsize

    layout = SimpleNamespace(
        cache_bytes_per_page=page_bytes,
        state_bytes_per_request=lambda *args, **kwargs: 0,
    )
    object.__setattr__(config, "model_config", SimpleNamespace(cache_config=layout))
    engine = object.__new__(Engine)
    engine.dtype = torch.bfloat16
    engine.model = SimpleNamespace(
        state_dict=lambda: {"weight": torch.ones(16, dtype=engine.dtype)}
    )
    # Two usable physical pages and one dummy page cost 3 * 64 * 6 * 2
    # bytes on EVERY rank. The old global-page interpretation under-counts it.
    required = 16 * 2 + 3 * 64 * 6 * kv_dtype.itemsize
    with pytest.raises(MemoryError):
        engine._check_capacity(2 * required - 1, config)
    assert seen == [config.logical_page_size]
