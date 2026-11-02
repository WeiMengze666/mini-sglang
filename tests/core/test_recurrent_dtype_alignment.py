"""Mixed storage dtypes must remain aligned for non-model-specific state shapes."""

import pytest
import torch
from minisgl.kvcache.config import HybridCacheConfig
from minisgl.kvcache.hybrid_pool import HybridCache


@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("conv_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize("width", [1, 4])
def test_mixed_dtype_state_views_are_aligned_and_disjoint(
    unified, conv_dtype, state_dtype, dim, width
):
    layout = HybridCacheConfig(2, (1,), (0,), 3, 2, 1, dim, width)
    cache = HybridCache(
        layout,
        64,
        1,
        2,
        conv_dtype,
        torch.device("cpu"),
        unified_memory=unified,
        state_dtype=state_dtype,
    )
    cache.begin_request(0, 10, 0)
    cache.begin_request(1, 11, 0)
    first = cache.get_kda_state(0, 0)
    second = cache.get_kda_state(0, 1)
    for index, state in enumerate((first, second)):
        for part in state:
            assert part.data_ptr() % part.element_size() == 0
            values = torch.arange(part.numel()).reshape_as(part).float() / 7 + index + 0.13
            part.copy_(values)
    for index, state in enumerate((first, second)):
        for part in state:
            expected = (torch.arange(part.numel()).reshape_as(part).float() / 7 + index + 0.13).to(
                part.dtype
            )
            torch.testing.assert_close(part, expected, rtol=0, atol=0)
    payload = dim * dim * state_dtype.itemsize + 3 * dim * (width - 1) * conv_dtype.itemsize
    assert layout.state_bytes_per_request(conv_dtype, 1, state_dtype=state_dtype) == payload
    cache.reset()
    if unified:
        cache.arena.check_integrity()
