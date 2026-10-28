"""Indexed state follows owners across batch reorder and arena compaction."""

from math import prod

import pytest
import torch
from minisgl.attention.kda import KDAState
from minisgl.kvcache.recurrent_pool import RecurrentStatePool
from minisgl.kvcache.unified import UnifiedMemory


@pytest.mark.parametrize("unified", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("width", [1, 3])
def test_indexed_state_reorder_and_compaction(unified, dtype, width):
    state_bytes = 2 * (2 * 4 * 4 * 4 + 3 * 2 * 4 * (width - 1) * dtype.itemsize)
    arena = UnifiedMemory(128, 256, state_bytes, "cpu") if unified else None
    pool = RecurrentStatePool((1, 3), 4, 2, 4, width, dtype, "cpu", arena=arena)
    for slot in (0, 3, 1):
        pool.begin_request(slot, 100 + slot, 0)
        for layer in (1, 3):
            state = pool.get_kda_state(layer, slot)
            state.conv.fill_(slot + layer)
            state.recurrent.copy_(
                torch.arange(prod(state.recurrent.shape)).reshape_as(state.recurrent)
                + 10 * slot
                + layer
            )
    for order in ([3, 0], [1, 3]):
        indices = pool.batch_indices(order)
        for layer in (1, 3):
            expected = KDAState(
                *(
                    torch.stack(parts)
                    for parts in zip(*(pool.get_kda_state(layer, slot) for slot in order))
                )
            )
            actual = pool.get_kda_batch(layer, indices)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            updated = KDAState(*(part + 1 for part in actual))
            pool.set_kda_batch(layer, indices, updated)
            for row, slot in enumerate(order):
                for a, b in zip(pool.get_kda_state(layer, slot), updated):
                    torch.testing.assert_close(a, b[row], rtol=0, atol=0)
        if 0 in order:
            pool.release_request(0, 100)  # Moves a physical arena row.
    with pytest.raises(ValueError, match="twice"):
        pool.batch_indices([1, 1])
    with pytest.raises(RuntimeError, match="begin_request"):
        pool.batch_indices([2])
    for slot in (3, 1):
        pool.finish_request(slot, 100 + slot, 1)
        pool.release_request(slot, 100 + slot)
    if arena is not None:
        arena.check_integrity()
        assert not arena._states
