"""Mechanism layers must work without a model-family config or global Context."""

import pytest
import torch
from minisgl.attention.config import KDAConfig, MLAConfig
from minisgl.kvcache.config import HybridCacheConfig
from minisgl.kvcache.hybrid_pool import HybridCache
from minisgl.layers.kda import KDALayer
from minisgl.layers.mla import MLALayer


@pytest.mark.parametrize(
    "layer",
    [
        lambda: KDALayer(KDAConfig(12, 2, 4, 3, full_rank_gate=True, gate_lower_bound=-5)),
        lambda: MLALayer(MLAConfig(12, 2, 3, 4, 3, 5, 6, 32, output_gate=True)),
    ],
)
def test_mechanism_config_streaming(layer):
    torch.manual_seed(67)
    op = layer()
    for name, tensor in op.state_dict().items():
        tensor.normal_(std=0.1)
        if "norm.weight" in name:
            tensor.fill_(1)
    x, positions = torch.randn(7, 12), torch.arange(7)
    expected, expected_state = op.forward(x, positions)
    outputs, state = [], None
    for start, stop in ((0, 3), (3, 6), (6, 7)):
        output, state = op.forward(x[start:stop], positions[start:stop], state)
        outputs.append(output)
    torch.testing.assert_close(torch.cat(outputs), expected, atol=3e-6, rtol=3e-5)
    for actual, reference in zip(state, expected_state):
        torch.testing.assert_close(actual, reference, atol=3e-6, rtol=3e-5)


def test_cache_layout_is_independent_of_checkpoint_schedule_conventions():
    layout = HybridCacheConfig(3, (1,), (0, 2), 5, 2, 2, 3, 4)
    cache = HybridCache(layout, 3, 4, 2, torch.float32, torch.device("cpu"))
    assert cache.allocated_bytes == (
        12 * layout.cache_bytes_per_token(torch.float32)
        + 2 * layout.state_bytes_per_request(torch.float32)
    )
    cache.begin_request(0, 101, 0)
    state = cache.get_kda_state(2, 0)
    state.recurrent.fill_(7)
    cache.finish_request(0, 101, 4)
    cache.release_request(0, 100)
    cache.begin_request(0, 101, 4)
    assert state.recurrent.eq(7).all()
    cache.release_request(0, 101)
    cache.begin_request(0, 102, 0)
    assert not cache.get_kda_state(2, 0).recurrent.any()
    with pytest.raises(ValueError, match="partition"):
        HybridCacheConfig(3, (1,), (0, 1), 5, 2, 2, 2, 4)
