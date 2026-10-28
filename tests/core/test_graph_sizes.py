from types import SimpleNamespace

import pytest
from minisgl.core import Batch
from minisgl.engine.graph import GraphRunner, _determine_cuda_graph_bs


@pytest.mark.parametrize("maximum,expected", [(0, []), (1, [1]), (3, [1, 2]), (8, [1, 2, 4, 8])])
def test_graph_maximum_limits_every_bucket(maximum, expected):
    assert _determine_cuda_graph_bs(None, maximum, 90 << 30) == expected


@pytest.mark.parametrize("sizes", [[0], [-1, 2], [1, 1]])
def test_explicit_graph_buckets_reject_invalid_sizes(sizes):
    with pytest.raises(ValueError, match="distinct positive"):
        _determine_cuda_graph_bs(sizes, None, 90 << 30)


@pytest.mark.parametrize("extra,maximum", [([], None), (["--cuda-graph-max-bs", "0"], 0)])
def test_shell_preserves_graph_preference(extra, maximum):
    from minisgl.server.args import parse_args

    config, shell = parse_args(["--model", "unused", "--dtype", "bfloat16", "--shell-mode", *extra])
    assert shell and config.max_running_req == 1
    assert config.cuda_graph_max_bs == maximum


def test_explicit_graph_bucket_cli():
    from minisgl.server.args import parse_args

    config, _ = parse_args(
        ["--model", "unused", "--dtype", "bfloat16", "--cuda-graph-bs", "1", "2", "3"]
    )
    assert config.cuda_graph_bs == [1, 2, 3]


@pytest.mark.parametrize(
    "size,phase,expected",
    [(1, "decode", True), (3, "decode", False), (4, "decode", True), (1, "prefill", False)],
)
def test_stateful_graph_uses_only_exact_sizes(size, phase, expected):
    runner = object.__new__(GraphRunner)
    runner.attn_backend = SimpleNamespace(supports_graph_padding=False)
    runner.max_graph_bs = 4
    runner.graph_bs_list = [1, 2, 4]
    runner.graph_map = dict.fromkeys(runner.graph_bs_list)
    runner.dummy_req = object()
    batch = Batch([object()] * size, phase)
    assert runner.can_use_cuda_graph(batch) == expected
    runner.pad_batch(batch)
    assert batch.padded_size == size
