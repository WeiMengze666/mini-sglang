"""Bounded work-grid contracts for grouped MXFP4 experts."""

import pytest
from minisgl.moe.mxfp4 import bounded_grid_size


@pytest.mark.parametrize(
    "blocks,columns,programs,expected",
    [(0, 1, 8, 1), (1, 1, 8, 1), (2, 4, 8, 8), (20, 4, 8, 8), (20, 4, 128, 80)],
)
def test_bounded_grid_size_uses_all_work_without_overlaunch(blocks, columns, programs, expected):
    assert bounded_grid_size(blocks, columns, programs) == expected


@pytest.mark.parametrize("blocks,columns,programs", [(1, 0, 8), (1, 1, 0), (-1, 1, 8)])
def test_bounded_grid_size_rejects_invalid_dimensions(blocks, columns, programs):
    with pytest.raises(ValueError):
        bounded_grid_size(blocks, columns, programs)
