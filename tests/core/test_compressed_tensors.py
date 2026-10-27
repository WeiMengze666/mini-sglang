"""Compressed checkpoint rules apply to any BaseOP tree."""

import torch
from minisgl.layers.base import BaseOP, OPList
from minisgl.layers.linear import Linear
from minisgl.quantization.compressed_tensors import replace_quantized_linears
from minisgl.quantization.mxfp4 import MXFP4Linear


def test_generic_selection_preserves_expert_ids_and_dense_exclusions():
    model = BaseOP()
    with torch.device("meta"):
        model.experts = OPList([Linear(32, 16), Linear(32, 16)], indices=[2, 8])
        model.score = Linear(32, 1)
    quant = {
        "format": "mxfp4-pack-quantized",
        "config_groups": {
            "weights": {
                "targets": ["Linear"],
                "weights": {"num_bits": 4, "type": "float", "strategy": "group", "group_size": 32},
            }
        },
        "ignore": ["score"],
    }
    assert replace_quantized_linears(model, quant) == 2
    assert isinstance(model.experts.get_op(2), MXFP4Linear)
    assert isinstance(model.score, Linear)
    assert set(model.state_dict()) == {
        "experts.2.weight_packed",
        "experts.2.weight_scale",
        "experts.8.weight_packed",
        "experts.8.weight_scale",
        "score.weight",
    }
