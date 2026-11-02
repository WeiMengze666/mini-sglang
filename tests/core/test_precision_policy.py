import pytest
from minisgl.precision import resolve_bf16_precision


@pytest.mark.parametrize(
    ("requested", "expected"), [("auto", "default"), ("default", "default"), ("strict", "strict")]
)
def test_bf16_precision_policy(requested, expected):
    assert resolve_bf16_precision(requested) == expected


def test_bf16_precision_policy_rejects_unknown_values():
    with pytest.raises(ValueError, match="bf16_precision"):
        resolve_bf16_precision("invalid")


def test_engine_config_defaults_to_auto_precision_policy():
    import torch
    from minisgl.engine.config import EngineConfig

    config = EngineConfig("/tmp/model", object(), torch.bfloat16)
    assert config.bf16_precision == "auto"
