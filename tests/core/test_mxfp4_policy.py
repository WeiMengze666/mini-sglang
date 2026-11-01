import pytest
import torch

from minisgl.engine.config import EngineConfig
from minisgl.quantization.mxfp4 import MXFP4Linear


def test_engine_config_declares_packed_activation_precision():
    config = EngineConfig("/tmp/model", object(), torch.bfloat16)
    assert config.mxfp4_activation_precision == "bf16"


def test_mxfp4_linear_accepts_flashinfer_mxfp8_policy():
    op = MXFP4Linear(32, 16)
    op.set_backend("flashinfer", activation_precision="mxfp8")
    assert op._backend == "flashinfer"
    assert op._activation_precision == "mxfp8"


@pytest.mark.parametrize("backend", ["invalid", "flashinfer"])
def test_mxfp4_linear_rejects_invalid_policy(backend):
    op = MXFP4Linear(32, 16)
    precision = "bf16" if backend == "invalid" else "invalid"
    with pytest.raises(ValueError, match="activation precision|MXFP4 backend"):
        op.set_backend(backend, activation_precision=precision)
