import pytest
import torch
from minisgl.engine.config import EngineConfig
from minisgl.layers.linear import Linear


def test_engine_config_declares_packed_activation_precision():
    config = EngineConfig("/tmp/model", object(), torch.bfloat16)
    assert config.mxfp4_activation_precision == "auto"


def test_mxfp4_linear_accepts_flashinfer_mxfp8_policy():
    op = Linear(32, 16, weight_format="mxfp4")
    op.set_backend("flashinfer", activation_precision="mxfp8")
    assert op._backend == "flashinfer"
    assert op._activation_precision == "mxfp8"


@pytest.mark.parametrize("backend", ["invalid", "flashinfer"])
def test_mxfp4_linear_rejects_invalid_policy(backend):
    op = Linear(32, 16, weight_format="mxfp4")
    precision = "bf16" if backend == "invalid" else "invalid"
    with pytest.raises(ValueError, match="activation precision|MXFP4 backend"):
        op.set_backend(backend, activation_precision=precision)


def test_server_parses_flashinfer_precision_policy():
    from minisgl.server.args import parse_args

    args, _ = parse_args(
        [
            "--model-path",
            "/tmp/model",
            "--dtype",
            "bfloat16",
            "--mxfp4-backend",
            "flashinfer",
            "--mxfp4-activation-precision",
            "mxfp8",
        ]
    )
    assert args.mxfp4_backend == "flashinfer"
    assert args.mxfp4_activation_precision == "mxfp8"


def test_flashinfer_executor_validates_policy_without_cuda():
    from minisgl.moe.flashinfer import FlashInferMXFP4Experts

    with pytest.raises(ValueError, match="at least one local expert"):
        FlashInferMXFP4Experts(
            [],
            activation="silu",
            beta=1.0,
            linear_beta=None,
            num_global_experts=1,
            activation_precision="bf16",
        )
    with pytest.raises(ValueError, match="SiTU.*mxfp8"):
        FlashInferMXFP4Experts(
            [],
            activation="situ",
            beta=4.0,
            linear_beta=25.0,
            num_global_experts=1,
            activation_precision="bf16",
        )
