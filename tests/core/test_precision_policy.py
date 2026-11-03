from types import SimpleNamespace

import pytest
import torch
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


@pytest.mark.parametrize("precision", ["strict", "default"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("backend", ["auto", "pynccl"])
def test_strict_nccl_uses_caller_buffers(monkeypatch, precision, dtype, backend):
    import minisgl.engine.engine as engine_module

    group = object()
    calls = []
    distributed = SimpleNamespace(
        init_process_group=lambda **kwargs: calls.append(("init", kwargs)),
        group=SimpleNamespace(WORLD=group),
    )
    monkeypatch.setattr(engine_module, "torch", SimpleNamespace(distributed=distributed))
    monkeypatch.setattr(
        engine_module,
        "enable_pynccl_distributed",
        lambda info, cpu_group, max_bytes: calls.append(("enable", info, cpu_group, max_bytes)),
    )
    engine = engine_module.Engine.__new__(engine_module.Engine)
    engine.ctx = SimpleNamespace(bf16_precision=precision)
    engine.dtype = dtype
    config = SimpleNamespace(
        distributed_backend=backend,
        use_pynccl=True,
        tp_info=SimpleNamespace(rank=0, size=8),
        distributed_timeout=60,
        distributed_addr="env://",
        max_forward_len=129,
        model_config=SimpleNamespace(hidden_size=64),
    )
    result = engine._init_communication(config)
    assert result is group
    assert calls[0][1]["backend"] == "gloo"
    assert calls[1][:3] == ("enable", config.tp_info, group)
    assert calls[1][3] == (0 if precision == "strict" else 129 * 64 * dtype.itemsize)
