"""Communication policy is fixed before the first device query, without CUDA."""

import os
from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig

KEYS = ("NCCL_CUMEM_ENABLE", "NCCL_NVLS_ENABLE", "NCCL_GRAPH_MIXING_SUPPORT")


@pytest.mark.parametrize("dcp_size", [1, 8])
@pytest.mark.parametrize("preset", [{}, {KEYS[0]: "1", KEYS[1]: "2", KEYS[2]: "1"}, {KEYS[1]: "1"}])
def test_adjust_config_sets_only_missing_communication_defaults(monkeypatch, dcp_size, preset):
    import minisgl.engine.engine as engine
    from minisgl.distributed import info

    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k not in KEYS})
    for key, value in preset.items():
        monkeypatch.setenv(key, value)
    expected = {KEYS[0]: "0", KEYS[1]: "0"}
    if dcp_size > 1:
        expected[KEYS[2]] = "0"
    expected.update(preset)
    probes = []

    def device_capability():
        # _adjust_config consults capabilities before choosing native backends.
        assert {key: os.getenv(key) for key in KEYS} == {key: expected.get(key) for key in KEYS}
        probes.append(True)
        return False

    monkeypatch.setattr(engine, "is_sm100_supported", device_capability)
    monkeypatch.setattr(engine, "is_sm90_supported", device_capability)
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("CPU policy test initialized CUDA")
    )
    config = EngineConfig("unused", DistributedInfo(0, 8), torch.bfloat16, dcp_size=dcp_size)
    monkeypatch.setattr(info, "_TP_INFO", config.tp_info)
    config.__dict__["model_config"] = SimpleNamespace(
        hf_config=SimpleNamespace(quantization_config=None),
        cache_config=SimpleNamespace(state_bytes_per_request=lambda *args, **kwargs: 0),
        has_recurrent_state=True,
        is_moe=False,
        num_qo_heads=8,
    )
    engine._adjust_config(config)
    engine._adjust_config(config)  # Initialization is idempotent and keeps user overrides.
    assert probes
    assert {key: os.getenv(key) for key in KEYS} == {key: expected.get(key) for key in KEYS}
    assert not torch.cuda.is_initialized()
