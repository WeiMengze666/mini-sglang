"""CPU-only environment defaults and worker device-selection ordering."""

import os
from types import SimpleNamespace

import pytest
import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig

KEYS = (
    "NCCL_CUMEM_ENABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_GRAPH_MIXING_SUPPORT",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "TRTLLM_ENABLE_PDL",
)


@pytest.fixture(autouse=True)
def cpu_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k not in KEYS})
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("CPU bootstrap test initialized CUDA")
    )


@pytest.mark.parametrize("dcp_size", [1, 8])
@pytest.mark.parametrize(
    "preset",
    [
        {},
        {KEYS[0]: "1", KEYS[1]: "2", KEYS[2]: "1", KEYS[3]: "32", KEYS[4]: "0"},
        {KEYS[1]: "1"},
    ],
)
def test_runtime_environment_sets_only_missing_defaults(monkeypatch, dcp_size, preset):
    from minisgl.engine.engine import _set_runtime_environment

    for key, value in preset.items():
        monkeypatch.setenv(key, value)
    expected = {KEYS[0]: "0", KEYS[1]: "0", KEYS[3]: "8", KEYS[4]: "1"}
    if dcp_size > 1:
        expected[KEYS[2]] = "0"
    expected.update(preset)
    config = SimpleNamespace(dcp_size=dcp_size)
    _set_runtime_environment(config)
    _set_runtime_environment(config)  # Idempotent; no device/config probing.
    assert {key: os.getenv(key) for key in KEYS} == {key: expected.get(key) for key in KEYS}
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("failure", [None, "device", "provider", "probe"])
def test_rank_device_precedes_provider_and_probe_with_original_startup_order(monkeypatch, failure):
    import minisgl.distributed.info as info
    import minisgl.distributed.layout as layout
    import minisgl.engine.engine as engine

    events = []
    selected = None
    failure_error = RuntimeError("startup failure propagates")
    rank = 1
    config = EngineConfig(
        "unused",
        DistributedInfo(rank, 8),
        torch.bfloat16,
        dcp_size=8,
        attention_provider="recording.provider",
        attention_backend="torch",
        cuda_graph_bs=[],
        max_seq_len_override=16384,
    )
    config.__dict__["model_config"] = SimpleNamespace(
        hf_config=SimpleNamespace(quantization_config=None),
        cache_config=SimpleNamespace(supports_context_sharding=True, state_bytes_per_request=lambda *args, **kwargs: 0),
        has_recurrent_state=True,
        is_moe=False,
        num_qo_heads=8,
    )
    monkeypatch.setattr(info, "_TP_INFO", None)
    monkeypatch.setattr(layout, "_DCP_SIZE", 1)
    monkeypatch.setattr(layout, "_EP_SIZE", 1)
    set_tp, configure = engine.set_tp_info, engine.configure_parallel

    def tp(**kwargs):
        set_tp(**kwargs)
        events.append("tp")

    def parallel(**kwargs):
        configure(**kwargs)
        events.append("layout")

    def environment_ready():
        assert [os.getenv(key) for key in KEYS] == ["0", "0", "0", "8", "1"]

    def device(value):
        nonlocal selected
        environment_ready()
        assert value == torch.device(f"cuda:{rank}")
        events.append("device")
        if failure == "device":
            raise failure_error
        selected = rank

    def provider(name):
        environment_ready()
        assert selected == rank and name == "recording.provider"
        events.append("provider")
        if failure == "provider":
            raise failure_error

    def probe():
        assert selected == rank and events[-1] == "provider"
        events.append("probe")
        if failure == "probe":
            raise failure_error
        return False

    def communication(self, value):
        assert value is config and self.device == torch.device(f"cuda:{rank}")
        assert self.ctx.max_forward_len == 16384
        events.append("tp_comm")
        return "tp-group"

    class ReachedPostCommunication(Exception):
        pass

    def stop(**kwargs):
        events.append(("dcp", kwargs))
        raise ReachedPostCommunication

    monkeypatch.setattr(engine, "set_tp_info", tp)
    monkeypatch.setattr(engine, "configure_parallel", parallel)
    monkeypatch.setattr(torch.cuda, "set_device", device)
    monkeypatch.setattr(engine, "load_attention_provider", provider)
    monkeypatch.setattr(engine, "is_sm100_supported", probe)
    monkeypatch.setattr(torch, "manual_seed", lambda seed: events.append(("seed", seed)))
    stream, copy_stream = object(), object()

    def make_stream(*, device=None):
        if device is None:
            events.append("stream")
            return stream
        assert device == torch.device(f"cuda:{rank}")
        events.append(("copy_stream", device))
        return copy_stream

    monkeypatch.setattr(torch.cuda, "Stream", make_stream)
    monkeypatch.setattr(
        torch.cuda, "set_stream", lambda value: events.append(("set_stream", value))
    )
    monkeypatch.setattr(engine, "set_global_ctx", lambda value: events.append("context"))
    monkeypatch.setattr(engine.Engine, "_init_communication", communication)
    monkeypatch.setattr(engine, "init_context_parallel", stop)
    if failure is None:
        with pytest.raises(ReachedPostCommunication):
            engine.Engine(config)
        assert events == [
            "tp",
            "layout",
            "device",
            "provider",
            "probe",
            ("seed", 42),
            "stream",
            ("copy_stream", torch.device(f"cuda:{rank}")),
            ("set_stream", stream),
            "context",
            "tp_comm",
            (
                "dcp",
                {
                    "backend": "nccl",
                    "device": torch.device(f"cuda:{rank}"),
                    "cpu_group": "tp-group",
                },
            ),
        ]
    else:
        with pytest.raises(RuntimeError) as caught:
            engine.Engine(config)
        assert caught.value is failure_error
        assert (
            events
            == ["tp", "layout", "device", "provider", "probe"][
                : {"device": 3, "provider": 4, "probe": 5}[failure]
            ]
        )
    assert not torch.cuda.is_initialized()


def test_fresh_cuda_assert_remains_before_any_bootstrap_side_effect(monkeypatch):
    import minisgl.engine.engine as engine

    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(
        engine, "set_tp_info", lambda **kwargs: pytest.fail("bootstrap preceded assert")
    )
    with pytest.raises(AssertionError):
        engine.Engine(SimpleNamespace())
    assert all(key not in os.environ for key in KEYS)


def test_invalid_tp_layout_still_fails_before_environment_or_device(monkeypatch):
    import minisgl.distributed.info as info
    import minisgl.engine.engine as engine

    monkeypatch.setattr(info, "_TP_INFO", None)
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda *args: pytest.fail("invalid layout selected CUDA")
    )
    with pytest.raises(ValueError, match="divisible"):
        engine.Engine(SimpleNamespace(tp_info=DistributedInfo(1, 8), dcp_size=3, ep_size=1))
    assert all(key not in os.environ for key in KEYS)
