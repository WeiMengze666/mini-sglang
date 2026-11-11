from __future__ import annotations

from dataclasses import dataclass, field

from minisgl.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_prefill_tokens: int = 16384
    chunked_prefill_size: int = 16384
    state_cache_strategy: str = "auto"
    cache_type: str = "radix"
    offline_mode: bool = False
    trust_remote_code: bool = False

    def __post_init__(self):
        for name in ("max_prefill_tokens", "chunked_prefill_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.state_cache_strategy not in ("auto", "extra_buffer"):
            raise ValueError("Unsupported recurrent prefix-cache strategy")

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/minisgl_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/minisgl_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/minisgl_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return max(self.max_prefill_tokens, self.chunked_prefill_size)

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
