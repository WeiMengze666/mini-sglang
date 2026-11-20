from __future__ import annotations

import functools
from typing import Callable, Dict, TypeVar

import torch

T = TypeVar("T")


@functools.lru_cache(maxsize=None)
def empty_sentinel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Cached 0-element tensor for optional-tensor FFI slots (the numel-0
    "not present" convention). Allocating a fresh empty per call costs
    ~1.2us CPU on eager paths; the sentinel is never read, so one cached
    instance per (device, dtype) is safe to share."""
    return torch.empty(0, dtype=dtype, device=device)


_REGISTERED_CLASSES: Dict[type, type] = {}


def lazy_register_class(name: str, init_fn: Callable[[], None]) -> Callable[[T], T]:
    """A decorator to lazily register a tvm-ffi object class on first use.

    `init_fn` runs once (typically JIT-compiling and registering the C++
    reflection) right before the class is registered under the FFI type key
    `name`; afterwards instantiation proceeds normally.
    """

    def decorator(cls: T) -> T:
        def __new__(cls, *args, **kwargs):
            import tvm_ffi

            if cls not in _REGISTERED_CLASSES:
                init_fn()  # lazy initialization before registration once
                _REGISTERED_CLASSES[cls] = tvm_ffi.register_object(name)(cls)
            cls = _REGISTERED_CLASSES[cls]
            return original_new(cls, *args, **kwargs)

        original_new = cls.__new__
        cls.__new__ = __new__
        return cls

    return decorator
