"""Bind vendored C++ operators to TVM-FFI on the worker's CUDA architecture."""

import os
from threading import RLock

import torch

from .utils import load_jit as _load_jit
from .utils import make_cpp_args

_compile_lock = RLock()


def load_jit(name, *args, **options):
    # TVM-FFI reads its target from the environment and otherwise adds a
    # generic SM target. Architecture-specific instructions require a single
    # target for the detected device, including during the PTX assembly step.
    major, minor = torch.cuda.get_device_capability()
    arch = f"{major}.{minor}{'a' if major >= 9 else ''}"
    key = "TVM_FFI_CUDA_ARCH_LIST"
    with _compile_lock:
        previous = os.environ.get(key)
        os.environ[key] = arch
        try:
            return _load_jit(
                name, *args, "sm" + arch.replace(".", ""), namespace="sglang", **options
            )
        finally:
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


__all__ = ["load_jit", "make_cpp_args"]
