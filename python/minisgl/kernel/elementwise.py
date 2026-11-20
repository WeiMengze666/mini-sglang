"""Thin adapters for mature native elementwise kernels."""

from functools import cache

import torch


@cache
def _add3_ops():
    from minisgl.kernel import add3
    from tvm_ffi import use_torch_stream

    return add3, use_torch_stream


@cache
def _warm_add3():
    ops, _ = _add3_ops()
    return ops._jit_add3_module()


def try_add3(a, b, c):
    """Select the SDK operation before JIT; propagate any selected failure."""
    # covered() uses the widest architecture's 16 bf16 elements. Match its
    # 32-byte vector alignment conservatively without querying CUDA hardware.
    if (
        a.device != b.device
        or a.device != c.device
        or any(value.data_ptr() % 32 for value in (a, b, c))
    ):
        return None
    try:
        ops, stream = _add3_ops()
    except ImportError:
        return None  # Optional SDK unavailable before selection.
    if not ops.covered(a, b, c):
        return None
    if torch.cuda.is_current_stream_capturing() and not _warm_add3.cache_info().currsize:
        raise RuntimeError("Warm native add3 before CUDA graph capture")
    _warm_add3()
    with stream():
        # Generic callers do not promise b/c producers are two kernels back.
        # The SDK waits before all loads and preserves two bf16 additions.
        return ops.add3(a, b, c, prefetch_bc=False)
