"""Adapters for mature native pointwise activation kernels."""

from functools import cache

import torch


@cache
def _sigmoid_mul_ops():
    from tvm_ffi import use_torch_stream

    from .latent import output_gate as mla_output_gate

    return mla_output_gate, use_torch_stream


@cache
def _warm_sigmoid_mul():
    ops, _ = _sigmoid_mul_ops()
    # Resolve the same cached module used by the SDK public operation.
    # No input/output tensors are retained in this cache.
    return ops._jit_mla_output_gate_module()


def try_sigmoid_mul(x, gate):
    """Return an owned output, or decline before compilation/launch.

    The SDK preserves bf16(sigmoid(gate)) followed by the bf16 multiply.
    Its public operation allocates a fresh output; both inputs stay read-only.
    """
    # These native ABI limits are not checked by the SDK covered() predicate.
    if (
        x.device != gate.device
        or x.data_ptr() % 16 != 0
        or gate.data_ptr() % 16 != 0
        or x.numel() >= 1 << 32
    ):
        return None
    try:
        ops, stream = _sigmoid_mul_ops()
    except ImportError:
        # Optional Python dependencies can be absent on reference CUDA paths.
        # This is before selection; JIT/launch errors below are never caught.
        return None
    if not ops.covered(x, gate):
        return None
    if torch.cuda.is_current_stream_capturing() and not _warm_sigmoid_mul.cache_info().currsize:
        raise RuntimeError("Warm native sigmoid-mul before CUDA graph capture")
    _warm_sigmoid_mul()
    with stream():
        # Once selected, compilation/launch errors propagate without a retry.
        return ops.kimi_k3_mla_output_gate(x, gate)
