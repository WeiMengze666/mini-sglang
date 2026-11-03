"""Runtime arithmetic policies shared by model-independent layers."""

from __future__ import annotations

from contextlib import ExitStack

import torch


def resolve_bf16_precision(requested: str) -> str:
    """Resolve the BF16 arithmetic policy without referring to a model."""
    if requested not in ("auto", "default", "strict"):
        raise ValueError("bf16_precision must be auto, default or strict")
    if requested != "auto":
        return requested
    return "default"


def use_strict_bf16() -> bool:
    """Return whether device BF16 kernels should preserve strict rounding."""
    from minisgl.core import get_global_ctx

    try:
        ctx = get_global_ctx()
    except AssertionError:
        # Standalone kernel/component tests have no Engine context.
        return True
    return getattr(ctx, "bf16_precision", "strict") == "strict"


def native_precision_scope(policy: str, *, device: torch.device, dtype: torch.dtype) -> ExitStack:
    """Own the SDK execution policy used by the strict BF16 reference.

    Default execution and CPU references need no optional SDK. An existing
    external SDK scope keeps ownership; closing a Mini scope is idempotent.
    """
    scope = ExitStack()
    if (
        resolve_bf16_precision(policy) != "strict"
        or device.type != "cuda"
        or dtype != torch.bfloat16
    ):
        return scope
    from sglang.srt.batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
        is_batch_invariant_mode_enabled,
    )

    if not is_batch_invariant_mode_enabled():
        scope.callback(disable_batch_invariant_mode)
        try:
            enable_batch_invariant_mode()
        except BaseException:
            scope.close()
            raise
    return scope
