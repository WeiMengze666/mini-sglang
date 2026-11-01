"""Runtime arithmetic policies shared by model-independent layers."""

from __future__ import annotations


def resolve_bf16_precision(requested: str, attention_backend: str) -> str:
    """Resolve the BF16 arithmetic policy without referring to a model."""
    if requested not in ("auto", "default", "strict"):
        raise ValueError("bf16_precision must be auto, default or strict")
    if requested != "auto":
        return requested
    return "strict" if "recurrent-fi-triton" in attention_backend.split(",") else "default"


def use_strict_bf16() -> bool:
    """Return whether device BF16 kernels should preserve strict rounding."""
    from minisgl.core import get_global_ctx

    try:
        ctx = get_global_ctx()
    except AssertionError:
        # Standalone kernel/component tests have no Engine context.
        return True
    return getattr(ctx, "bf16_precision", "strict") == "strict"
