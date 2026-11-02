"""KDA numerical primitives; serving execution is in backend.py."""

from .reference import KDAState, causal_conv1d, gated_rms_norm, kda_log_decay, kda_recurrence

__all__ = ["KDAState", "causal_conv1d", "gated_rms_norm", "kda_log_decay", "kda_recurrence"]
