"""KDA projections and output gating."""

from __future__ import annotations

import torch
from minisgl.attention.config import KDAConfig
from minisgl.attention.kda import (
    KDAState,
    causal_conv1d,
    gated_rms_norm,
    kda_log_decay,
    kda_recurrence,
)
from minisgl.distributed.parallel import TensorParallel, local_size
from minisgl.layers.base import BaseOP
from minisgl.layers.linear import Linear
from minisgl.layers.norm import TorchRMSNorm


class _ConvWeight(BaseOP):
    def __init__(self, channels, width):
        self.weight = torch.empty(channels, 1, width, dtype=torch.float32)


class KDALayer(BaseOP):
    state_kind = "recurrent"

    def __init__(self, config: KDAConfig) -> None:
        self._config = config
        self._tp = TensorParallel()
        self._heads = local_size(config.num_heads, self._tp.size, "KDA heads")
        self._dim = config.head_dim
        width = self._heads * self._dim
        hidden = config.hidden_size
        self.q_proj = Linear(hidden, width)
        self.k_proj = Linear(hidden, width)
        self.v_proj = Linear(hidden, width)
        self.b_proj = Linear(hidden, self._heads)
        self.f_a_proj = Linear(hidden, self._dim)
        self.f_b_proj = Linear(self._dim, width)
        self._full_gate = config.full_rank_gate
        if self._full_gate:
            self.g_proj = Linear(hidden, width)
        else:
            self.g_a_proj = Linear(hidden, self._dim)
            self.g_b_proj = Linear(self._dim, width)
        kernel = config.conv_kernel_size
        self.q_conv1d = _ConvWeight(width, kernel)
        self.k_conv1d = _ConvWeight(width, kernel)
        self.v_conv1d = _ConvWeight(width, kernel)
        self.A_log = torch.empty(self._heads, dtype=torch.float32)
        self.dt_bias = torch.empty(width, dtype=torch.float32)
        self.o_norm = TorchRMSNorm(self._dim, config.rms_norm_eps)
        self.o_proj = Linear(width, hidden)

    def forward(self, x, positions=None, state: KDAState | None = None):
        channels = self._heads * self._dim
        conv_state = (
            None
            if state is None
            else state.conv.reshape(3, channels, self.q_conv1d.weight.shape[-1] - 1)
        )
        values, histories = [], []
        for i, (proj, conv) in enumerate(
            (
                (self.q_proj, self.q_conv1d),
                (self.k_proj, self.k_conv1d),
                (self.v_proj, self.v_conv1d),
            )
        ):
            y, history = causal_conv1d(
                proj.forward(x), conv.weight, None if conv_state is None else conv_state[i]
            )
            values.append(y.reshape(-1, self._heads, self._dim))
            histories.append(history)
        forget = self.f_b_proj.forward(self.f_a_proj.forward(x))
        decay = kda_log_decay(
            forget.reshape(-1, self._heads, self._dim),
            self.A_log,
            self.dt_bias,
            self._config.gate_lower_bound,
        )
        beta = self.b_proj.forward(x).float().sigmoid()
        output, recurrent = kda_recurrence(
            *values, decay, beta, None if state is None else state.recurrent
        )
        gate = (
            self.g_proj.forward(x)
            if self._full_gate
            else self.g_b_proj.forward(self.g_a_proj.forward(x))
        )
        output = gated_rms_norm(
            output, gate.reshape_as(output), self.o_norm.weight, self.o_norm.eps
        ).flatten(-2)
        return self._tp.all_reduce(self.o_proj.forward(output)), KDAState(
            torch.stack(histories), recurrent
        )
