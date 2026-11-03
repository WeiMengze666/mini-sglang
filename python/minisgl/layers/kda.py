"""KDA projections and output gating."""

from __future__ import annotations

import torch
from minisgl.attention.config import KDAConfig
from minisgl.attention.kda import (
    KDAState,
    causal_conv1d,
    gated_rms_norm,
)
from minisgl.attention.kda.registry import KDADecodeResult, KDAKernelSpec, get_kda_kernel
from minisgl.distributed.parallel import TensorParallel, local_size
from minisgl.layers.base import BaseOP
from minisgl.layers.linear import Linear
from minisgl.layers.norm import TorchRMSNorm


class _ConvWeight(BaseOP):
    def __init__(self, channels, width):
        self.weight = torch.empty(channels, 1, width, dtype=torch.float32)


class KDALayer(BaseOP):
    def __init__(self, config: KDAConfig) -> None:
        self._config = config
        self._kernel_plans = {}
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

    def after_load(self):
        self._kernel_plans.clear()

    def prepare_kernel(self, name, create):
        """Cache weight-dependent backend metadata until the next weight load."""
        if name not in self._kernel_plans:
            self._kernel_plans[name] = create()
        return self._kernel_plans[name]

    def project(self, x):
        """Project all real batch rows before touching request-owned state."""
        return (
            self.q_proj.forward(x),
            self.k_proj.forward(x),
            self.v_proj.forward(x),
            self.f_b_proj.forward(self.f_a_proj.forward(x)).reshape(-1, self._heads, self._dim),
            self.b_proj.forward(x),
        )

    def update(
        self,
        q,
        k,
        v,
        forget,
        beta,
        state=None,
        *,
        kernel: KDAKernelSpec | None = None,
        is_decode: bool = False,
    ):
        """Update one sequence [T,C] or independent equal-length sequences [B,T,C]."""
        if q.ndim not in (2, 3):
            raise ValueError("KDA projections must have shape [T,C] or [B,T,C]")
        batched = q.ndim == 3
        batch = q.shape[0] if batched else 1
        tokens = q.shape[-2]
        if is_decode and tokens != 1:
            raise ValueError("KDA decode requires one token per request")
        conv_state = None if state is None else state.conv
        values, histories = [], []
        for i, (projected, conv) in enumerate(
            zip((q, k, v), (self.q_conv1d, self.k_conv1d, self.v_conv1d))
        ):
            y, history = causal_conv1d(
                projected,
                conv.weight,
                None if conv_state is None else conv_state[..., i, :, :],
                is_decode=is_decode,
            )
            values.append(y.reshape(*q.shape[:-1], self._heads, self._dim))
            histories.append(history)
        # Independent requests are independent heads to either recurrence backend.
        # Convolution and persistent state retain their explicit request dimension.
        a, dt = self.A_log, self.dt_bias
        if batched:
            values = [part.transpose(0, 1).flatten(1, 2) for part in values]
            forget = forget.transpose(0, 1).flatten(1, 2)
            beta = beta.transpose(0, 1).flatten(1, 2)
            a, dt = a.repeat(batch), dt.repeat(batch)
        previous = None if state is None else state.recurrent.reshape(-1, self._dim, self._dim)
        kernel = get_kda_kernel("torch") if kernel is None else kernel
        output, recurrent = kernel.recurrence(
            *values, forget, beta, a, dt, previous, self._config.gate_lower_bound
        )
        if batched:
            output = output.reshape(tokens, batch, self._heads, self._dim).transpose(0, 1)
            recurrent = recurrent.reshape(batch, self._heads, self._dim, self._dim)
        return output, KDAState(torch.stack(histories, dim=-3), recurrent)

    def decode(self, x, state=None, *, kernel: KDAKernelSpec | None = None):
        """Try a complete fused one-token backend, then return ``None``.

        The backend owns only its supported kernel and state update.  This
        layer still applies the tensor-parallel output projection, which keeps
        fused KDA kernels reusable across models with different TP layouts.
        """
        kernel = get_kda_kernel("torch") if kernel is None else kernel
        if kernel.decode is None:
            return None
        result = kernel.decode(self, x, state)
        if result is None:
            return None
        if not isinstance(result, KDADecodeResult):
            raise TypeError("KDA decode backend must return KDADecodeResult or None")
        output = result.output
        if not result.normed:
            gate = (
                self.g_proj.forward(x)
                if self._full_gate
                else self.g_b_proj.forward(self.g_a_proj.forward(x))
            )
            output = gated_rms_norm(
                output, gate.reshape_as(output), self.o_norm.weight, self.o_norm.eps
            )
        output = self._tp.all_reduce(self.o_proj.forward(output.flatten(-2)))
        return output, result.state

    def finish(self, x, output):
        gate = (
            self.g_proj.forward(x)
            if self._full_gate
            else self.g_b_proj.forward(self.g_a_proj.forward(x))
        )
        output = gated_rms_norm(
            output, gate.reshape_as(output), self.o_norm.weight, self.o_norm.eps
        ).flatten(-2)
        return self._tp.all_reduce(self.o_proj.forward(output))

    def forward(
        self,
        x,
        positions=None,
        state: KDAState | None = None,
        *,
        kernel: KDAKernelSpec | None = None,
    ):
        kernel = get_kda_kernel("torch") if kernel is None else kernel
        projected = self.project(x) if kernel.project is None else kernel.project(self, x)
        output, state = self.update(*projected, state, kernel=kernel)
        return self.finish(x, output), state
