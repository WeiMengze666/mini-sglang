"""TP=1 eager Kimi attention wrappers with original checkpoint parameter names."""

from __future__ import annotations

import torch
from minisgl.layers.base import BaseOP

from .kimi_attention import (
    KDAState,
    MLAState,
    apply_rotary,
    causal_conv1d,
    gated_rms_norm,
    kda_log_decay,
    kda_recurrence,
    mla_attention,
)
from .kimi_moe import KimiLinear, KimiRMSNorm


class _ConvWeight(BaseOP):
    def __init__(self, channels, width):
        self.weight = torch.empty(channels, 1, width, dtype=torch.float32)


class KimiKDA(BaseOP):
    def __init__(self, config):
        self._config = config
        linear = config.linear_attn_config
        self._heads, self._dim = linear["num_heads"], linear["head_dim"]
        width = self._heads * self._dim
        hidden = config.hidden_size
        self.q_proj = KimiLinear(hidden, width)
        self.k_proj = KimiLinear(hidden, width)
        self.v_proj = KimiLinear(hidden, width)
        self.b_proj = KimiLinear(hidden, self._heads)
        self.f_a_proj = KimiLinear(hidden, self._dim)
        self.f_b_proj = KimiLinear(self._dim, width)
        self._full_gate = linear.get("use_full_rank_gate", False)
        if self._full_gate:
            self.g_proj = KimiLinear(hidden, width)
        else:
            self.g_a_proj = KimiLinear(hidden, self._dim)
            self.g_b_proj = KimiLinear(self._dim, width)
        kernel = linear["short_conv_kernel_size"]
        self.q_conv1d = _ConvWeight(width, kernel)
        self.k_conv1d = _ConvWeight(width, kernel)
        self.v_conv1d = _ConvWeight(width, kernel)
        self.A_log = torch.empty(self._heads, dtype=torch.float32)
        self.dt_bias = torch.empty(width, dtype=torch.float32)
        self.o_norm = KimiRMSNorm(self._dim, config.rms_norm_eps)
        self.o_proj = KimiLinear(width, hidden)

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
            self._config.linear_attn_config.get("gate_lower_bound"),
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
        return self.o_proj.forward(output), KDAState(torch.stack(histories), recurrent)


class KimiMLA(BaseOP):
    def __init__(self, config):
        self._config = config
        self._heads = config.num_attention_heads
        self._nope = config.qk_nope_head_dim
        self._rope = config.qk_rope_head_dim
        self._value = config.v_head_dim
        self._latent = config.kv_lora_rank
        self._skip_rope = getattr(config, "mla_use_nope", True)
        if not self._skip_rope and getattr(config, "rope_scaling", None):
            raise ValueError("Eager Kimi MLA supports NoPE or unscaled RoPE only")
        qdim = self._heads * (self._nope + self._rope)
        if config.q_lora_rank is None:
            self.q_proj = KimiLinear(config.hidden_size, qdim)
        else:
            self.q_a_proj = KimiLinear(config.hidden_size, config.q_lora_rank)
            self.q_a_layernorm = KimiRMSNorm(config.q_lora_rank, config.rms_norm_eps)
            self.q_b_proj = KimiLinear(config.q_lora_rank, qdim)
        self.kv_a_proj_with_mqa = KimiLinear(config.hidden_size, self._latent + self._rope)
        self.kv_a_layernorm = KimiRMSNorm(self._latent, config.rms_norm_eps)
        self.kv_b_proj = KimiLinear(self._latent, self._heads * (self._nope + self._value))
        self._output_gate = getattr(config, "mla_use_output_gate", False)
        if self._output_gate:
            self.g_proj = KimiLinear(config.hidden_size, self._heads * self._value)
        self.o_proj = KimiLinear(self._heads * self._value, config.hidden_size)

    def project(self, x, positions):
        if self._config.q_lora_rank is None:
            q = self.q_proj.forward(x)
        else:
            q = self.q_b_proj.forward(self.q_a_layernorm.forward(self.q_a_proj.forward(x)))
        q = q.reshape(-1, self._heads, self._nope + self._rope)
        latent, key = self.kv_a_proj_with_mqa.forward(x).split((self._latent, self._rope), -1)
        latent = self.kv_a_layernorm.forward(latent)
        # K3 deliberately keeps this shared suffix unrotated: NoPE is not
        # equivalent to removing the suffix or excluding it from the scale.
        if not self._skip_rope and self._rope:
            theta = getattr(self._config, "rope_theta", 10000.0)
            interleaved = getattr(self._config, "rope_interleave", True)
            q = torch.cat(
                (
                    q[..., : self._nope],
                    apply_rotary(q[..., self._nope :], positions, theta, interleaved=interleaved),
                ),
                -1,
            )
            key = apply_rotary(key, positions, theta, interleaved=interleaved)
        return q, latent, key

    def attend(self, x, q, latent_history, key_history, positions, history_positions):
        output = mla_attention(
            q,
            latent_history,
            key_history,
            self.kv_b_proj.weight,
            positions,
            history_positions,
            nope_dim=self._nope,
            value_dim=self._value,
        ).flatten(-2)
        if self._output_gate:
            output = output * self.g_proj.forward(x).sigmoid()
        return self.o_proj.forward(output)

    def forward(self, x, positions, state: MLAState | None = None):
        """Convenience CPU/reference path; serving uses project/attend + pooled cache."""
        q, latent, key = self.project(x, positions)
        history_positions = positions
        if state is not None:
            latent = torch.cat((state.latent, latent))
            key = torch.cat((state.key, key))
            history_positions = torch.cat((state.positions, positions))
        if latent.shape[0] > self._config.max_position_embeddings:
            raise ValueError("MLA context exceeds max_position_embeddings")
        output = self.attend(x, q, latent, key, positions, history_positions)
        return output, MLAState(latent, key, history_positions)
