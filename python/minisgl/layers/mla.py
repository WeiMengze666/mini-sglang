"""MLA projections and output gating."""

from __future__ import annotations

import torch
from minisgl.attention.config import MLAConfig
from minisgl.attention.mla import MLAState, apply_rotary, mla_attention
from minisgl.distributed.parallel import TensorParallel, local_size
from minisgl.layers.base import BaseOP
from minisgl.layers.linear import Linear
from minisgl.layers.norm import TorchRMSNorm


class MLALayer(BaseOP):
    state_kind = "mla"

    def __init__(self, config: MLAConfig) -> None:
        self._config = config
        self._tp = TensorParallel()
        self._heads = local_size(config.num_heads, self._tp.size, "MLA heads")
        self._nope = config.nope_dim
        self._rope = config.rope_dim
        self._value = config.value_dim
        self._latent = config.latent_dim
        self._skip_rope = not config.use_rope
        qdim = self._heads * (self._nope + self._rope)
        if config.q_lora_rank is None:
            self.q_proj = Linear(config.hidden_size, qdim)
        else:
            self.q_a_proj = Linear(config.hidden_size, config.q_lora_rank)
            self.q_a_layernorm = TorchRMSNorm(config.q_lora_rank, config.rms_norm_eps)
            self.q_b_proj = Linear(config.q_lora_rank, qdim)
        self.kv_a_proj_with_mqa = Linear(config.hidden_size, self._latent + self._rope)
        self.kv_a_layernorm = TorchRMSNorm(self._latent, config.rms_norm_eps)
        self.kv_b_proj = Linear(self._latent, self._heads * (self._nope + self._value))
        self._output_gate = config.output_gate
        if self._output_gate:
            self.g_proj = Linear(config.hidden_size, self._heads * self._value)
        self.o_proj = Linear(self._heads * self._value, config.hidden_size)

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
        return self._tp.all_reduce(self.o_proj.forward(output))

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
