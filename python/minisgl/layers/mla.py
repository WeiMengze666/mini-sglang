"""MLA projections and output gating."""

from __future__ import annotations

import torch
from minisgl.attention.config import MLAConfig
from minisgl.attention.mla import MLALatentOutput, MLAOutput, MLAState, apply_rotary, mla_attention
from minisgl.distributed.context import ContextParallel
from minisgl.distributed.layout import get_parallel_layout
from minisgl.distributed.parallel import TensorParallel, local_size
from minisgl.layers.base import BaseOP
from minisgl.layers.linear import Linear, headwise_linear
from minisgl.layers.norm import TorchRMSNorm


class MLALayer(BaseOP):
    def __init__(self, config: MLAConfig) -> None:
        self._config = config
        self._tp = TensorParallel()
        self._cp = ContextParallel()
        self._heads = local_size(
            config.num_heads, get_parallel_layout().mla_decode.size, "MLA heads"
        )
        self._output_heads = local_size(config.num_heads, self._tp.size, "MLA output heads")
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
            self.g_proj = Linear(config.hidden_size, self._output_heads * self._value)
        self.o_proj = Linear(self._output_heads * self._value, config.hidden_size)

    def project(self, x, positions):
        if self._config.q_lora_rank is None:
            q = self.q_proj.forward(x)
        else:
            q = self.q_b_proj.forward(self.q_a_layernorm.forward(self.q_a_proj.forward(x)))
        q = q.reshape(-1, self._heads, self._nope + self._rope)
        latent, key = self.kv_a_proj_with_mqa.forward(x).split((self._latent, self._rope), -1)
        latent = self.kv_a_layernorm.forward(latent)
        # An unrotated shared suffix still participates in the attention score
        # and its scale; NoPE does not remove these projection dimensions.
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

    def attend_partial(
        self,
        q,
        latent_history,
        key_history,
        positions,
        history_positions,
        *,
        attention=mla_attention,
        causal_layout=None,
        context_sharded=True,
    ):
        """Attend cached history; full histories use only this rank's TP heads.

        KV-sharded fallbacks retain replicated query heads and return LSE for
        context exchange. Native paged decode instead returns compressed values,
        whose value projection is deferred until after that exchange.
        """
        weight = self.kv_b_proj.weight
        if not context_sharded and self._cp.size > 1:
            q = q.chunk(self._cp.size, dim=1)[self._cp.rank]
            weight = weight.view(self._heads, self._nope + self._value, self._latent)
            weight = weight.chunk(self._cp.size, dim=0)[self._cp.rank].flatten(0, 1)
        sharded = context_sharded and self._cp.size > 1
        output = attention(
            q,
            latent_history,
            key_history,
            weight,
            positions,
            history_positions,
            nope_dim=self._nope,
            value_dim=self._value,
            return_lse=sharded,
            causal_layout=causal_layout,
        )
        if isinstance(output, MLAOutput):
            return output
        return MLAOutput(*output) if sharded else MLAOutput(output)

    def attend_paged_partial(self, q, pages, metadata, *, attention):
        return attention.decode(
            q,
            pages,
            self.kv_b_proj.weight,
            metadata,
            nope_dim=self._nope,
            value_dim=self._value,
        )

    def finish(self, x, partial: MLAOutput):
        output = (
            self._cp.exchange(partial.values, partial.lse)
            if partial.lse is not None and self._cp.size > 1
            else partial.values
        )
        if isinstance(partial, MLALatentOutput):
            weight = self.kv_b_proj.weight.view(self._heads, self._nope + self._value, self._latent)
            weight = weight[:, self._nope :]
            if self._cp.size > 1:
                weight = weight.chunk(self._cp.size, dim=0)[self._cp.rank]
            output = headwise_linear(output.to(x.dtype), weight)
        output = output.to(x.dtype).flatten(-2)
        if self._output_gate:
            output = output * self.g_proj.forward(x).sigmoid()
        return self._tp.all_reduce(self.o_proj.forward(output))

    def attend(
        self,
        x,
        q,
        latent_history,
        key_history,
        positions,
        history_positions,
        *,
        attention=mla_attention,
    ):
        return self.finish(
            x,
            self.attend_partial(
                q, latent_history, key_history, positions, history_positions, attention=attention
            ),
        )

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
        owned = self._cp.owns(history_positions)
        latent, key, history_positions = latent[owned], key[owned], history_positions[owned]
        output = self.attend(x, q, latent, key, positions, history_positions)
        return output, MLAState(latent, key, history_positions)
