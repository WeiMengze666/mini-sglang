"""MLA projections and layer-owned native decode weight plans."""

from __future__ import annotations

import torch
from minisgl.attention.config import MLAConfig
from minisgl.attention.mla import MLALatentOutput, MLAOutput, MLAState, apply_rotary, mla_attention
from minisgl.distributed.context import ContextParallel
from minisgl.distributed.layout import get_parallel_layout
from minisgl.distributed.parallel import TensorParallel, local_size
from minisgl.layers.activation import sigmoid_mul
from minisgl.layers.base import BaseOP
from minisgl.layers.linear import (
    Linear,
    LinearRowParallel,
    MergedLinear,
    headwise_linear,
    linear_bf16,
)
from minisgl.layers.norm import TorchRMSNorm


class _LatentKVProjection(Linear):
    def new_with_weight_format(self, weight_format, *, chunk_rows=128):
        raise NotImplementedError("MLA kv_b_proj requires a dense weight for latent attention")


class MLALayer(BaseOP):
    @staticmethod
    def weight_shards(tp, heads, nope_dim, rope_dim, value_dim):
        query = tp.shard(0, heads, "MLA heads", nope_dim + rope_dim)
        return {
            "q_proj.weight": query,
            "q_b_proj.weight": query,
            "kv_b_proj.weight": tp.shard(0, heads, "MLA heads", nope_dim + value_dim),
            "g_proj.weight": tp.shard(0, heads, "MLA heads", value_dim),
            "o_proj.weight": tp.shard(1, heads, "MLA heads", value_dim),
        }

    def __init__(self, config: MLAConfig) -> None:
        self._config = config
        self._tp, self._cp = TensorParallel(), ContextParallel()
        self._heads = local_size(config.num_heads, self._tp.size, "MLA heads")
        self._decode_heads = local_size(
            config.num_heads, get_parallel_layout().mla_decode.size, "MLA decode heads"
        )
        self._nope, self._rope = config.nope_dim, config.rope_dim
        self._value, self._latent = config.value_dim, config.latent_dim
        self._skip_rope = not config.use_rope
        shards = self.weight_shards(
            self._tp.group, config.num_heads, self._nope, self._rope, self._value
        )
        qdim = shards["q_proj.weight"].length
        if config.q_lora_rank is None:
            self.q_proj = Linear(config.hidden_size, qdim)
        else:
            self.q_a_proj = Linear(config.hidden_size, config.q_lora_rank)
            self.q_a_layernorm = TorchRMSNorm(config.q_lora_rank, config.rms_norm_eps)
            self.q_b_proj = Linear(config.q_lora_rank, qdim)
        self.kv_a_proj_with_mqa = Linear(config.hidden_size, self._latent + self._rope)
        self.kv_a_layernorm = TorchRMSNorm(self._latent, config.rms_norm_eps)
        self.kv_b_proj = _LatentKVProjection(self._latent, shards["kv_b_proj.weight"].length)
        self._output_gate = config.output_gate
        if self._output_gate:
            self.g_proj = Linear(config.hidden_size, self._heads * self._value)
        self.o_proj = LinearRowParallel(
            self._heads * self._value * self._tp.size,
            config.hidden_size,
            output_key="attention.output",
            tp=self._tp,
        )
        self._key_weight = self._value_weight = None
        self._decode_q_weight = self._decode_key_weight = None
        self._inference_prepared = False
        self._qkv_a = None
        projections = self._qkv_a_projections()
        if projections and self.q_a_proj.weight.is_meta:
            # Preserve the checkpoint names while declaring one stream-load group.
            MergedLinear.try_create(projections)

    def _qkv_a_projections(self):
        if self._config.q_lora_rank is None:
            return {}
        projections = {"q": self.q_a_proj, "kv": self.kv_a_proj_with_mqa}
        # Adapters and packed executors retain their own forward contracts.
        if any(
            type(op) is not Linear or op.weight_format != "dense" for op in projections.values()
        ):
            return {}
        return projections

    def after_load(self) -> None:
        self._qkv_a = MergedLinear.try_create(self._qkv_a_projections())
        self._key_weight = self._value_weight = None
        self._decode_q_weight = self._decode_key_weight = None
        self._inference_prepared = False
        if self.kv_b_proj.weight_format != "dense":
            return
        weight = self.kv_b_proj.weight
        if weight.is_meta or weight.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            return
        weights = weight.view(self._heads, self._nope + self._value, self._latent)
        key, value = weights.split((self._nope, self._value), dim=1)
        # Match the native batched-GEMM layouts. These local copies belong to
        # the loaded weights and are included in the pre-cache memory budget.
        self._key_weight = key.transpose(1, 2).contiguous().transpose(1, 2)
        self._value_weight = value.contiguous().transpose(1, 2)

    def prepare_inference(self) -> None:
        if self._inference_prepared:
            return
        if self._key_weight is not None:
            if self._cp.size == 1:
                self._decode_key_weight = self._key_weight
            else:
                query = self.q_proj if self._config.q_lora_rank is None else self.q_b_proj
                if (
                    query.weight_format == "dense"
                    and query.bias is None
                    and query.weight.dtype in (torch.float16, torch.bfloat16)
                    and not query.weight.is_meta
                ):
                    self._decode_q_weight = self._cp.gather_heads(query.weight, dim=0)
                self._decode_key_weight = self._cp.gather_heads(
                    self._key_weight.contiguous(), dim=0
                )
        self._inference_prepared = True

    def project(self, x, positions, *, is_decode=False):
        projected = None
        plan = self._qkv_a
        if plan is not None and x.dtype == plan.weight.dtype and x.device == plan.weight.device:
            # MLA QKV-A returns input dtype, unlike an FP32-accumulator MoE front.
            projected = plan.forward(x, output_dtype=x.dtype)
        if self._config.q_lora_rank is None:
            query, source = self.q_proj, x
        else:
            query = self.q_b_proj
            q_lora = self.q_a_proj.forward(x) if projected is None else projected["q"]
            source = self.q_a_layernorm.forward(q_lora)
        if is_decode and self._decode_q_weight is not None:
            q = linear_bf16(source, self._decode_q_weight)
            heads = self._decode_heads
        else:
            q = query.forward(source)
            heads = self._heads
        q = q.reshape(-1, heads, self._nope + self._rope)
        if is_decode and self._cp.size > 1 and self._decode_q_weight is None:
            # Quantized or other unsupported projection formats keep their
            # normal Linear executor; exchange activations, never reinterpret weights.
            q = self._cp.gather_heads(q, dim=1)
        kv = self.kv_a_proj_with_mqa.forward(x) if projected is None else projected["kv"]
        latent, key = kv.split((self._latent, self._rope), -1)
        latent = self.kv_a_layernorm.forward(latent)
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
        context_sharded=False,
    ):
        """Expanded fallback attends a complete history with TP-local heads."""
        if context_sharded and self._cp.size > 1:
            raise ValueError("Expanded MLA fallback requires a complete KV history")
        if q.shape[1] != self._heads:
            if q.shape[1] != self._decode_heads:
                raise ValueError("MLA query heads do not match the TP/DCP layout")
            q = q.chunk(self._cp.size, dim=1)[self._cp.rank]
        output = attention(
            q,
            latent_history,
            key_history,
            self.kv_b_proj.weight,
            positions,
            history_positions,
            nope_dim=self._nope,
            value_dim=self._value,
            return_lse=False,
            causal_layout=causal_layout,
        )
        return output if isinstance(output, MLAOutput) else MLAOutput(output)

    def attend_paged_partial(self, q, pages, metadata, *, attention, kv_write=None):
        weight = self._decode_key_weight
        if weight is None:
            if self._cp.size > 1 or self.kv_b_proj.weight_format != "dense":
                return None
            weight = self.kv_b_proj.weight.view(
                self._heads, self._nope + self._value, self._latent
            )[:, : self._nope]
        return attention.decode(
            q,
            pages,
            weight,
            metadata,
            nope_dim=self._nope,
            value_dim=self._value,
            **({"kv_write": kv_write} if kv_write is not None else {}),
        )

    def finish(self, x, partial: MLAOutput, *, residual=None):
        output = (
            self._cp.exchange(partial.values, partial.lse)
            if partial.lse is not None and self._cp.size > 1
            else partial.values
        )
        if isinstance(partial, MLALatentOutput):
            weight = self._value_weight
            if weight is None:
                weight = self.kv_b_proj.weight.view(
                    self._heads, self._nope + self._value, self._latent
                )[:, self._nope :].transpose(1, 2)
            output = headwise_linear(output.to(x.dtype), weight.transpose(1, 2))
        output = output.to(x.dtype).flatten(-2)
        if self._output_gate:
            output = sigmoid_mul(output, self.g_proj.forward(x))
        return self.o_proj.forward(output, residual=residual)

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
        residual=None,
    ):
        return self.finish(
            x,
            self.attend_partial(
                q, latent_history, key_history, positions, history_positions, attention=attention
            ),
            residual=residual,
        )

    def forward(self, x, positions, state: MLAState | None = None, *, residual=None):
        """Complete-history CPU/reference path; serving owns sharded pooled KV."""
        q, latent, key = self.project(x, positions)
        history_positions = positions
        if state is not None:
            latent = torch.cat((state.latent, latent))
            key = torch.cat((state.key, key))
            history_positions = torch.cat((state.positions, positions))
        if latent.shape[0] > self._config.max_position_embeddings:
            raise ValueError("MLA context exceeds max_position_embeddings")
        output = self.attend(x, q, latent, key, positions, history_positions, residual=residual)
        return output, MLAState(latent, key, history_positions)
