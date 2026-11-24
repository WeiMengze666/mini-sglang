"""K3 text decoder assembled from reusable attention, residual and MoE layers."""

from __future__ import annotations

import torch
from minisgl.attention.config import KDAConfig, MLAConfig
from minisgl.core import get_global_ctx
from minisgl.distributed.layout import get_parallel_layout
from minisgl.distributed.parallel import TensorParallel
from minisgl.layers.attn_residual import ResidualBank, attention_residual
from minisgl.layers.base import BaseOP, OPList
from minisgl.layers.embedding import Embedding
from minisgl.layers.kda import KDALayer
from minisgl.layers.latent_moe import GatedMLP, LatentMoE
from minisgl.layers.linear import Linear
from minisgl.layers.mla import MLALayer
from minisgl.layers.norm import TorchRMSNorm

from ..base import BaseLLMModel
from .checkpoint import KimiCheckpoint, configure_linears
from .config import is_moe_layer, kda_config, kda_layer_ids, mla_config, validate_partitions


class KimiDecoderLayer(BaseOP):
    def __init__(self, config, layer_id, attention_config):
        self._layer_id = layer_id
        self._is_kda = isinstance(attention_config, KDAConfig)
        self.self_attn = KDALayer(attention_config) if self._is_kda else MLALayer(attention_config)
        is_moe = is_moe_layer(config, layer_id)
        self.mlp = LatentMoE(config) if is_moe else GatedMLP(config)
        self.input_layernorm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
        self._block_size = config.attn_res_block_size
        if self._block_size is not None:
            self.self_attention_res_norm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.self_attention_res_proj = Linear(config.hidden_size, 1)
            self.mlp_res_norm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.mlp_res_proj = Linear(config.hidden_size, 1)

    def _attention(self, x, batch, cache, page_table, backend=None, *, residual=None):
        return backend.forward_layer(
            self.self_attn, self._layer_id, x, batch, cache, page_table, residual=residual
        )

    def forward(self, head, bank, batch, cache, page_table, backend=None):
        if self._block_size is None:
            x = self.input_layernorm.forward(head)
            head = self._attention(x, batch, cache, page_table, backend, residual=head)
            return self.mlp.forward(self.post_attention_layernorm.forward(head), residual=head)

        x = attention_residual(
            head,
            bank,
            self.self_attention_res_norm,
            self.self_attention_res_proj,
            self.input_layernorm,
        )
        # A boundary freezes the RAW sum, not the attention-weighted mixture.
        writes_block = self._layer_id % self._block_size == 0
        if writes_block:
            bank.append(head)
        head = self._attention(
            x, batch, cache, page_table, backend, residual=None if writes_block else head
        )
        x = attention_residual(
            head, bank, self.mlp_res_norm, self.mlp_res_proj, self.post_attention_layernorm
        )
        return self.mlp.forward(x, residual=head)


class KimiModel(BaseOP):
    def __init__(self, config, attention_configs=()):
        if not isinstance(attention_configs, tuple) or any(
            not isinstance(record, tuple) or len(record) != 2 for record in attention_configs
        ):
            raise ValueError("Attention records must be immutable (global ID, config) pairs")
        declared = dict(attention_configs)
        if attention_configs and (
            len(declared) != len(attention_configs)
            or any(not isinstance(c, (KDAConfig, MLAConfig)) for c in declared.values())
            or any(type(i) is not int for i in declared)
            or set(declared) != set(range(config.num_hidden_layers))
        ):
            raise ValueError("Attention records must cover every decoder layer exactly once")
        kda_ids = frozenset(kda_layer_ids(config))
        if not declared:
            kda, mla = kda_config(config), mla_config(config)
            declared = {i: kda if i in kda_ids else mla for i in range(config.num_hidden_layers)}
        elif any(isinstance(spec, KDAConfig) != (i in kda_ids) for i, spec in declared.items()):
            raise ValueError("Declared attention configuration differs from the decoder layer")
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [KimiDecoderLayer(config, i, declared[i]) for i in range(config.num_hidden_layers)]
        )
        self.norm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
        self._attn_res = config.attn_res_block_size is not None
        self._residual_blocks = (
            (config.num_hidden_layers + config.attn_res_block_size - 1)
            // config.attn_res_block_size
            if self._attn_res
            else 0
        )
        if self._attn_res:
            self.output_attn_res_norm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.output_attn_res_proj = Linear(config.hidden_size, 1)

    def forward(self, input_ids, batch, cache, page_table, backend=None):
        head = self.embed_tokens.forward(input_ids)
        bank = ResidualBank(head, self._residual_blocks) if self._attn_res else []
        for layer in self.layers.op_list:
            head = layer.forward(head, bank, batch, cache, page_table, backend)
        if self._attn_res:
            head = attention_residual(
                head, bank, self.output_attn_res_norm, self.output_attn_res_proj, self.norm
            )
            return head
        return self.norm.forward(head)


class KimiK3LinearForCausalLM(BaseLLMModel):
    checkpoint_type = KimiCheckpoint

    def __init__(self, config):
        self._config = config
        text = config.hf_config
        self._tp = TensorParallel()
        layout = get_parallel_layout()
        validate_partitions(text, layout)
        vocab_shard = Embedding.weight_shard(layout.tensor, text.vocab_size)
        self._sharded_vocab = vocab_shard is not None
        vocab_width = vocab_shard.length if vocab_shard else text.vocab_size
        self.model = KimiModel(text, config.attention_configs)
        self.lm_head = Linear(text.hidden_size, vocab_width)
        self._tie_embeddings = text.tie_word_embeddings
        if self._tie_embeddings:
            raise ValueError("Kimi-K3 text decoder requires untied embeddings")
        configure_linears(self, text)

    def forward(self):
        ctx = get_global_ctx()
        return self._forward_batch(
            ctx.batch, ctx.kv_cache, ctx.page_table, backend=ctx.attn_backend
        )

    def forward_batch(self, batch, cache, page_table, *, all_logits=False, backend=None):
        """Explicit owned-state seam for direct/reference execution."""
        with cache.forward_batch(batch):
            return self._forward_batch(
                batch, cache, page_table, all_logits=all_logits, backend=backend
            )

    def _forward_batch(self, batch, cache, page_table, *, all_logits=False, backend=None):
        if not batch.reqs:
            raise ValueError("Kimi-K3 requires a nonempty batch")
        real_tokens = sum(req.extend_len for req in batch.reqs)
        if len(batch.input_ids) < real_tokens or len(batch.positions) < real_tokens:
            raise ValueError("Batch tensors do not cover the real token rows")
        if backend is None:
            from minisgl.attention import create_torch_backend

            backend = create_torch_backend(self._config, cache=cache)
            backend.prepare_metadata(batch)
        hidden = self.model.forward(
            batch.input_ids[:real_tokens], batch, cache, page_table, backend
        )
        if not all_logits:
            metadata = getattr(batch, "attn_metadata", None)
            if metadata is None:
                lengths = torch.tensor([req.extend_len for req in batch.reqs], device=hidden.device)
                last = lengths.cumsum(0) - 1
            else:
                last = metadata.get_last_indices(len(batch.reqs))
            hidden = hidden[last]
        logits = self.lm_head.forward(hidden)
        if self._sharded_vocab:
            logits = self._tp.all_gather(logits, dim=-1)
        return logits.float()
