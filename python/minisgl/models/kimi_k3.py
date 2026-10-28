"""Text-only, eager K3 with correctness-first tensor parallelism."""

from __future__ import annotations

import torch
from minisgl.core import get_global_ctx
from minisgl.distributed.layout import get_parallel_layout
from minisgl.distributed.parallel import TensorParallel
from minisgl.layers.attn_residual import attention_residual
from minisgl.layers.base import BaseOP, OPList
from minisgl.layers.embedding import Embedding
from minisgl.layers.kda import KDALayer
from minisgl.layers.latent_moe import GatedMLP, LatentMoE
from minisgl.layers.linear import Linear
from minisgl.layers.mla import MLALayer
from minisgl.layers.norm import TorchRMSNorm
from minisgl.models.kimi_config import kda_config, mla_config

from .base import BaseLLMModel
from .kimi_parallel import validate_tp_config


class KimiDecoderLayer(BaseOP):
    def __init__(self, config, layer_id):
        self._layer_id = layer_id
        self._is_kda = layer_id + 1 in config.linear_attn_config["kda_layers"]
        self.self_attn = (
            KDALayer(kda_config(config)) if self._is_kda else MLALayer(mla_config(config))
        )
        is_moe = (
            config.num_experts > 0
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        )
        self.mlp = LatentMoE(config) if is_moe else GatedMLP(config)
        self.input_layernorm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
        self._block_size = config.attn_res_block_size
        if self._block_size is not None:
            self.self_attention_res_norm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.self_attention_res_proj = Linear(config.hidden_size, 1)
            self.mlp_res_norm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.mlp_res_proj = Linear(config.hidden_size, 1)

    def _attention(self, x, batch, cache, page_table, backend=None):
        from minisgl.attention.recurrent import forward_layer

        run = backend.forward_layer if backend is not None else forward_layer
        return run(self.self_attn, self._layer_id, x, batch, cache, page_table)

    def forward(self, head, bank, batch, cache, page_table, backend=None):
        if self._block_size is None:
            x = self.input_layernorm.forward(head)
            head = head + self._attention(x, batch, cache, page_table, backend)
            return head + self.mlp.forward(self.post_attention_layernorm.forward(head))

        x = attention_residual(
            head, bank, self.self_attention_res_norm, self.self_attention_res_proj
        )
        x = self.input_layernorm.forward(x)
        # A boundary freezes the RAW sum, not the attention-weighted mixture.
        writes_block = self._layer_id % self._block_size == 0
        if writes_block:
            bank.append(head)
        out = self._attention(x, batch, cache, page_table, backend)
        head = out if writes_block else head + out
        x = attention_residual(head, bank, self.mlp_res_norm, self.mlp_res_proj)
        return head + self.mlp.forward(self.post_attention_layernorm.forward(x))


class KimiModel(BaseOP):
    def __init__(self, config):
        self.embed_tokens = Embedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([KimiDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
        self._attn_res = config.attn_res_block_size is not None
        if self._attn_res:
            self.output_attn_res_norm = TorchRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.output_attn_res_proj = Linear(config.hidden_size, 1)

    def forward(self, input_ids, batch, cache, page_table, backend=None):
        head, bank = self.embed_tokens.forward(input_ids), []
        for layer in self.layers.op_list:
            head = layer.forward(head, bank, batch, cache, page_table, backend)
        if self._attn_res:
            head = attention_residual(
                head, bank, self.output_attn_res_norm, self.output_attn_res_proj
            )
        return self.norm.forward(head)


class KimiK3LinearForCausalLM(BaseLLMModel):
    def __init__(self, config):
        text = config.hf_config
        self._tp = TensorParallel()
        layout = get_parallel_layout()
        validate_tp_config(text, self._tp.size, dcp_size=layout.dcp_size, ep_size=layout.ep_size)
        self._sharded_vocab = self._tp.size > 1 and text.vocab_size % self._tp.size == 0
        vocab_width = text.vocab_size // self._tp.size if self._sharded_vocab else text.vocab_size
        self.model = KimiModel(text)
        self.lm_head = Linear(text.hidden_size, vocab_width)
        self._tie_embeddings = text.tie_word_embeddings
        if self._tie_embeddings:
            raise ValueError("Kimi-K3 eager baseline requires untied embeddings")
        from .kimi_quant import replace_quantized_linears

        replace_quantized_linears(self, text)

    def forward(self):
        ctx = get_global_ctx()
        return self.forward_batch(ctx.batch, ctx.kv_cache, ctx.page_table, backend=ctx.attn_backend)

    def forward_batch(self, batch, cache, page_table, *, all_logits=False, backend=None):
        """Explicit context seam for CPU correctness tests, also used by serving."""
        if not batch.reqs:
            raise ValueError("Kimi-K3 requires a nonempty batch")
        real_tokens = sum(req.extend_len for req in batch.reqs)
        if len(batch.input_ids) < real_tokens or len(batch.positions) < real_tokens:
            raise ValueError("Batch tensors do not cover the real token rows")
        with cache.forward_batch(batch):
            hidden = self.model.forward(
                batch.input_ids[:real_tokens], batch, cache, page_table, backend
            )
            if not all_logits:
                last = torch.tensor([req.extend_len for req in batch.reqs], device=hidden.device)
                hidden = hidden[last.cumsum(0) - 1]
            logits = self.lm_head.forward(hidden).float()
            if self._sharded_vocab:
                logits = self._tp.all_gather(logits.T.contiguous()).T.contiguous()
            return logits
