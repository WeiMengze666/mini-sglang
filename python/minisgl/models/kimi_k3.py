"""Text-only, eager K3 with correctness-first tensor parallelism."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.layers.base import BaseOP, OPList

from .base import BaseLLMModel
from .kimi_attention_ops import KimiKDA, KimiMLA
from .kimi_moe import KimiK3MLP, KimiK3MoE, KimiLinear, KimiRMSNorm
from .kimi_parallel import KimiTP, validate_tp_config


def attention_residual(head, bank, norm, proj):
    """Score frozen block sums and the current block independently per token."""
    if not bank:
        return head
    rows = torch.stack([*bank, head], dim=1)
    # Keep the score and mixture reductions in FP32.
    xf = rows.float()
    normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + norm.eps)
    scores = F.linear(normalized * norm.weight.float(), proj.weight.float())
    weights = scores.softmax(dim=1)
    return (weights * xf).sum(dim=1).to(head.dtype)


class _Embedding(BaseOP):
    def __init__(self, vocab_size, hidden_size):
        self._tp = KimiTP()
        self._vocab_size = vocab_size
        self._sharded = self._tp.size > 1 and vocab_size % self._tp.size == 0
        width = vocab_size // self._tp.size if self._sharded else vocab_size
        self._start = self._tp.rank * width if self._sharded else 0
        self.weight = torch.empty(width, hidden_size)

    def forward(self, input_ids):
        if not self._sharded:
            return F.embedding(input_ids.long(), self.weight)
        if ((input_ids < 0) | (input_ids >= self._vocab_size)).any():
            raise IndexError("Kimi token ID is outside the vocabulary")
        local = input_ids.long() - self._start
        mask = (local >= 0) & (local < self.weight.shape[0])
        output = F.embedding(local.masked_fill(~mask, 0), self.weight)
        output = output * mask.unsqueeze(-1).to(output.dtype)
        return self._tp.all_reduce(output)


class KimiDecoderLayer(BaseOP):
    def __init__(self, config, layer_id):
        self._layer_id = layer_id
        self._is_kda = layer_id + 1 in config.linear_attn_config["kda_layers"]
        self.self_attn = KimiKDA(config) if self._is_kda else KimiMLA(config)
        is_moe = (
            config.num_experts > 0
            and layer_id >= config.first_k_dense_replace
            and layer_id % config.moe_layer_freq == 0
        )
        self.mlp = KimiK3MoE(config) if is_moe else KimiK3MLP(config)
        self.input_layernorm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self._block_size = config.attn_res_block_size
        if self._block_size is not None:
            self.self_attention_res_norm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.self_attention_res_proj = KimiLinear(config.hidden_size, 1)
            self.mlp_res_norm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.mlp_res_proj = KimiLinear(config.hidden_size, 1)

    def _attention(self, x, batch, cache, page_table):
        outputs, offset = [], 0
        for req in batch.reqs:
            rows = slice(offset, offset + req.extend_len)
            current, positions = x[rows], batch.positions[rows]
            if self._is_kda:
                state = cache.get_kda_state(self._layer_id, req.table_idx)
                out, state = self.self_attn.forward(current, positions, state)
                cache.set_kda_state(self._layer_id, req.table_idx, state)
            else:
                q, latent, key = self.self_attn.project(current, positions)
                cache.store_mla(self._layer_id, latent, key, batch.out_loc[rows])
                latent, key = cache.gather_mla(
                    self._layer_id, page_table[req.table_idx, : req.device_len]
                )
                history_positions = torch.arange(req.device_len, device=x.device)
                out = self.self_attn.attend(current, q, latent, key, positions, history_positions)
            outputs.append(out)
            offset += req.extend_len
        return torch.cat(outputs, dim=0)

    def forward(self, head, bank, batch, cache, page_table):
        if self._block_size is None:
            x = self.input_layernorm.forward(head)
            head = head + self._attention(x, batch, cache, page_table)
            return head + self.mlp.forward(self.post_attention_layernorm.forward(head))

        x = attention_residual(
            head, bank, self.self_attention_res_norm, self.self_attention_res_proj
        )
        x = self.input_layernorm.forward(x)
        # A boundary freezes the RAW sum, not the attention-weighted mixture.
        writes_block = self._layer_id % self._block_size == 0
        if writes_block:
            bank.append(head)
        out = self._attention(x, batch, cache, page_table)
        head = out if writes_block else head + out
        x = attention_residual(head, bank, self.mlp_res_norm, self.mlp_res_proj)
        return head + self.mlp.forward(self.post_attention_layernorm.forward(x))


class KimiModel(BaseOP):
    def __init__(self, config):
        self.embed_tokens = _Embedding(config.vocab_size, config.hidden_size)
        self.layers = OPList([KimiDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
        self._attn_res = config.attn_res_block_size is not None
        if self._attn_res:
            self.output_attn_res_norm = KimiRMSNorm(config.hidden_size, config.rms_norm_eps)
            self.output_attn_res_proj = KimiLinear(config.hidden_size, 1)

    def forward(self, input_ids, batch, cache, page_table):
        head, bank = self.embed_tokens.forward(input_ids), []
        for layer in self.layers.op_list:
            head = layer.forward(head, bank, batch, cache, page_table)
        if self._attn_res:
            head = attention_residual(
                head, bank, self.output_attn_res_norm, self.output_attn_res_proj
            )
        return self.norm.forward(head)


class KimiK3LinearForCausalLM(BaseLLMModel):
    def __init__(self, config):
        text = config.kimi
        self._tp = KimiTP()
        validate_tp_config(text, self._tp.size)
        self._sharded_vocab = self._tp.size > 1 and text.vocab_size % self._tp.size == 0
        vocab_width = text.vocab_size // self._tp.size if self._sharded_vocab else text.vocab_size
        self.model = KimiModel(text)
        self.lm_head = KimiLinear(text.hidden_size, vocab_width)
        self._tie_embeddings = text.tie_word_embeddings
        if self._tie_embeddings:
            raise ValueError("Kimi-K3 eager baseline requires untied embeddings")
        from .kimi_quant import replace_quantized_linears

        replace_quantized_linears(self, text)

    def forward(self):
        ctx = get_global_ctx()
        return self.forward_batch(ctx.batch, ctx.kv_cache, ctx.page_table)

    def forward_batch(self, batch, cache, page_table, *, all_logits=False):
        """Explicit context seam for CPU correctness tests, also used by serving."""
        if not batch.reqs:
            raise ValueError("Kimi-K3 requires a nonempty batch")
        real_tokens = sum(req.extend_len for req in batch.reqs)
        if len(batch.input_ids) < real_tokens or len(batch.positions) < real_tokens:
            raise ValueError("Batch tensors do not cover the real token rows")
        started = []
        try:
            for req in batch.reqs:
                cache.begin_request(req.table_idx, req.uid, req.cached_len)
                started.append(req)
            hidden = self.model.forward(batch.input_ids[:real_tokens], batch, cache, page_table)
            if not all_logits:
                last = torch.tensor([req.extend_len for req in batch.reqs], device=hidden.device)
                hidden = hidden[last.cumsum(0) - 1]
            logits = self.lm_head.forward(hidden).float()
            if self._sharded_vocab:
                logits = self._tp.all_gather(logits.T.contiguous()).T.contiguous()
            for req in batch.reqs:
                cache.finish_request(req.table_idx, req.uid, req.device_len)
            return logits
        except Exception:
            # A partial recurrent update cannot safely be replayed as a prefix.
            for req in started:
                cache.release_request(req.table_idx, req.uid)
            raise
