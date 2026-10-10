"""Optional independent hybrid math oracle, not an official GPU-kernel run.

Uses unchanged hash-pinned HF decoder/KDA definitions and FLA naive recurrence.
Three GPU kernel boundaries are explicitly adapted with independent PyTorch math:
causal convolution, fused gated norm, and raw-gate/QK normalization. No mini
numerical operation is used by the reference. Cache kernel layout is adapted,
so this establishes tiny hybrid wiring/math parity, not GPU layout parity.
"""

import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from minisgl.kvcache.kimi_pool import KimiHybridCache
from minisgl.models.config import ModelConfig
from minisgl.models.kimi_k3 import KimiK3LinearForCausalLM
from test_kimi_official_reference import (
    assert_close,
    causal_mask,
    config,
    initialize,
    official,  # noqa: F401 - shared hash-checking pytest fixture
    official_config,
    official_fla,  # noqa: F401 - shared hash-checking pytest fixture
)


class CPUShortConvolution(torch.nn.Module):
    def __init__(self, hidden_size, kernel_size, activation):
        super().__init__()
        assert activation == "silu"
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, 1, kernel_size))
        self.width = kernel_size

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None):
        assert cu_seqlens is None
        values = x.transpose(1, 2)
        if cache is None:
            cache = values.new_zeros(values.shape[0], values.shape[1], self.width - 1)
        history = torch.cat((cache, values), -1)
        out = F.conv1d(history.float(), self.weight.float(), groups=values.shape[1])
        state = history[..., -(self.width - 1) :].clone() if self.width > 1 else history[..., :0]
        return F.silu(out).transpose(1, 2).to(x.dtype), state if output_final_state else None


class CPUGatedNorm(torch.nn.Module):
    def __init__(self, size, eps, activation):
        super().__init__()
        assert activation == "sigmoid"
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x, gate):
        values = x.float()
        norm = values / torch.sqrt(torch.mean(values**2, dim=-1, keepdim=True) + self.eps)
        return (norm * self.weight.float() * torch.sigmoid(gate.float())).to(x.dtype)


@pytest.fixture
def hybrid_reference(official, official_fla):  # noqa: F811 - pytest fixture injection
    # The shared fixture has already verified the whole file's pinned hash.
    namespace = dict(official.KimiDecoderLayer.__init__.__globals__)
    source = ast.parse(Path(os.environ["KIMI_OFFICIAL_REFERENCE_PATH"]).read_bytes())
    names = {"KimiDeltaAttention", "KimiDecoderLayer"}
    nodes = [n for n in source.body if isinstance(n, ast.ClassDef) and n.name in names]
    assert {n.name for n in nodes} == names

    def kda(
        *,
        q,
        k,
        v,
        g,
        beta,
        A_log,
        dt_bias,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        transpose_state_layout,
        cu_seqlens,
        lower_bound=None,
        **kwargs,
    ):
        assert use_qk_l2norm_in_kernel and use_gate_in_kernel and use_beta_sigmoid_in_kernel
        assert transpose_state_layout and cu_seqlens is None
        q, k = [
            value.float() / torch.sqrt((value.float() ** 2).sum(-1, keepdim=True) + 1e-6)
            for value in (q, k)
        ]
        raw = g.float() + dt_bias.float().reshape(g.shape[-2:])
        scale = torch.exp(A_log.float())[..., None]
        decay = (
            -scale * F.softplus(raw)
            if lower_bound is None
            else lower_bound / (1 + torch.exp(-scale * raw))
        )
        out, state = official_fla["naive_recurrent_kda"](
            q,
            k,
            v,
            decay,
            torch.sigmoid(beta.float()),
            initial_state=None if initial_state is None else initial_state.transpose(-1, -2),
            output_final_state=output_final_state,
        )
        return out, None if state is None else state.transpose(-1, -2)

    namespace.update(
        ShortConvolution=CPUShortConvolution,
        FusedRMSNormGated=CPUGatedNorm,
        chunk_kda=kda,
        fused_recurrent_kda=kda,
        rearrange=pytest.importorskip("einops").rearrange,
    )
    module = ast.Module(
        body=[ast.parse("from __future__ import annotations").body[0], *nodes], type_ignores=[]
    )
    exec(compile(module, "pinned_hf_hybrid_with_cpu_kernel_adapters", "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("full_gate", [False, True])
@pytest.mark.parametrize("lower_bound", [None, -5.0])
@pytest.mark.parametrize("block_size", [1, 2])
def test_hybrid_layers_logits_and_greedy_steps(
    hybrid_reference, full_gate, lower_bound, block_size
):
    linear = {
        "num_heads": 2,
        "head_dim": 4,
        "short_conv_kernel_size": 3,
        "kda_layers": [1, 3],
        "full_attn_layers": [2, 4],
        "use_full_rank_gate": full_gate,
        "gate_lower_bound": lower_bound,
    }
    c = config(linear_attn_config=linear, attn_res_block_size=block_size)
    model = KimiK3LinearForCausalLM(ModelConfig.from_hf(c))
    initialize(model)
    rc = official_config(c)
    rc.is_kda_layer = lambda i: i + 1 in linear["kda_layers"]
    refs = []
    for i, layer in enumerate(model.model.layers.op_list):
        ref = hybrid_reference.KimiDecoderLayer(rc, i)
        weights = {
            name.replace("mlp.", "block_sparse_moe.", 1)
            if name.startswith("mlp.") and hasattr(ref, "block_sparse_moe")
            else name: value.clone()
            for name, value in layer.state_dict().items()
        }
        ref.load_state_dict(weights, strict=True)
        refs.append(ref.eval())
    ref_cache = hybrid_reference.KimiDynamicCache(rc)
    cache = KimiHybridCache(c, 32, 1, 1, torch.float32, torch.device("cpu"))
    page_table = torch.arange(32)[None]
    final_norm = hybrid_reference.KimiRMSNorm(c.hidden_size, c.rms_norm_eps)
    final_norm.load_state_dict({"weight": model.model.norm.weight.clone()})
    res_norm = hybrid_reference.KimiRMSNorm(c.hidden_size, c.rms_norm_eps)
    res_norm.load_state_dict({"weight": model.model.output_attn_res_norm.weight.clone()})
    res_proj = torch.nn.Linear(c.hidden_size, 1, bias=False)
    res_proj.load_state_dict({"weight": model.model.output_attn_res_proj.weight.clone()})
    tokens = [2, 4, 8, 3, 9]
    for start, end in [(0, 3), (3, 5), (5, 6), (6, 7), (7, 8)]:
        current = torch.tensor(tokens[start:end])
        req = SimpleNamespace(
            uid=10, table_idx=0, cached_len=start, device_len=end, extend_len=end - start
        )
        batch = SimpleNamespace(
            reqs=[req],
            input_ids=current,
            positions=torch.arange(start, end),
            out_loc=torch.arange(start, end),
        )
        head = F.embedding(current, model.model.embed_tokens.weight)
        expected = head[None].clone()
        bank, ref_bank = [], head.new_empty(end - start, 0, c.hidden_size)
        cache.begin_request(0, 10, start)
        for layer, ref in zip(model.model.layers.op_list, refs):
            head = layer.forward(head, bank, batch, cache, page_table)
            expected, ref_bank = ref(
                expected,
                attention_mask=causal_mask(start, end, torch.float32),
                past_key_values=ref_cache,
                block_residual=ref_bank,
            )
            assert_close(head, expected[0], stack=True)
            assert_close(torch.stack(bank, 1), ref_bank, stack=True)
        cache.finish_request(0, 10, end)
        reference_hidden = hybrid_reference._apply_attn_res(
            expected[0], ref_bank, res_proj, res_norm
        )
        reference_logits = F.linear(final_norm(reference_hidden), model.lm_head.weight)
        # Independently evaluate the complete production path on a separate fresh
        # cache/prefix, so final normalization/projection wiring is covered too.
        fresh = KimiHybridCache(c, 32, 1, 1, torch.float32, torch.device("cpu"))
        whole_req = SimpleNamespace(
            uid=11, table_idx=0, cached_len=0, device_len=end, extend_len=end
        )
        whole = SimpleNamespace(
            reqs=[whole_req],
            input_ids=torch.tensor(tokens[:end]),
            positions=torch.arange(end),
            out_loc=torch.arange(end),
        )
        logits = model.forward_batch(whole, fresh, page_table, all_logits=True)[start:end]
        assert_close(logits, reference_logits, stack=True)
        assert logits[-1].argmax().item() == reference_logits[-1].argmax().item()
        if end >= 5:
            tokens.append(reference_logits[-1].argmax().item())
