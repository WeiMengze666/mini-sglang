"""Small, checkpoint-name-based tensor parallel contract for eager Kimi.

Hidden states, routers and low-rank projections remain replicated. DCP widens
MLA query/KV expansion heads within each group; EP assigns global expert IDs and
splits each expert only over its remaining tensor-parallel ranks.
"""

from __future__ import annotations

import re

from minisgl.distributed.layout import ParallelLayout
from minisgl.distributed.parallel import local_size
from minisgl.distributed.partition import ShardSpec


def _text_config(config):
    return getattr(config, "hf_config", None) or getattr(config, "text_config", config)


def validate_tp_config(config, tp_size: int, *, dcp_size=1, ep_size=1) -> None:
    """Fail before allocation/collectives for unsupported uneven head/FFN splits."""
    c = _text_config(config)
    layout = ParallelLayout(0, tp_size, dcp_size, ep_size)
    if tp_size < 1:
        raise ValueError("Kimi TP size must be positive")
    linear = c.linear_attn_config
    if linear["kda_layers"]:
        local_size(linear["num_heads"], tp_size, "KDA heads")
    if linear["full_attn_layers"]:
        local_size(c.num_attention_heads, tp_size, "MLA heads")
    moe_layers = [
        i
        for i in range(c.num_hidden_layers)
        if c.num_experts > 0 and i >= c.first_k_dense_replace and i % c.moe_layer_freq == 0
    ]
    if len(moe_layers) < c.num_hidden_layers:
        local_size(c.intermediate_size, tp_size, "dense intermediate_size")
    if moe_layers:
        local_size(c.num_experts, ep_size, "experts")
        local_size(c.moe_intermediate_size, layout.expert_tensor_size, "moe_intermediate_size")
        shared = getattr(c, "num_shared_experts", 0) or 0
        if shared:
            local_size(c.moe_intermediate_size * shared, tp_size, "shared intermediate_size")


def shard_spec(
    name: str, config, tp_rank: int, tp_size: int, *, dcp_size=1, ep_size=1
) -> ShardSpec | None:
    """Map a canonical model parameter name to its pre-decode checkpoint slice.

    Call this with the logical ``.weight`` name for packed weight/scale tensors.
    Unrecognized names and uneven vocabulary rows are explicitly replicated.
    Checkpoint canonicalization/unknown-key validation belongs to the loader.
    """
    if not 0 <= tp_rank < tp_size:
        raise ValueError("Kimi TP rank must be in [0, tp_size)")
    if tp_size == 1:
        return None
    c = _text_config(config)
    layout = ParallelLayout(tp_rank, tp_size, dcp_size, ep_size)

    def split(axis, size, label, block=1, *, partition=layout.tensor):
        return partition.shard(axis, size, label, block)

    if name in ("model.embed_tokens.weight", "lm_head.weight"):
        # Keep the rare uneven vocabulary case functional without padded weights.
        return split(0, c.vocab_size, "vocabulary") if c.vocab_size % tp_size == 0 else None
    match = re.fullmatch(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(.+)", name)
    if match is None:
        return None
    layer, module, parameter = int(match[1]), match[2], match[3]
    if module == "self_attn":
        linear = c.linear_attn_config
        if layer + 1 in linear["kda_layers"]:
            heads, dim = linear["num_heads"], linear["head_dim"]
            if parameter in ("b_proj.weight", "A_log"):
                return split(0, heads, "KDA heads")
            if parameter in (
                "q_proj.weight",
                "k_proj.weight",
                "v_proj.weight",
                "f_b_proj.weight",
                "g_proj.weight",
                "g_b_proj.weight",
                "q_conv1d.weight",
                "k_conv1d.weight",
                "v_conv1d.weight",
                "dt_bias",
            ):
                return split(0, heads, "KDA heads", dim)
            if parameter == "o_proj.weight":
                return split(1, heads, "KDA heads", dim)
        else:
            heads = c.num_attention_heads
            attention = {"partition": layout.attention}
            if parameter in ("q_proj.weight", "q_b_proj.weight"):
                return split(
                    0, heads, "MLA heads", c.qk_nope_head_dim + c.qk_rope_head_dim, **attention
                )
            if parameter == "kv_b_proj.weight":
                return split(0, heads, "MLA heads", c.qk_nope_head_dim + c.v_head_dim, **attention)
            if parameter == "g_proj.weight":
                return split(0, heads, "MLA heads", c.v_head_dim)
            if parameter == "o_proj.weight":
                return split(1, heads, "MLA heads", c.v_head_dim)
        return None
    expert = re.fullmatch(r"experts\.\d+\.(w[123])\.weight", parameter)
    if expert is not None:
        return split(
            1 if expert[1] == "w2" else 0,
            c.moe_intermediate_size,
            "expert intermediate_size",
            partition=layout.expert_tensor,
        )
    shared = parameter.startswith("shared_experts.")
    if shared:
        parameter = parameter.removeprefix("shared_experts.")
    if parameter in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
        width = c.moe_intermediate_size * c.num_shared_experts if shared else c.intermediate_size
        return split(1 if parameter == "down_proj.weight" else 0, width, "MLP intermediate_size")
    return None


def owns_expert(name: str, config, layout: ParallelLayout) -> bool:
    """Select checkpoint expert IDs before reading any dense or packed payload."""
    if layout.ep_size == 1:
        return True
    match = re.search(r"\.mlp\.experts\.(\d+)\.", name)
    if match is None:
        return True
    count = _text_config(config).num_experts
    if not 0 <= int(match[1]) < count:
        raise ValueError("Checkpoint expert ID exceeds configured expert count")
    return int(match[1]) in layout.expert_range(count)
