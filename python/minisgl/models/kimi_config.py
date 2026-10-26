"""Local, data-only K3 config; loading it never executes checkpoint Python."""

from transformers import PretrainedConfig


class KimiTextConfig(PretrainedConfig):
    model_type = "kimi_linear"

    def __init__(self, **kwargs):
        defaults = {
            "hidden_act": "situ",
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "rope_scaling": None,
            "tie_word_embeddings": False,
            "num_experts": 0,
            "num_experts_per_token": 0,
            "num_shared_experts": 0,
            "first_k_dense_replace": 0,
            "moe_layer_freq": 1,
            "moe_renormalize": True,
            "moe_router_activation_func": "sigmoid",
            "num_expert_group": 1,
            "topk_group": 1,
            "use_grouped_topk": True,
            "topk_method": "noaux_tc",
            "routed_scaling_factor": 1.0,
            "routed_expert_hidden_size": None,
            "latent_moe_use_norm": False,
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "attn_res_block_size": None,
            "mla_use_output_gate": True,
            "mla_use_nope": True,
            "num_nextn_predict_layers": 0,
            "architectures": ["KimiK3LinearForCausalLM"],
        }
        defaults.update(kwargs)
        super().__init__(**defaults)

    def validate_text(self):
        positive = (
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "max_position_embeddings",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_nope_head_dim",
            "qk_rope_head_dim",
            "v_head_dim",
        )
        for name in positive:
            if not isinstance(getattr(self, name, None), int) or getattr(self, name) <= 0:
                raise ValueError(f"Kimi-K3 requires positive {name}")
        linear = getattr(self, "linear_attn_config", None)
        if not isinstance(linear, dict):
            raise ValueError("Kimi-K3 requires linear_attn_config")
        for name in ("num_heads", "head_dim", "short_conv_kernel_size"):
            if not isinstance(linear.get(name), int) or linear[name] <= 0:
                raise ValueError(f"Kimi-K3 requires positive linear_attn_config.{name}")
        kda, mla = linear.get("kda_layers", []), linear.get("full_attn_layers", [])
        if not mla:
            raise ValueError("Kimi-K3 baseline requires at least one MLA layer")
        expected = list(range(1, self.num_hidden_layers + 1))
        if sorted(kda + mla) != expected:
            raise ValueError(
                "KDA/MLA layer lists must partition all layers exactly once (one-based)"
            )
        if self.attn_res_block_size is not None and self.attn_res_block_size <= 0:
            raise ValueError("attn_res_block_size must be positive")
        if self.num_nextn_predict_layers:
            raise ValueError("Kimi-K3 next-token prediction layers are not supported")
        if self.num_experts:
            if not 0 < self.num_experts_per_token <= self.num_experts:
                raise ValueError("Invalid Kimi-K3 expert top-k")
            if not getattr(self, "moe_intermediate_size", 0):
                raise ValueError("moe_intermediate_size is required")


class KimiK3Config(PretrainedConfig):
    model_type = "kimi_k3"

    def __init__(self, text_config=None, **kwargs):
        super().__init__(**kwargs)
        self.text_config = (
            text_config
            if isinstance(text_config, KimiTextConfig)
            else KimiTextConfig(**(text_config or {}))
        )


def kda_config(config):
    from minisgl.attention.config import KDAConfig

    c = config.linear_attn_config
    return KDAConfig(
        config.hidden_size,
        c["num_heads"],
        c["head_dim"],
        c["short_conv_kernel_size"],
        config.rms_norm_eps,
        c.get("use_full_rank_gate", False),
        c.get("gate_lower_bound"),
    )


def mla_config(config):
    from minisgl.attention.config import MLAConfig

    if not getattr(config, "mla_use_nope", True) and getattr(config, "rope_scaling", None):
        raise ValueError("MLA supports NoPE or unscaled RoPE only")
    return MLAConfig(
        config.hidden_size,
        config.num_attention_heads,
        config.qk_nope_head_dim,
        config.qk_rope_head_dim,
        config.v_head_dim,
        config.kv_lora_rank,
        config.q_lora_rank,
        config.max_position_embeddings,
        config.rms_norm_eps,
        not getattr(config, "mla_use_nope", True),
        getattr(config, "mla_use_output_gate", False),
        getattr(config, "rope_theta", 10000.0),
        getattr(config, "rope_interleave", True),
    )


def cache_config(config):
    from minisgl.kvcache.config import HybridCacheConfig

    c = config.linear_attn_config
    return HybridCacheConfig(
        config.num_hidden_layers,
        tuple(i - 1 for i in c["full_attn_layers"]),
        tuple(i - 1 for i in c["kda_layers"]),
        config.kv_lora_rank,
        config.qk_rope_head_dim,
        c["num_heads"],
        c["head_dim"],
        c["short_conv_kernel_size"],
    )


def cache_bytes_per_token(config, dtype):
    return cache_config(config).cache_bytes_per_token(dtype)


def state_bytes_per_request(config, dtype, tp_size=None):
    return cache_config(config).state_bytes_per_request(dtype, tp_size)
