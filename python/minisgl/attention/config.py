"""Model-independent projection layouts. Layer indices belong to the model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KDAConfig:
    hidden_size: int
    num_heads: int
    head_dim: int
    conv_kernel_size: int
    rms_norm_eps: float = 1e-6
    full_rank_gate: bool = False
    gate_lower_bound: float | None = None

    def __post_init__(self):
        if min(self.hidden_size, self.num_heads, self.head_dim, self.conv_kernel_size) < 1:
            raise ValueError("KDA dimensions must be positive")
        if self.gate_lower_bound is not None and self.gate_lower_bound >= 0:
            raise ValueError("KDA gate_lower_bound must be negative")


@dataclass(frozen=True)
class MLAConfig:
    hidden_size: int
    num_heads: int
    nope_dim: int
    rope_dim: int
    value_dim: int
    latent_dim: int
    q_lora_rank: int | None
    max_position_embeddings: int
    rms_norm_eps: float = 1e-6
    use_rope: bool = False
    output_gate: bool = False
    rope_theta: float = 10000.0
    rope_interleave: bool = True

    def __post_init__(self):
        if (
            min(
                self.hidden_size,
                self.num_heads,
                self.nope_dim,
                self.value_dim,
                self.latent_dim,
                self.max_position_embeddings,
            )
            < 1
            or self.rope_dim < 0
        ):
            raise ValueError("Invalid MLA projection dimensions")
        if self.q_lora_rank is not None and self.q_lora_rank < 1:
            raise ValueError("MLA q_lora_rank must be positive or None")
        if self.use_rope and self.rope_dim % 2:
            raise ValueError("MLA rotary dimension must be even")
