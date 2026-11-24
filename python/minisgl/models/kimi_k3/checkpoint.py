"""Map the released text checkpoint to Mini layers and their logical weight slices."""

from __future__ import annotations

import re
from functools import cached_property

from minisgl.checkpoint.loader import Checkpoint
from minisgl.layers.embedding import Embedding
from minisgl.layers.kda import KDALayer
from minisgl.layers.latent_moe import GatedExpert, GatedMLP
from minisgl.layers.mla import MLALayer
from minisgl.layers.weight import quantization_rules, replace_quantized_linears

from .config import KimiTextConfig, kda_layer_ids, validate_partitions

_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def text_weight_name(name: str, num_layers: int | None = None) -> str | None:
    """Keep native text names; exclude vision, projector and prediction heads."""
    if name.startswith("language_model.layers."):
        name = "language_model.model.layers." + name.removeprefix("language_model.layers.")
    name = name.removeprefix("language_model.").replace(".block_sparse_moe.", ".mlp.")
    components = name.split(".")
    if any(
        part
        in {
            "vision_tower",
            "mm_projector",
            "multi_modal_projector",
            "vision_model",
            "mtp",
            "mtp_layers",
            "nextn_predict_layers",
        }
        for part in components
    ):
        return None
    if any(part in {"inv_freq", "cos_cached", "sin_cached"} for part in components):
        return None
    layer = _LAYER.search(name)
    if layer is not None and num_layers is not None and int(layer.group(1)) >= num_layers:
        return None
    return name


# These checkpoint modules use dense nn.Linear rather than the quantized class.
_NATIVE_DENSE = {
    "routed_expert_down_proj",
    "routed_expert_up_proj",
    "self_attention_res_proj",
    "mlp_res_proj",
    "output_attn_res_proj",
}


def _aliases(name):
    names = (name, "language_model." + name)
    return (*names, *(alias.replace(".mlp.", ".block_sparse_moe.") for alias in names))


def configure_linears(model, config, *, chunk_rows=128):
    return replace_quantized_linears(
        model,
        getattr(config, "quantization_config", None),
        chunk_rows=chunk_rows,
        aliases=_aliases,
        linear_types=("Linear", "KimiLinear"),
        exclude=lambda name, op: name.rsplit(".", 1)[-1] in _NATIVE_DENSE,
    )


class KimiCheckpoint(Checkpoint):
    @staticmethod
    def from_metadata(metadata):
        return KimiTextConfig(**metadata)

    def __init__(self, config, layout):
        config = getattr(config, "hf_config", None) or config
        super().__init__(config, layout)

    @property
    def packed_weights(self):
        return bool(getattr(self.config, "quantization_config", None))

    def validate(self):
        quantization_rules(getattr(self.config, "quantization_config", None))
        if self.layout.tensor.size > 1:
            validate_partitions(self.config, self.layout)

    def normalize(self, name):
        return text_weight_name(name, getattr(self.config, "num_hidden_layers", None))

    @cached_property
    def kda_layers(self):
        return frozenset(kda_layer_ids(self.config))

    @cached_property
    def kda(self):
        kda = self.config.linear_attn_config
        return KDALayer.weight_shards(self.layout.tensor, kda["num_heads"], kda["head_dim"])

    @cached_property
    def mla(self):
        c = self.config
        return MLALayer.weight_shards(
            self.layout.tensor,
            c.num_attention_heads,
            c.qk_nope_head_dim,
            c.qk_rope_head_dim,
            c.v_head_dim,
        )

    @cached_property
    def dense(self):
        return GatedMLP.weight_shards(self.layout.tensor, self.config.intermediate_size)

    @cached_property
    def shared(self):
        return GatedMLP.weight_shards(
            self.layout.tensor, self.config.moe_intermediate_size * self.config.num_shared_experts
        )

    @cached_property
    def expert(self):
        return GatedExpert.weight_shards(
            self.layout.expert_tensor, self.config.moe_intermediate_size
        )

    def shard(self, name, shape=None):
        if self.layout.tensor.size == 1:
            return None
        if name in ("model.embed_tokens.weight", "lm_head.weight"):
            return Embedding.weight_shard(self.layout.tensor, self.config.vocab_size)
        parts = name.split(".", 4)
        if len(parts) != 5 or parts[:2] != ["model", "layers"]:
            return None
        _, _, layer, module, parameter = parts
        if module == "self_attn":
            weights = self.kda if int(layer) in self.kda_layers else self.mla
        elif module != "mlp":
            return None
        elif parameter.startswith("experts."):
            _, _, parameter = parameter.split(".", 2)
            weights = self.expert
        elif parameter.startswith("shared_experts."):
            parameter = parameter.removeprefix("shared_experts.")
            weights = self.shared
        elif parameter in ("gate_proj.weight", "up_proj.weight", "down_proj.weight"):
            weights = self.dense
        else:
            return None
        return weights.get(parameter)

    def read_dense(self, reader, raw_name, name, view, spec):
        if view.get_dtype() not in ("F64", "F32", "F16", "BF16"):
            raise ValueError(f"Unexpected non-floating text weight {raw_name}: {view.get_dtype()}")
        if not name.endswith(".self_attn.A_log"):
            return super().read_dense(reader, raw_name, name, view, spec)
        shape = view.get_shape()
        # The public K3 checkpoint stores head_dim entries, although only the
        # first global num_heads entries are live. Slice local heads before read.
        legacy = len(shape) == 4 and shape[:2] == [1, 1] and shape[-1] == 1
        if len(shape) != 1 and not legacy:
            raise ValueError(f"Unsupported KDA A_log shape: {tuple(shape)}")
        count = shape[2] if legacy else shape[0]
        heads = (getattr(self.config, "linear_attn_config", None) or {}).get("num_heads", count)
        if count < heads:
            raise ValueError(f"KDA A_log has fewer than {heads} heads")
        start, stop = (spec.start, spec.stop) if spec is not None else (0, heads)
        if not 0 <= start < stop <= heads:
            raise ValueError("Kimi TP A_log slice exceeds global head count")
        tensor = view[0, 0, start:stop, 0] if legacy else view[start:stop]
        return tensor.contiguous().float()
