"""Map split Hugging Face projections to fused Mini parameters."""

import re

import torch
from minisgl.layers.embedding import VocabParallelEmbedding
from minisgl.layers.linear import LinearColParallelMerged, LinearQKVMerged, LinearRowParallel
from minisgl.utils import div_ceil, div_even

from .loader import Checkpoint

_QKV = ("q_proj", "k_proj", "v_proj")
_GATE_UP = ("gate_proj", "up_proj")
_EXPERT = re.compile(r"^(.+\.experts)\.(\d+)\.(.+)$")


class HuggingFaceCheckpoint(Checkpoint):
    @staticmethod
    def from_metadata(metadata):
        from minisgl.models.config import ModelConfig
        from transformers import AutoConfig

        values = dict(metadata)
        model_type = values.pop("model_type")
        return ModelConfig.from_hf(AutoConfig.for_model(model_type, **values))

    def normalize(self, name):
        if name.startswith(("vision_tower.", "multi_modal_projector.")):
            return None
        return name.removeprefix("language_model.")

    def shard(self, name, shape):
        tp = self.layout.expert_tensor if _EXPERT.match(name) else self.layout.tensor
        if tp.size == 1:
            return None
        module = name.split(".")[-2] if "." in name else ""
        if module in ("k_proj", "v_proj"):
            heads = self.config.num_kv_heads
            return LinearQKVMerged.head_shard(tp, heads, div_even(shape[0], heads))
        if module in ("q_proj", *_GATE_UP):
            return LinearColParallelMerged.weight_shard(tp, shape[0])
        if module in ("o_proj", "down_proj") and name.endswith(".weight"):
            return LinearRowParallel.weight_shard(tp, shape[1])
        if module in ("lm_head", "embed_tokens"):
            return VocabParallelEmbedding.weight_shard(tp, shape[0])
        return None

    def read_dense(self, reader, raw_name, name, view, spec):
        weight = super().read_dense(reader, raw_name, name, view, spec)
        if spec is not None and name.split(".")[-2] in ("lm_head", "embed_tokens"):
            local_rows = div_ceil(view.get_shape()[0], self.layout.tensor.size)
            if spec.length < local_rows:
                padding = weight.new_zeros((local_rows - spec.length, *weight.shape[1:]))
                weight = torch.cat((weight, padding))
        return weight

    def assemble(self, weights):
        projections, experts = {}, {}
        local = (
            self.layout.expert_range(self.config.num_experts) if self.config.is_moe else range(0)
        )
        for name, tensor in weights:
            path = name.split(".")
            module = path[-2] if len(path) > 1 else ""
            if module in (*_QKV, *_GATE_UP):
                query_group = module in _QKV
                parts = _QKV if query_group else _GATE_UP
                path[-2] = "qkv_proj" if query_group else "gate_up_proj"
                name = ".".join(path)
                slots = projections.setdefault(name, {})
                slots[module] = tensor
                if len(slots) < len(parts):
                    continue
                tensor = torch.cat([slots[p] for p in parts])
                del projections[name], slots
            match = _EXPERT.match(name)
            if match and self.config.is_moe:
                name = match[1] + "." + match[3].removesuffix(".weight")
                slots = experts.setdefault(name, {})
                slots[int(match[2])] = tensor
                if len(slots) < len(local):
                    continue
                tensor = torch.stack([slots[i] for i in local])
                del experts[name], slots
            yield name, tensor
        if projections or experts:
            raise ValueError(f"Incomplete checkpoint groups: {list(projections)} {list(experts)}")
