"""K3 checkpoint aliases and dense projection constraints."""

from collections.abc import Mapping

from minisgl.quantization.compressed_tensors import quantization_rules as _rules
from minisgl.quantization.compressed_tensors import replace_quantized_linears as _replace

# These checkpoint modules use dense nn.Linear rather than the quantized class.
_NATIVE_DENSE = {
    "routed_expert_down_proj",
    "routed_expert_up_proj",
    "self_attention_res_proj",
    "mlp_res_proj",
    "output_attn_res_proj",
}


def quantization_rules(config):
    quant = (
        config.get("quantization_config")
        if isinstance(config, Mapping)
        else getattr(config, "quantization_config", None)
    )
    return _rules(quant)


def _aliases(name):
    names = (name, "language_model." + name)
    return (*names, *(alias.replace(".mlp.", ".block_sparse_moe.") for alias in names))


def _validate(name, op):
    if name.endswith("self_attn.kv_b_proj"):
        raise NotImplementedError(
            "Packed MLA kv_b_proj is unsupported: latent attention requires its dense weight; keep it ignored in quantization_config"
        )


def replace_quantized_linears(model, config, *, chunk_rows=128):
    quant = (
        config.get("quantization_config")
        if isinstance(config, Mapping)
        else getattr(config, "quantization_config", None)
    )
    return _replace(
        model,
        quant,
        chunk_rows=chunk_rows,
        aliases=_aliases,
        linear_types=("Linear", "KimiLinear"),
        exclude=lambda name, op: name.rsplit(".", 1)[-1] in _NATIVE_DENSE,
        validate=_validate,
    )
