import functools
import json
import os
from typing import Any

from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.asyncio import tqdm
from transformers import AutoConfig, AutoTokenizer, PretrainedConfig, PreTrainedTokenizerBase
from transformers.models.auto.tokenization_auto import get_tokenizer_config


class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs.pop("name", None)
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def load_tokenizer(model_path: str) -> PreTrainedTokenizerBase:
    metadata = get_tokenizer_config(model_path)
    end_marker = metadata.get("added_tokens_decoder", {}).get("163586", {}).get("content")
    if metadata.get("tokenizer_class") == "TikTokenTokenizer" and end_marker == "<|end_of_msg|>":
        from minisgl.tokenizer.kimi import KimiTokenizer

        folder = model_path
        if not os.path.isdir(folder):
            revision = metadata.get("_commit_hash")
            if not revision:
                raise ValueError("Kimi tokenizer metadata has no pinned snapshot revision")
            folder = snapshot_download(
                model_path,
                revision=revision,
                allow_patterns=[
                    "tokenizer_config.json",
                    "tiktoken.model",
                    "generation_config.json",
                ],
            )
        generation_file = os.path.join(folder, "generation_config.json")
        generation = None
        if os.path.isfile(generation_file):
            with open(generation_file, encoding="utf-8") as file:
                generation = json.load(file)
        return KimiTokenizer(os.path.join(folder, "tiktoken.model"), metadata, generation)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # Some Mistral models store chat_template in a separate JSON file
    if not getattr(tokenizer, "chat_template", None):
        try:
            path = hf_hub_download(repo_id=model_path, filename="chat_template.json")
            with open(path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = json.load(f)["chat_template"]
        except Exception:
            pass
    return tokenizer


@functools.cache
def _load_hf_config(model_path: str) -> Any:
    raw, _ = PretrainedConfig.get_config_dict(model_path)
    if raw.get("model_type") in ("kimi_k3", "kimi_linear"):
        from minisgl.models.kimi_config import KimiK3Config, KimiTextConfig

        cls = KimiK3Config if raw["model_type"] == "kimi_k3" else KimiTextConfig
        return cls(**raw)
    return AutoConfig.from_pretrained(model_path)


def cached_load_hf_config(model_path: str) -> PretrainedConfig:
    config = _load_hf_config(model_path)
    return type(config)(**config.to_dict())


def download_hf_weight(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    try:
        return snapshot_download(
            model_path,
            allow_patterns=["*.safetensors"],
            tqdm_class=DisabledTqdm,
        )
    except Exception as e:
        raise ValueError(
            f"Model path '{model_path}' is neither a local directory nor a valid model ID: {e}"
        )


def get_stop_token_ids(tokenizer: PreTrainedTokenizerBase) -> frozenset[int]:
    """Generation terminators may differ from a tokenizer's corpus EOS identity."""
    ids = getattr(tokenizer, "stop_token_ids", None)
    if ids is None:
        ids = tokenizer.eos_token_id
    if ids is None:
        return frozenset()
    if isinstance(ids, int):
        ids = [ids]
    return frozenset(ids)
