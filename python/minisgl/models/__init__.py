from .base import BaseLLMModel
from .config import ModelConfig, RotaryConfig
from .register import get_model_class, model_class


def create_model(model_config: ModelConfig) -> BaseLLMModel:
    return get_model_class(model_config.architectures[0], model_config)


def load_weight(model_path, device, *, config=None, prefetch=False, prefetch_num_threads=4):
    from minisgl.checkpoint.loader import load_checkpoint
    from minisgl.distributed.layout import get_parallel_layout
    from minisgl.utils import cached_load_hf_config

    if config is None:
        config = ModelConfig.from_hf(cached_load_hf_config(model_path))
    checkpoint = model_class(config.architectures[0]).checkpoint_type
    return load_checkpoint(
        model_path,
        device,
        checkpoint_type=checkpoint,
        config=config,
        layout=get_parallel_layout(),
        packed=None,
        prefetch=prefetch,
        prefetch_num_threads=prefetch_num_threads,
    )


__all__ = ["create_model", "load_weight", "RotaryConfig"]
