from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from minisgl.checkpoint.huggingface import HuggingFaceCheckpoint
from minisgl.layers import BaseOP

if TYPE_CHECKING:
    import torch


class BaseLLMModel(ABC, BaseOP):
    checkpoint_type = HuggingFaceCheckpoint

    @abstractmethod
    def forward(self) -> torch.Tensor: ...
