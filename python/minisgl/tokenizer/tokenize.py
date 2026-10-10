from __future__ import annotations

from typing import List

import torch
from minisgl.message import TokenizeMsg
from transformers import PreTrainedTokenizerBase


class TokenizeManager:
    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer

    def tokenize(self, msgs: List[TokenizeMsg]) -> List[torch.Tensor]:
        results: List[torch.Tensor] = []
        # TODO: batch tokenization
        for msg in msgs:
            if isinstance(msg.text, list):
                # Keep tokenizer-owned segment boundaries and special-token escaping.
                input_ids = self.tokenizer.apply_chat_template(
                    msg.text,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
            else:
                input_ids = self.tokenizer.encode(msg.text, return_tensors="pt")
            input_ids = torch.as_tensor(input_ids)
            results.append(input_ids.view(-1).to(torch.int32))
        return results
