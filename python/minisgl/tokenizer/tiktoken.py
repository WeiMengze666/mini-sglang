"""Local TikToken BPE vocabulary adapter; no checkpoint Python is executed."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from transformers import AddedToken, PreTrainedTokenizer
from transformers.convert_slow_tokenizer import bytes_to_unicode


class TikTokenTokenizer(PreTrainedTokenizer):
    pattern: str | None = None
    default_stop_token: str | None = None
    num_reserved_tokens = 256
    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "tiktoken.model"}

    def __init__(
        self,
        vocab_file: str,
        tokenizer_config: dict[str, Any],
        generation_config: dict[str, Any] | None = None,
    ):
        import tiktoken

        self.vocab_file = str(vocab_file)
        # Read local BPE data directly; tiktoken's general file loader can fetch
        # URLs. This adapter must never execute or fetch checkpoint Python.
        ranks = {}
        for line in Path(vocab_file).read_bytes().splitlines():
            if line:
                token, rank = line.split()
                token, rank = base64.b64decode(token, validate=True), int(rank)
                if token in ranks:
                    raise ValueError("Duplicate TikToken BPE token")
                ranks[token] = rank
        if set(ranks.values()) != set(range(len(ranks))):
            raise ValueError("TikToken BPE ranks must be contiguous")
        if not all(bytes([i]) in ranks for i in range(256)):
            raise ValueError("TikToken BPE vocabulary must contain every byte")
        added = {
            int(i): AddedToken(**value)
            for i, value in tokenizer_config.get("added_tokens_decoder", {}).items()
        }
        if any(i < len(ranks) or i >= len(ranks) + self.num_reserved_tokens for i in added):
            raise ValueError("TikToken added tokens must use the 256 reserved IDs")
        self.special_tokens = {
            added[i].content if i in added else f"<|reserved_token_{i}|>": i
            for i in range(len(ranks), len(ranks) + self.num_reserved_tokens)
        }
        pattern = tokenizer_config.get("pat_str") or self.pattern
        if not pattern:
            raise ValueError("TikToken tokenization requires a pre-tokenization pattern")
        self._encoding = tiktoken.Encoding(
            name="local-bpe",
            pat_str=pattern,
            mergeable_ranks=ranks,
            special_tokens=self.special_tokens,
        )
        byte_encoder = bytes_to_unicode()
        self._byte_decoder = {v: k for k, v in byte_encoder.items()}
        self._decoder = {
            i: "".join(byte_encoder[b] for b in self._encoding.decode_single_token_bytes(i))
            for i in range(self._encoding.n_vocab)
        }
        self._encoder = {token: i for i, token in self._decoder.items()}
        options = {
            key: tokenizer_config[key]
            for key in (
                "bos_token",
                "eos_token",
                "pad_token",
                "unk_token",
                "additional_special_tokens",
                "model_max_length",
            )
            if key in tokenizer_config
        }
        super().__init__(added_tokens_decoder=added, clean_up_tokenization_spaces=False, **options)
        stop = (generation_config or {}).get("eos_token_id")
        if stop is None:
            stop = (
                self.special_tokens[self.default_stop_token]
                if self.default_stop_token
                else self.eos_token_id
            )
        stop = [stop] if isinstance(stop, int) else stop
        if not stop or any(type(i) is not int or not 0 <= i < self.vocab_size for i in stop):
            raise ValueError("Invalid TikToken generation eos_token_id")
        self.stop_token_ids = frozenset(stop)

    @property
    def vocab_size(self) -> int:
        return self._encoding.n_vocab

    def get_vocab(self) -> dict[str, int]:
        return dict(self._encoder, **self.added_tokens_encoder)

    def _encode_piece(self, text: str, allow_special: bool) -> list[int]:
        # Match the published tokenizer's long-input chunking to avoid tiktoken
        # pathological runs without changing ordinary segment tokenization.
        result = []
        for offset in range(0, len(text), 400_000):
            chunk = text[offset : offset + 400_000]
            start, run = 0, 0
            was_space = chunk[0].isspace()
            for i, char in enumerate(chunk):
                is_space = char.isspace()
                run = run + 1 if is_space == was_space else 1
                was_space = is_space
                if run > 25_000:
                    result.extend(
                        self._encoding.encode(
                            chunk[start:i],
                            allowed_special="all" if allow_special else set(),
                            disallowed_special=(),
                        )
                    )
                    start, run = i, 1
            result.extend(
                self._encoding.encode(
                    chunk[start:],
                    allowed_special="all" if allow_special else set(),
                    disallowed_special=(),
                )
            )
        return result

    def _tokenize(self, text: str, **kwargs) -> list[str]:
        return [self._decoder[i] for i in self._encode_piece(text, True)]

    def _convert_token_to_id(self, token: str) -> int:
        return self._encoder.get(token, self.special_tokens[str(self.unk_token)])

    def _convert_id_to_token(self, index: int) -> str:
        return self._decoder[index]

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return bytes(self._byte_decoder[c] for c in "".join(tokens)).decode("utf-8", "replace")

    def _decode(self, token_ids, skip_special_tokens=False, **kwargs) -> str:
        # Decode the byte stream directly, including across token boundaries.
        # HF's generic slow-tokenizer decoder can insert spaces around markers.
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            special = set(self.all_special_ids)
            token_ids = [i for i in token_ids if i not in special]
        return self._encoding.decode(token_ids)

    def save_vocabulary(self, save_directory: str, filename_prefix=None) -> tuple[str]:
        path = Path(save_directory) / (
            (filename_prefix + "-" if filename_prefix else "") + "tiktoken.model"
        )
        path.write_bytes(Path(self.vocab_file).read_bytes())
        return (str(path),)
