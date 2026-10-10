"""Local, data-only Kimi K3 tokenization and basic text chat formatting.

The published tokenizer uses segmented XTML, not a Jinja template. Encoding a
rendered prompt again loses both segment boundaries and literal-marker escaping.
Only system/user/assistant text messages are supported here, without tools/images.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from transformers import AddedToken, PreTrainedTokenizer
from transformers.convert_slow_tokenizer import bytes_to_unicode

# Pinned K3 tokenizer's pre-tokenization pattern. Keep the Han branch and Unicode
# set intersections: replacing this with a generic GPT regex changes token IDs.
PATTERN = "|".join(
    [
        r"[\p{Han}]+",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]+[\p{Ll}\p{Lm}\p{Lo}\p{M}&&[^\p{Han}]]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?",
        r"\p{N}{1,3}",
        r" ?[^\s\p{L}\p{N}]+[\r\n]*",
        r"\s*[\r\n]+",
        r"\s+(?!\S)",
        r"\s+",
    ]
)


class KimiTokenizer(PreTrainedTokenizer):
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
                    raise ValueError("Duplicate Kimi BPE token")
                ranks[token] = rank
        if set(ranks.values()) != set(range(len(ranks))):
            raise ValueError("Kimi BPE ranks must be contiguous")
        if not all(bytes([i]) in ranks for i in range(256)):
            raise ValueError("Kimi BPE vocabulary must contain every byte")
        added = {
            int(i): AddedToken(**value)
            for i, value in tokenizer_config.get("added_tokens_decoder", {}).items()
        }
        if any(i < len(ranks) or i >= len(ranks) + 256 for i in added):
            raise ValueError("Kimi added tokens must use the 256 reserved IDs")
        self.special_tokens = {
            added[i].content if i in added else f"<|reserved_token_{i}|>": i
            for i in range(len(ranks), len(ranks) + 256)
        }
        self._encoding = tiktoken.Encoding(
            name="kimi-k3",
            pat_str=PATTERN,
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
            stop = self.special_tokens["<|end_of_msg|>"]
        stop = [stop] if isinstance(stop, int) else stop
        if not stop or any(type(i) is not int or not 0 <= i < self.vocab_size for i in stop):
            raise ValueError("Invalid Kimi generation eos_token_id")
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

    @staticmethod
    def _chat_segments(messages, *, add_generation_prompt, thinking, thinking_effort):
        segments: list[tuple[str, bool]] = []

        def text(value):
            if str(value):
                segments.append((str(value), False))

        def control(value):
            segments.append((value, True))

        def tag(name, attrs=(), close=False):
            control("<|close|>" if close else "<|open|>")
            text(name)
            for key, value in attrs:
                text(f" {key}")
                text('="')
                text(str(value).replace("&", "&amp;").replace('"', "&quot;"))
                text('"')
            control("<|sep|>")

        def content(value):
            if isinstance(value, str):
                text(value)
            elif value is not None:
                if not isinstance(value, list):
                    raise ValueError("Kimi chat content must be text or text-only blocks")
                for part in value:
                    if (
                        not isinstance(part, dict)
                        or part.get("type") != "text"
                        or not isinstance(part.get("text"), str)
                    ):
                        raise ValueError(
                            "Kimi basic chat does not support images or non-text content"
                        )
                    text(part["text"])

        if thinking and thinking_effort is not None:
            if thinking_effort not in {"low", "high", "max"}:
                raise ValueError("Kimi thinking_effort must be low, high, max, or None")
            tag("message", [("role", "system"), ("type", "thinking-effort")])
            text(
                "`thinking_effort` guides on how much to think in your "
                "thinking channel (not including the response channel), "
                "supported values include `low`, `medium`, `high`, and `max`.\n"
                f"Now the system is invoked with `thinking_effort={thinking_effort}`."
            )
            tag("message", close=True)
            control("<|end_of_msg|>")
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in {
                "system",
                "user",
                "assistant",
            }:
                raise ValueError("Kimi basic chat supports only system, user, and assistant roles")
            if any(
                message.get(k) for k in ("tools", "tool_calls", "function_call", "tool_call_id")
            ):
                raise ValueError("Kimi basic chat does not support tool calls")
            role = message["role"]
            attrs = [("role", role)]
            if message.get("name"):
                attrs.append(("name", message["name"]))
            tag("message", attrs)
            if role == "assistant":
                if thinking:
                    tag("think")
                    reasoning = message.get("reasoning_content") or message.get("reasoning")
                    if reasoning is not None:
                        text(reasoning)
                    tag("think", close=True)
                tag("response")
            content(message.get("content"))
            if role == "assistant":
                tag("response", close=True)
            tag("message", close=True)
            control("<|end_of_msg|>")
        if add_generation_prompt:
            tag("message", [("role", "assistant")])
            tag("think" if thinking else "response")
        return segments

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        tokenize=False,
        add_generation_prompt=True,
        thinking=True,
        thinking_effort="max",
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
        return_dict=False,
        **kwargs,
    ):
        if tools or kwargs:
            raise ValueError(
                "Kimi basic chat does not support tools or additional formatting options"
            )
        if not isinstance(conversation, list):
            raise ValueError("Kimi chat requires a list of messages")
        batched = bool(conversation) and isinstance(conversation[0], list)
        batches = conversation if batched else [conversation]
        rendered = [
            self._chat_segments(
                messages,
                add_generation_prompt=add_generation_prompt,
                thinking=thinking,
                thinking_effort=thinking_effort,
            )
            for messages in batches
        ]
        if not tokenize:
            texts = ["".join(text for text, _ in parts) for parts in rendered]
            return texts if batched else texts[0]
        ids = [
            [i for text, special in parts for i in self._encode_piece(text, special)]
            for parts in rendered
        ]
        if truncation and max_length is not None:
            ids = [values[:max_length] for values in ids]
        if not (batched or padding or return_tensors is not None or return_dict):
            return ids[0]
        result = self.pad(
            [{"input_ids": values, "attention_mask": [1] * len(values)} for values in ids],
            padding=padding,
            max_length=max_length if padding else None,
            return_tensors=return_tensors,
            return_attention_mask=True,
        )
        if return_dict:
            return result
        values = result["input_ids"]
        return values if batched or return_tensors is not None else values[0]
