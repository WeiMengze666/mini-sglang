"""Segmented XTML text conversations with escaped literal control markers."""

from .tiktoken import TikTokenTokenizer

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


class XTMLTokenizer(TikTokenTokenizer):
    pattern = PATTERN
    default_stop_token = "<|end_of_msg|>"

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
                    raise ValueError("XTML chat content must be text or text-only blocks")
                for part in value:
                    if (
                        not isinstance(part, dict)
                        or part.get("type") != "text"
                        or not isinstance(part.get("text"), str)
                    ):
                        raise ValueError(
                            "XTML basic chat does not support images or non-text content"
                        )
                    text(part["text"])

        if thinking and thinking_effort is not None:
            if thinking_effort not in {"low", "high", "max"}:
                raise ValueError("XTML thinking_effort must be low, high, max, or None")
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
                raise ValueError("XTML basic chat supports only system, user, and assistant roles")
            if any(
                message.get(k) for k in ("tools", "tool_calls", "function_call", "tool_call_id")
            ):
                raise ValueError("XTML basic chat does not support tool calls")
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
                "XTML basic chat does not support tools or additional formatting options"
            )
        if not isinstance(conversation, list):
            raise ValueError("XTML chat requires a list of messages")
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
