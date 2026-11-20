from __future__ import annotations

from typing import List, Optional

from .xtml import (
    MESSAGE_CLOSE,
    RESPONSE_CLOSE,
    RESPONSE_OPEN,
    THINK_CLOSE,
    THINK_OPEN,
    TOOLS_OPEN,
    partial_suffix_len,
    strip_partial_marker_suffix,
    strip_response_wrappers,
)


class StreamingParseResult:
    """Result of streaming incremental parsing."""

    def __init__(
        self,
        normal_text: Optional[str] = None,
        reasoning_text: Optional[str] = None,
    ):
        self.normal_text = normal_text or ""
        self.reasoning_text = reasoning_text or ""


class BaseReasoningFormatDetector:
    """Base class providing two sets of interfaces: one-time and streaming incremental."""

    def __init__(
        self,
        think_start_token: str,
        think_end_token: str,
        think_excluded_tokens: Optional[List[str]] = None,
        force_reasoning: bool = False,
        stream_reasoning: bool = True,
        tool_start_token: Optional[str] = None,
        continue_final_message: bool = False,
        previous_content: str = "",
        thinks_internally: bool = False,
        reasoning_default: str = "always",
        force_nonempty_content: bool = False,
        tool_start_at_line_start: bool = False,
    ):
        self.think_start_token = think_start_token
        self.think_end_token = think_end_token
        self.think_excluded_tokens = think_excluded_tokens
        self.tool_start_token = tool_start_token
        # Only a tool_start_token that begins a line interrupts reasoning, so
        # the model mentioning the tag mid-sentence does not end the block.
        self.tool_start_at_line_start = tool_start_at_line_start
        self.force_reasoning = force_reasoning
        self._in_reasoning = force_reasoning
        self.stream_reasoning = stream_reasoning
        self.thinks_internally = thinks_internally
        self.reasoning_default = reasoning_default

        self._buffer = ""
        self._streamed_reasoning_tail = ""
        self.stripped_think_start = False
        self.think_start_self_label = ""

        self._force_nonempty_content = force_nonempty_content
        self._accumulated_reasoning = ""

        self.continue_final_message = continue_final_message
        if self.continue_final_message:
            self.previous_content = previous_content
            self.previous_count = len(previous_content)
        else:
            self.previous_content = ""
            self.previous_count = 0

        if self.think_start_token in self.previous_content:
            self._in_reasoning = True
        if self.think_end_token in self.previous_content:
            self._in_reasoning = False

    def get_think_end_token_ids(self, tokenizer) -> List[int]:
        return tokenizer.encode(self.think_end_token, add_special_tokens=False)

    def _maybe_apply_force_nonempty_content(
        self, ret: StreamingParseResult
    ) -> StreamingParseResult:
        if self._force_nonempty_content and not ret.normal_text:
            ret.normal_text, ret.reasoning_text = ret.reasoning_text, ret.normal_text
        return ret

    def _find_tool_start(self, text: str, preceded_by: str = "") -> int:
        """Index of the first tool_start_token that interrupts reasoning, or -1.

        `preceded_by` is the character just before `text`, for a streaming
        buffer whose leading newline was already emitted as reasoning."""
        if not self.tool_start_token:
            return -1
        idx = text.find(self.tool_start_token)
        while idx != -1:
            prev = text[idx - 1] if idx > 0 else preceded_by
            # Nothing precedes the token at the absolute start of generation;
            # that counts as line start too.
            if not self.tool_start_at_line_start or prev in ("", "\n"):
                return idx
            idx = text.find(self.tool_start_token, idx + 1)
        return -1

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        """
        One-time parsing: Detects and parses reasoning sections in the provided text.
        Returns both reasoning content and normal text separately.
        """
        return self._maybe_apply_force_nonempty_content(self._detect_and_parse_impl(text))

    def _detect_and_parse_impl(self, text: str) -> StreamingParseResult:
        in_reasoning = self._in_reasoning or self.think_start_token in text

        if not in_reasoning:
            return StreamingParseResult(normal_text=text)

        # The text is considered to be in a reasoning block.
        think_start_text = self.think_start_token + self.think_start_self_label
        processed_text = text
        while processed_text.startswith(think_start_text):
            processed_text = processed_text[len(think_start_text) :]

        if (
            self.think_end_token not in processed_text
            and self.think_end_token not in self.previous_content
        ):
            # Check for tool_start_token interruption
            tool_idx = self._find_tool_start(processed_text) if in_reasoning else -1
            if tool_idx != -1:
                reasoning_text = processed_text[:tool_idx]
                # Preserve tool_start_token in normal text
                normal_text = processed_text[tool_idx:]
                return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)
            # Assume reasoning was truncated before end token
            return StreamingParseResult(
                reasoning_text=self._strip_partial_tool_start(processed_text)
            )

        # Extract reasoning content
        if self.think_end_token in processed_text:
            splits = processed_text.split(self.think_end_token, maxsplit=1)
            reasoning_text = splits[0]
            normal_text = splits[1]

            return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)
        else:
            # think_end_token is in self.previous_content for continue_final_message=True case
            return StreamingParseResult(normal_text=processed_text)

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        """
        Streaming incremental parsing for reasoning content.
        Handles partial reasoning tags and content.

        If stream_reasoning is False:
            Accumulates reasoning content until the end tag is found
        If stream_reasoning is True:
            Streams reasoning content as it arrives
        """
        ret = self._parse_streaming_increment_impl(new_text)
        if self._force_nonempty_content:
            if self._in_reasoning:
                self._accumulated_reasoning += ret.reasoning_text
            else:
                self._accumulated_reasoning = ""
        return ret

    def _parse_streaming_increment_impl(self, new_text: str) -> StreamingParseResult:
        self._buffer += new_text
        current_text = self._buffer

        think_start_text = self.think_start_token + self.think_start_self_label

        # If the current text is a prefix of the think token, keep buffering
        tokens_to_check = [think_start_text, self.think_end_token]
        if self.tool_start_token:
            tokens_to_check.append(self.tool_start_token)
        if any(
            token.startswith(current_text) and token != current_text for token in tokens_to_check
        ):
            return StreamingParseResult()

        # Strip `<think>` token if present
        if not self.stripped_think_start and think_start_text in current_text:
            current_text = current_text.replace(think_start_text, "", 1)
            # Write back, or stream_reasoning=False carries the token into finish().
            self._buffer = current_text
            self.stripped_think_start = True
            self._in_reasoning = True

        # Handle end of reasoning block
        if self._in_reasoning and self.think_end_token in current_text:
            end_idx = current_text.find(self.think_end_token)

            reasoning_text = current_text[:end_idx]

            self._buffer = ""
            self._in_reasoning = False
            normal_text = current_text[end_idx + len(self.think_end_token) :]

            return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)

        # Continue with reasoning content
        if self._in_reasoning:
            # Check for tool_start_token interruption. Streaming cannot see a
            # think_end_token that has not arrived yet; see the chunk_dependent test.
            tool_idx = self._find_tool_start(current_text, self._streamed_reasoning_tail)
            if tool_idx != -1:
                reasoning_text = current_text[:tool_idx]
                # Preserve tool_start_token in normal text
                normal_text = current_text[tool_idx:]
                self._buffer = ""
                self._in_reasoning = False
                return StreamingParseResult(normal_text=normal_text, reasoning_text=reasoning_text)
            if self.stream_reasoning:
                # Minus any trailing slice that could be a token split across chunks.
                holdback_tokens = [self.think_end_token]
                if self.tool_start_token:
                    holdback_tokens.append(self.tool_start_token)
                if not self.stripped_think_start:
                    # force_reasoning never saw the opening token; it can still split.
                    holdback_tokens.append(think_start_text)
                holdback = max(
                    self._ends_with_partial_token(current_text, token) for token in holdback_tokens
                )
                self._buffer = current_text[len(current_text) - holdback :]
                reasoning_text = current_text[: len(current_text) - holdback]
                if reasoning_text:
                    self._streamed_reasoning_tail = reasoning_text[-1]
                return StreamingParseResult(reasoning_text=reasoning_text)
            else:
                return StreamingParseResult()

        # If we're not in a reasoning block return as normal text
        if not self._in_reasoning:
            self._buffer = ""
            return StreamingParseResult(normal_text=current_text)

        return StreamingParseResult()

    def _strip_leading_think_start(self, text: str) -> str:
        think_start_text = self.think_start_token + self.think_start_self_label
        if text.startswith(think_start_text):
            return text[len(think_start_text) :]
        return text

    @staticmethod
    def _ends_with_partial_token(buffer: str, token: str) -> int:
        """Length of the longest trailing slice of `buffer` that is a strict prefix
        of `token`. Longest, so a token whose prefix repeats inside itself does not
        get cut short and leak the rest of the marker."""
        for i in range(min(len(buffer), len(token) - 1), 0, -1):
            if token.startswith(buffer[-i:]):
                return i
        return 0

    def _strip_partial_tool_start(self, text: str, preceded_by: str = "") -> str:
        """Drop a trailing partial tool-start token that begins a line: generation
        ended mid-tag, and flushing the fragment as reasoning would surface raw
        markup to the client. `preceded_by` is as in `_find_tool_start`."""
        if not self.tool_start_at_line_start or not text:
            return text
        n = self._ends_with_partial_token(text, self.tool_start_token or "")
        if n == 0:
            return text
        start = len(text) - n
        prev = text[start - 1] if start > 0 else preceded_by
        if prev in ("", "\n"):
            return text[:start]
        return text

    def finish(self) -> StreamingParseResult:
        """Flush reasoning still buffered when the stream ends before the end token
        (e.g. max_tokens cut it short), instead of dropping it: the whole block under
        stream_reasoning=False, the held-back token suffix under stream_reasoning=True.
        force_nonempty_content emits it as normal_text, else as reasoning_text."""
        if not self._in_reasoning:
            # Same as the reasoning-side flush below: a held-back slice that never
            # became a token is content.
            leftover = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=leftover)

        # Defensive: subclasses that fill _buffer themselves may not have stripped
        # the opening think token that _parse_streaming_increment_impl removes.
        buffer = self._strip_leading_think_start(self._buffer)
        buffer = self._strip_partial_tool_start(buffer, self._streamed_reasoning_tail)
        self._buffer = ""

        if self._force_nonempty_content:
            normal_text = self._accumulated_reasoning + buffer
            self._accumulated_reasoning = ""
            if normal_text:
                return StreamingParseResult(normal_text=normal_text)
            return StreamingParseResult()

        if buffer:
            return StreamingParseResult(reasoning_text=buffer)

        return StreamingParseResult()


class Qwen3Detector(BaseReasoningFormatDetector):
    """
    Detector for Qwen3 models (e.g., Qwen/Qwen3-235B-A22B).
    Assumes reasoning format:
      (<think>)*(.*)</think>

    Qwen3 models released before 07/2025 supports switching between thinking mode and normal
    mode using `enable_thinking` parameter in the request parameter.
      - enable_thinking=True: "<think>reasoning content</think>The answer is 42."
      - enable_thinking=False: "The answer is 42." (no thinking tokens)

    Args:
        stream_reasoning (bool): If False, accumulates reasoning content until the end tag.
            If True, streams reasoning content as it arrives.
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = False,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        think_excluded_tokens = [
            "<tool_call>",
            "</tool_call>",
            "<|im_end|>",
            "<|endoftext|>",
        ]
        super().__init__(
            "<think>",
            "</think>",
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            # Qwen3.5 sometimes opens ``<tool_call>`` without closing
            # ``</think>``; treat it as an implicit reasoning close.
            tool_start_token="<tool_call>",
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            thinks_internally=True,
            reasoning_default="enable_thinking",
            force_nonempty_content=force_nonempty_content,
        )


class KimiK3Detector(BaseReasoningFormatDetector):
    """Detector for the Kimi K3 XTML think channel.

    K3 wraps reasoning as ``<|open|>think<|sep|>...<|close|>think<|sep|>``
    where each marker is a multi-token special sequence, so partial markers
    can straddle streaming chunks and must be held back. In thinking mode
    the serving layer may feed the open marker as the generation prefix, so
    output can begin inside the think channel with no open marker
    (``force_reasoning=True`` covers this).

    Post-reasoning content is unwrapped from the XTML ``response`` /
    ``message`` markers; a ``tools`` channel is passed through raw for the
    kimi_k3 tool-call detector.

    The model does not always honour the pre-filled think channel: on very long
    prompts (~1M tokens) it sometimes emits a zero-length think section and
    writes the reply directly, closing with
    ``<|close|>response<|sep|><|close|>message<|sep|>`` and never producing
    ``<|close|>think<|sep|>`` or ``<|open|>response<|sep|>``. A bare
    ``<|close|>response<|sep|>`` therefore proves the preceding text was the
    response channel, and it is reported as content rather than reasoning
    (see :meth:`_skipped_think_channel`).
    """

    def __init__(
        self,
        stream_reasoning: bool = True,
        force_reasoning: bool = True,
        continue_final_message: bool = False,
        previous_content: str = "",
        force_nonempty_content: bool = False,
    ):
        # strict-thinking flattens these to single token ids, so the full marker
        # "<|open|>response<|sep|>" is inexpressible. The bare name works: it
        # follows <|open|> unspaced, so it tokenizes to the no-space variant, not
        # the " response"/" message" tokens prose uses -- at the cost of not being
        # able to start those words unspaced mid-reasoning. tools is left out on
        # purpose: the model may jump from think straight into that channel.
        think_excluded_tokens = [
            "response",
            "message",
            "<|end_of_msg|>",
            "[EOS]",
            "[EOT]",
        ]
        super().__init__(
            THINK_OPEN,
            THINK_CLOSE,
            think_excluded_tokens=think_excluded_tokens,
            force_reasoning=force_reasoning,
            stream_reasoning=stream_reasoning,
            tool_start_token=TOOLS_OPEN,
            continue_final_message=continue_final_message,
            previous_content=previous_content,
            reasoning_default="thinking",
        )
        # Unlike the base class, K3 cannot use `normal_text == ""` alone:
        # skipped-think and truncated marker-free reasoning end up identical.
        self._force_nonempty_content = force_nonempty_content
        self._reasoning_done = False
        self._tools_passthrough = False
        self._stream_text = ""
        self._streamed_reasoning: list[str] = []
        self._discard_delayed_think_close = False

    def _clean_content(self, text: str) -> str:
        tools_idx = text.find(TOOLS_OPEN)
        if tools_idx != -1:
            return strip_response_wrappers(text[:tools_idx]) + text[tools_idx:]
        return strip_response_wrappers(text)

    def _next_channel_idx(self, text: str, start: int = 0) -> int:
        found = [
            idx
            for token in (RESPONSE_OPEN, self.tool_start_token)
            if (idx := text.find(token, start)) != -1
        ]
        return min(found) if found else -1

    @staticmethod
    def _skipped_think_channel(
        text: str,
        start: int = 0,
        think_close_idx: int = -1,
        next_channel_idx: int = -1,
    ) -> bool:
        response_close_idx = text.find(RESPONSE_CLOSE, start)
        return response_close_idx != -1 and all(
            boundary_idx == -1 or response_close_idx < boundary_idx
            for boundary_idx in (think_close_idx, next_channel_idx)
        )

    def _clean_skipped_think_content(self, text: str) -> str:
        return self._clean_content(text.replace(self.think_end_token, ""))

    def detect_and_parse(self, text: str) -> StreamingParseResult:
        in_reasoning = self._in_reasoning or self.think_start_token in text
        if not in_reasoning and self.think_end_token not in text:
            return StreamingParseResult(normal_text=self._clean_content(text))
        if self._force_nonempty_content and self._is_skipped_think_answer(text):
            return StreamingParseResult(normal_text=self._clean_content(text))

        open_idx = text.find(self.think_start_token)
        start = open_idx + len(self.think_start_token) if open_idx != -1 else 0
        close_idx = text.find(self.think_end_token, start)
        tools_idx = text.find(self.tool_start_token, start)
        channel_idx = self._next_channel_idx(text, start)
        if self._skipped_think_channel(text, start, close_idx, channel_idx):
            return StreamingParseResult(normal_text=self._clean_skipped_think_content(text[start:]))
        if close_idx != -1 and tools_idx != -1 and tools_idx < close_idx:
            return StreamingParseResult(
                reasoning_text=strip_partial_marker_suffix(text[start:tools_idx]),
                normal_text=self._clean_content(text[tools_idx:]),
            )
        if close_idx == -1:
            if channel_idx != -1:
                return StreamingParseResult(
                    reasoning_text=strip_partial_marker_suffix(text[start:channel_idx]),
                    normal_text=self._clean_content(text[channel_idx:]),
                )
            return StreamingParseResult(reasoning_text=strip_partial_marker_suffix(text[start:]))

        reasoning_text = text[start:close_idx]
        rest = text[close_idx + len(self.think_end_token) :]
        return StreamingParseResult(
            reasoning_text=reasoning_text, normal_text=self._clean_content(rest)
        )

    def _is_skipped_think_answer(self, text: str) -> bool:
        return (
            self.think_start_token not in text
            and self.think_end_token not in text
            and self.think_start_token.removesuffix("<|sep|>") not in text
            and self.think_end_token.removesuffix("<|sep|>") not in text
            and (RESPONSE_CLOSE in text or MESSAGE_CLOSE in text)
        )

    def parse_streaming_increment(self, new_text: str) -> StreamingParseResult:
        self._buffer += new_text
        if self._force_nonempty_content:
            self._stream_text += new_text

        if not self._in_reasoning and not self._reasoning_done:
            open_idx = self._buffer.find(self.think_start_token)
            if open_idx != -1:
                self._buffer = self._buffer[open_idx + len(self.think_start_token) :]
                self._in_reasoning = True
                self.stripped_think_start = True
            elif self.think_start_token.startswith(self._buffer):
                return StreamingParseResult()
            else:
                self._reasoning_done = True

        if self._in_reasoning:
            buf = self._buffer
            if not self.stripped_think_start:
                open_idx = buf.find(self.think_start_token)
                if open_idx != -1:
                    buf = buf[open_idx + len(self.think_start_token) :]
                    self._buffer = buf
                    self.stripped_think_start = True

            close_idx = buf.find(self.think_end_token)
            tools_idx = buf.find(self.tool_start_token)
            channel_idx = self._next_channel_idx(buf)
            if self._skipped_think_channel(
                buf,
                think_close_idx=close_idx,
                next_channel_idx=channel_idx,
            ):
                replay = "".join(self._streamed_reasoning)
                self._streamed_reasoning.clear()
                self._in_reasoning = False
                self._reasoning_done = True
                self._discard_delayed_think_close = True
                return StreamingParseResult(normal_text=(replay + self._drain_content()) or None)

            if close_idx != -1 and not (tools_idx != -1 and tools_idx < close_idx):
                reasoning_text = buf[:close_idx]
                self._buffer = buf[close_idx + len(self.think_end_token) :]
                self._in_reasoning = False
                self._reasoning_done = True
                self._streamed_reasoning.clear()
                return StreamingParseResult(
                    reasoning_text=reasoning_text or None,
                    normal_text=self._drain_content() or None,
                )

            if channel_idx != -1:
                reasoning_text = strip_partial_marker_suffix(buf[:channel_idx])
                self._buffer = buf[channel_idx:]
                self._in_reasoning = False
                self._reasoning_done = True
                self._streamed_reasoning.clear()
                self._tools_passthrough = buf.startswith(self.tool_start_token, channel_idx)
                return StreamingParseResult(
                    reasoning_text=reasoning_text or None,
                    normal_text=self._drain_content() or None,
                )

            if not self.stream_reasoning:
                return StreamingParseResult()
            markers = [
                self.think_end_token,
                self.tool_start_token,
                RESPONSE_OPEN,
                RESPONSE_CLOSE,
                MESSAGE_CLOSE,
            ]
            if not self.stripped_think_start:
                markers.append(self.think_start_token)
            holdback = partial_suffix_len(buf, markers)
            emit = buf[: len(buf) - holdback] if holdback else buf
            emit = strip_partial_marker_suffix(emit)
            self._buffer = buf[len(emit) :]
            self._streamed_reasoning.append(emit)
            return StreamingParseResult(reasoning_text=emit)

        return StreamingParseResult(normal_text=self._drain_content())

    def finish(self) -> StreamingParseResult:
        self._streamed_reasoning.clear()
        if not self._force_nonempty_content:
            return super().finish()
        text, self._stream_text = self._stream_text, ""
        if self._in_reasoning and self._is_skipped_think_answer(text):
            # _in_reasoning means no channel decision happened mid-stream, so the
            # answer went out as reasoning; without this gate the re-emit duplicates
            # answers already streamed as content (RESPONSE_OPEN / force_reasoning=False).
            self._buffer = ""
            return StreamingParseResult(normal_text=self._clean_content(text))
        if self._in_reasoning and not self.stream_reasoning and self._buffer:
            # super().finish() would emit this buffer as content under
            # force_nonempty_content — the leak the flag exists to prevent.
            buffer, self._buffer = self._buffer, ""
            return StreamingParseResult(reasoning_text=buffer)
        return StreamingParseResult()

    def _drain_content(self) -> str:
        buf = self._buffer
        if not buf:
            return ""
        if self._tools_passthrough:
            holdback = (
                partial_suffix_len(buf, [self.think_end_token])
                if self._discard_delayed_think_close
                else 0
            )
            emit = buf[: len(buf) - holdback] if holdback else buf
            self._buffer = buf[len(emit) :]
            if self._discard_delayed_think_close:
                emit = emit.replace(self.think_end_token, "")
            return emit

        tools_idx = buf.find(TOOLS_OPEN)
        if tools_idx != -1:
            holdback = (
                partial_suffix_len(buf, [self.think_end_token])
                if self._discard_delayed_think_close
                else 0
            )
            emit = buf[: len(buf) - holdback] if holdback else buf
            self._buffer = buf[len(emit) :]
            head = emit[:tools_idx]
            tail = emit[tools_idx:]
            for marker in (RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE):
                head = head.replace(marker, "")
            if self._discard_delayed_think_close:
                head = head.replace(self.think_end_token, "")
                tail = tail.replace(self.think_end_token, "")
            self._tools_passthrough = True
            return head + tail

        markers = [RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE, TOOLS_OPEN]
        if self._discard_delayed_think_close:
            markers.append(self.think_end_token)
        holdback = partial_suffix_len(buf, markers)
        emit = buf[: len(buf) - holdback] if holdback else buf
        self._buffer = buf[len(emit) :]
        for marker in (RESPONSE_OPEN, RESPONSE_CLOSE, MESSAGE_CLOSE):
            emit = emit.replace(marker, "")
        if self._discard_delayed_think_close:
            emit = emit.replace(self.think_end_token, "")
        return emit


FORMATS = {"kimi_k3": (KimiK3Detector, "thinking"), "qwen3": (Qwen3Detector, "enable_thinking")}


class ReasoningParser:
    def __init__(self, name, request):
        detector, toggle = FORMATS[name]
        options = request.get("chat_template_kwargs") or {}
        messages = request.get("messages") or []
        continued = bool(
            request.get("continue_final_message")
            and messages
            and messages[-1]["role"] == "assistant"
        )
        self.detector = detector(
            stream_reasoning=request.get("stream_reasoning", True),
            force_reasoning=options.get(toggle) is not False,
            continue_final_message=continued,
            previous_content=(messages[-1].get("content") or "") if continued else "",
            force_nonempty_content=options.get("force_nonempty_content") is True,
        )

    def parse_stream_chunk(self, text):
        result = self.detector.parse_streaming_increment(text)
        return result.reasoning_text, result.normal_text

    def parse_non_stream(self, text):
        result = self.detector.detect_and_parse(text)
        return result.reasoning_text, result.normal_text

    def parse_stream_end(self):
        result = self.detector.finish()
        return result.reasoning_text, result.normal_text
