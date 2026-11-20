"""Optional native format parsers behind a model-independent completion interface."""

from __future__ import annotations

import secrets

from .formats import reasoning as reasoning_formats
from .formats.xtml_tools import ToolDetector


def validate_parser_names(reasoning=None, tool_calls=None):
    unknown_reasoning = (
        reasoning if reasoning and reasoning.lower() not in reasoning_formats.FORMATS else None
    )
    unknown_tools = tool_calls if tool_calls and tool_calls != "kimi_k3" else None
    if unknown_reasoning or unknown_tools:
        from .formats.external import validate_parser_names as validate_external

        validate_external(unknown_reasoning, unknown_tools)


class CompletionParser:
    """Per-request streaming state; kernel execution and token accounting stay outside."""

    def __init__(self, *, reasoning=None, tool_calls=None, request=None, tokenizer=None):
        self.reasoning = self.tools = None
        self.call_ids = {}
        if not reasoning and not tool_calls:
            return
        data = request or {}
        reasoning = (
            reasoning.lower() if reasoning and data.get("separate_reasoning", True) else None
        )
        tool_calls = tool_calls if data.get("tools") and data.get("tool_choice") != "none" else None
        if reasoning in reasoning_formats.FORMATS:
            self.reasoning = reasoning_formats.ReasoningParser(reasoning, data)
            reasoning = None
        if tool_calls == "kimi_k3":
            self.tools = ToolDetector()
            tool_calls = None
        if reasoning or tool_calls:
            from .formats.external import create

            extra_reasoning, extra_tools = create(reasoning, tool_calls, data, tokenizer)
            self.reasoning = self.reasoning or extra_reasoning
            self.tools = self.tools or extra_tools

    def finish_reason(self, reason=None):
        if reason in (None, "stop"):
            return "tool_calls" if self.call_ids else "stop"
        return reason

    def _content(self, reasoning, text, *, streaming):
        calls = []
        if self.tools is not None:
            parse = self.tools.parse_stream_chunk if streaming else self.tools.parse_non_stream
            text, calls = parse(text or "")
        result = {}
        if reasoning:
            result["reasoning_content"] = reasoning
        if text or not streaming:
            result["content"] = text or (None if calls else "")
        if calls:
            output = []
            for call in calls:
                first = call.tool_index not in self.call_ids
                if first:
                    self.call_ids[call.tool_index] = "call_" + secrets.token_hex(12)
                function = {"arguments": call.parameters}
                if call.name:
                    function["name"] = call.name
                item = {"function": function}
                if first or not streaming:
                    item.update(id=self.call_ids[call.tool_index], type="function")
                if streaming:
                    item["index"] = call.tool_index
                output.append(item)
            result["tool_calls"] = output
        return result

    def parse_chunk(self, text):
        reasoning, content = (
            self.reasoning.parse_stream_chunk(text) if self.reasoning else ("", text)
        )
        return self._content(reasoning, content, streaming=True)

    def parse_non_stream(self, text):
        reasoning, content = self.reasoning.parse_non_stream(text) if self.reasoning else ("", text)
        return self._content(reasoning, content, streaming=False)

    def finish(self):
        reasoning, content = self.reasoning.parse_stream_end() if self.reasoning else ("", "")
        result = self._content(reasoning, content, streaming=True)
        if self.tools:
            tail, _ = self.tools.parse_stream_end()
            if tail:
                result["content"] = (result.get("content") or "") + tail
        return result
