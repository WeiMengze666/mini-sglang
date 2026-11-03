"""Optional native format parsers behind a model-independent completion interface."""

from __future__ import annotations

import secrets


def validate_parser_names(reasoning=None, tool_calls=None):
    if reasoning:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        if reasoning.lower() not in ReasoningParser.DetectorMap:
            raise ValueError(f"Unknown reasoning parser: {reasoning}")
    if tool_calls:
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        if tool_calls not in FunctionCallParser.ToolCallParserEnum:
            raise ValueError(f"Unknown tool-call parser: {tool_calls}")


class CompletionParser:
    """Per-request streaming state; kernel execution and token accounting stay outside."""

    def __init__(self, *, reasoning=None, tool_calls=None, request=None, tokenizer=None):
        self.reasoning = self.tools = None
        self.call_ids = {}
        if not reasoning and not tool_calls:
            return
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest

        data = dict(request or {})
        if not data.get("messages"):
            data["messages"] = [{"role": "user", "content": data.get("prompt") or ""}]
        native_request = ChatCompletionRequest.model_validate(data)
        active_tools = bool(
            tool_calls and native_request.tools and native_request.tool_choice != "none"
        )
        if reasoning and native_request.separate_reasoning:
            from sglang.srt.parser.reasoning_parser import ReasoningParser

            options = {
                "model_type": reasoning,
                "request": native_request,
                "tokenizer": tokenizer,
                "stream_reasoning": native_request.stream_reasoning,
                "tool_call_parser_active": active_tools,
            }
            self.reasoning = ReasoningParser(**options)
            mode = self.reasoning.detector.reasoning_default
            if mode in (
                "thinking",
                "enable_thinking",
                "explicit_thinking",
                "explicit_enable_thinking",
            ):
                toggle = mode.removeprefix("explicit_")
                value = (native_request.chat_template_kwargs or {}).get(toggle)
                enabled = value is True if mode.startswith("explicit_") else value is not False
                if self.reasoning.detector.force_reasoning != enabled:
                    self.reasoning = ReasoningParser(**options, force_reasoning=enabled)
        if active_tools:
            from sglang.srt.function_call.function_call_parser import FunctionCallParser

            self.tools = FunctionCallParser(
                native_request.tools,
                tool_calls,
                tokenizer=tokenizer,
                tool_choice=native_request.tool_choice,
            )

    @property
    def finish_reason(self):
        return "tool_calls" if self.call_ids else "stop"

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
        return self._content(reasoning, content, streaming=True)
