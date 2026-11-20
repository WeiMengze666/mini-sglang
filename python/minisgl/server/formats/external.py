"""Optional adapter for additional formats from a separately installed parser suite."""


def validate_parser_names(reasoning=None, tool_calls=None):
    if reasoning:
        from sglang.srt.parser.reasoning_parser import ReasoningParser

        if reasoning.lower() not in ReasoningParser.DetectorMap:
            raise ValueError(f"Unknown reasoning parser: {reasoning}")
    if tool_calls:
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        if tool_calls not in FunctionCallParser.ToolCallParserEnum:
            raise ValueError(f"Unknown tool-call parser: {tool_calls}")


def create(reasoning, tool_calls, request, tokenizer):
    reasoning_parser = tools_parser = None
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
        reasoning_parser = ReasoningParser(**options)
        mode = reasoning_parser.detector.reasoning_default
        if mode in (
            "thinking",
            "enable_thinking",
            "explicit_thinking",
            "explicit_enable_thinking",
        ):
            toggle = mode.removeprefix("explicit_")
            value = (native_request.chat_template_kwargs or {}).get(toggle)
            enabled = value is True if mode.startswith("explicit_") else value is not False
            if reasoning_parser.detector.force_reasoning != enabled:
                reasoning_parser = ReasoningParser(**options, force_reasoning=enabled)
    if active_tools:
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        tools_parser = FunctionCallParser(
            native_request.tools,
            tool_calls,
            tokenizer=tokenizer,
            tool_choice=native_request.tool_choice,
        )

    return reasoning_parser, tools_parser
