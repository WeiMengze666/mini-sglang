"""Opt-in CPU integration against the installed native parser/tokenizer providers."""

import json
import os
from pathlib import Path

import pytest
import torch
from minisgl.server.parsers import CompletionParser

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_NATIVE_PARSER_TESTS") != "1",
    reason="optional native parser CPU integration",
)
REQUEST = {"model": "fixture", "messages": [{"role": "user", "content": "test"}]}
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
            },
        },
    }
]
K3_RESPONSE = "<|close|>think<|sep|><|open|>response<|sep|>done<|close|>response<|sep|>"
K3_CALL = (
    '<|open|>tools<|sep|><|open|>call tool="lookup" index="1"<|sep|>'
    '<|open|>argument key="q" type="string"<|sep|>a & b<|close|>argument<|sep|>'
    "<|close|>call<|sep|><|close|>tools<|sep|>"
)


@pytest.mark.parametrize("chunk", [1, 7, 10000])
@pytest.mark.parametrize(
    "name,options,text,reason,content",
    [
        ("kimi_k3", {}, "work" + K3_RESPONSE, "work", "done"),
        ("kimi_k3", {"thinking": False}, "answer", "", "answer"),
        ("qwen3", {}, "work</think>done", "work", "done"),
    ],
)
def test_native_reasoning_stream_and_complete(name, options, text, reason, content, chunk):
    def make():
        return CompletionParser(
            reasoning=name, request={**REQUEST, "chat_template_kwargs": options}
        )

    result = make().parse_non_stream(text)
    assert result.get("reasoning_content", "") == reason
    assert result["content"] == content
    parser = make()
    pieces = [parser.parse_chunk(text[i : i + chunk]) for i in range(0, len(text), chunk)]
    pieces.append(parser.finish())
    assert "".join(p.get("reasoning_content", "") for p in pieces) == reason
    assert "".join(p.get("content") or "" for p in pieces) == content
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("chunk", [1, 7, 10000])
@pytest.mark.parametrize(
    "name,text",
    [
        ("kimi_k3", K3_CALL),
        ("qwen25", '<tool_call>\n{"name":"lookup","arguments":{"q":"a & b"}}\n</tool_call>'),
    ],
)
def test_native_tool_stream_and_complete(name, text, chunk):
    def make():
        return CompletionParser(tool_calls=name, request={**REQUEST, "tools": TOOLS})

    result = make().parse_non_stream(text)
    (call,) = result["tool_calls"]
    assert call["type"] == "function" and call["function"]["name"] == "lookup"
    assert json.loads(call["function"]["arguments"]) == {"q": "a & b"}
    parser = make()
    pieces = [parser.parse_chunk(text[i : i + chunk]) for i in range(0, len(text), chunk)]
    pieces.append(parser.finish())
    calls = [c for p in pieces for c in p.get("tool_calls", [])]
    assert {c["index"] for c in calls} == {0}
    assert sum("id" in c for c in calls) == 1
    assert "".join(c["function"].get("name", "") for c in calls) == "lookup"
    assert json.loads("".join(c["function"]["arguments"] for c in calls)) == {"q": "a & b"}
    assert parser.finish_reason == "tool_calls"
    assert not torch.cuda.is_initialized()


def test_reasoning_and_tool_states_are_per_request():
    first = CompletionParser(
        reasoning="kimi_k3", tool_calls="kimi_k3", request={**REQUEST, "tools": TOOLS}
    )
    second = CompletionParser(reasoning="kimi_k3", request=REQUEST)
    a = first.parse_chunk("first")
    b = second.parse_chunk("second")
    a.update(first.parse_chunk(K3_RESPONSE + K3_CALL))
    b.update(second.parse_chunk(K3_RESPONSE))
    assert a.get("tool_calls") and not b.get("tool_calls")
    assert first.finish_reason == "tool_calls" and second.finish_reason == "stop"
    assert first.reasoning is not second.reasoning


@pytest.mark.parametrize("thinking", [False, True])
def test_trusted_chat_tokenizer_matches_native_and_generation_eos(thinking):
    from minisgl.utils import get_stop_token_ids, load_tokenizer
    from sglang.srt.utils.hf_transformers_utils import get_tokenizer

    model = Path(os.environ.get("MINISGL_K3_MODEL", "/data/models/Kimi-K3"))
    if not model.is_dir():
        pytest.skip("local released checkpoint is required")
    messages = [
        {"role": "user", "content": "look up a & b"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": '{"q":"a & b"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        {"role": "user", "content": "Explain the result."},
    ]
    mini = load_tokenizer(str(model), trust_remote_code=True)
    native = get_tokenizer(str(model), trust_remote_code=True)
    options = {
        "tools": TOOLS,
        "thinking": thinking,
        "tokenize": True,
        "add_generation_prompt": True,
    }
    assert mini.apply_chat_template(messages, **options) == native.apply_chat_template(
        messages, **options
    )
    generation = json.loads((model / "generation_config.json").read_text())["eos_token_id"]
    expected = frozenset([generation] if isinstance(generation, int) else generation)
    assert get_stop_token_ids(mini) == expected
    assert mini.eos_token_id == native.eos_token_id
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("stream", [False, True])
def test_chat_api_composes_native_parsers_and_preserves_usage(monkeypatch, stream):
    from types import SimpleNamespace

    import minisgl.server.api_server as api
    from fastapi.testclient import TestClient
    from minisgl.message import UserReply

    config = SimpleNamespace(
        model_path="fixture",
        served_model_name="org/model",
        reasoning_parser="kimi_k3",
        tool_call_parser="kimi_k3",
    )
    queue = SimpleNamespace(stop=lambda: None)
    state = api.FrontendManager(config, queue, queue, initialized=True)
    state.__dict__["tokenizer"] = None
    seen = []

    async def feed(msg):
        seen.append(msg)
        state.ack_map[msg.uid] = [
            UserReply(msg.uid, "work" + K3_RESPONSE, False, prompt_tokens=7, completion_tokens=1),
            UserReply(msg.uid, K3_CALL, True, prompt_tokens=7, completion_tokens=2),
        ]
        state.event_map[msg.uid].set()

    state.send_one = feed
    monkeypatch.setattr(api, "_GLOBAL_STATE", state)
    with TestClient(api.app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                **REQUEST,
                "tools": TOOLS,
                "stream": stream,
                "stream_options": {"include_usage": True},
            },
        )
    assert response.status_code == 200, response.text
    assert seen[0].chat_template_kwargs["tools"] == TOOLS
    if stream:
        chunks = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and "[DONE]" not in line
        ]
        assert all(c["model"] == "org/model" for c in chunks)
        deltas = [c["choices"][0]["delta"] for c in chunks if c["choices"]]
        assert "".join(d.get("reasoning_content", "") for d in deltas) == "work"
        assert "".join(d.get("content") or "" for d in deltas) == "done"
        assert chunks[-2]["choices"][0]["finish_reason"] == "tool_calls"
        usage = chunks[-1]["usage"]
    else:
        result = response.json()
        assert result["choices"][0]["message"]["reasoning_content"] == "work"
        assert result["choices"][0]["message"]["content"] == "done"
        assert result["choices"][0]["finish_reason"] == "tool_calls"
        usage = result["usage"]
    assert usage == {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
