from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from minisgl.core import SamplingParams
from minisgl.message import BaseTokenizerMsg, TokenizeMsg
from minisgl.server.parsers import CompletionParser
from minisgl.tokenizer.tokenize import TokenizeManager
from minisgl.utils import get_stop_token_ids


@pytest.mark.parametrize("trusted", [False, True])
def test_tokenizer_trust_policy_preserves_corpus_eos(monkeypatch, trusted):
    import minisgl.utils.hf as hf
    from transformers import GenerationConfig

    tokenizer = SimpleNamespace(eos_token_id=3, chat_template="template")
    calls = []
    monkeypatch.setattr(hf, "get_tokenizer_config", lambda _: {"_commit_hash": "revision"})
    monkeypatch.setattr(
        hf.AutoTokenizer, "from_pretrained", lambda path, **kw: (calls.append(kw), tokenizer)[1]
    )
    monkeypatch.setattr(
        GenerationConfig, "from_pretrained", lambda *a, **k: SimpleNamespace(eos_token_id=[7, 8])
    )
    actual = hf.load_tokenizer("fixture", trust_remote_code=trusted)
    assert actual.eos_token_id == 3
    assert get_stop_token_ids(actual) == (frozenset([7, 8]) if trusted else frozenset([3]))
    assert calls == ([{"trust_remote_code": True, "revision": "revision"}] if trusted else [{}])


def test_chat_options_survive_transport_without_overriding_tensor_contract():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "read",
                "parameters": {
                    "type": "object",
                    "properties": {"__type__": {"type": "string"}},
                },
            },
        }
    ]
    options = {"tools": tools, "thinking": False, "tokenize": False, "return_tensors": "invalid"}
    msg = TokenizeMsg(2, [{"role": "user", "content": "hi"}], SamplingParams(), options)
    restored = BaseTokenizerMsg.decoder(BaseTokenizerMsg.encoder(msg))
    assert restored.chat_template_kwargs == options
    seen = []

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            seen.append((messages, kwargs))
            return torch.tensor([[1, 2, 3]])

    (ids,) = TokenizeManager(Tokenizer()).tokenize([restored])
    assert ids.dtype == torch.int32 and ids.tolist() == [1, 2, 3]
    assert seen[0][1] == {
        **options,
        "tokenize": True,
        "return_tensors": "pt",
        "add_generation_prompt": True,
        "return_dict": False,
    }


def test_default_completion_parser_is_plain_text():
    parser = CompletionParser()
    assert parser.parse_chunk("hello") == {"content": "hello"}
    assert parser.parse_chunk("") == {}
    assert parser.parse_non_stream("hello") == {"content": "hello"}
    assert parser.finish() == {} and parser.finish_reason() == "stop"


def test_served_name_is_api_metadata(monkeypatch):
    import minisgl.server.api_server as api

    queue = SimpleNamespace(stop=lambda: None)
    state = api.FrontendManager(
        SimpleNamespace(model_path="/weights/model", served_model_name="org/model"), queue, queue
    )
    monkeypatch.setattr(api, "_GLOBAL_STATE", state)
    with TestClient(api.app) as client:
        result = client.get("/v1/models").json()["data"][0]
    assert result["id"] == "org/model" and result["root"] == "/weights/model"


def test_server_cli_retains_trust_and_served_name():
    from minisgl.server.args import parse_args

    config, shell = parse_args(
        [
            "--model-path",
            "fixture",
            "--dtype",
            "float32",
            "--trust-remote-code",
            "--served-model-name",
            "org/model",
            "--distributed-timeout",
            "1800",
        ]
    )
    assert config.trust_remote_code and config.served_model_name == "org/model"
    assert config.distributed_timeout == 1800
    assert config.reasoning_parser is None and config.tool_call_parser is None
    assert not shell
