from types import SimpleNamespace

import pytest
import torch
from fastapi.testclient import TestClient
from minisgl.core import SamplingParams
from minisgl.message import (
    AbortMsg,
    BaseFrontendMsg,
    BatchFrontendMsg,
    BatchTokenizerMsg,
    DetokenizeMsg,
    TokenizeMsg,
    UserReply,
)


def test_tokenizer_counts_actual_ids_and_cleans_cancelled_usage(monkeypatch):
    import minisgl.tokenizer.server as server

    class Tokenizer:
        eos_token_id = 9

        def encode(self, text, **options):
            if text == "bad":
                raise ValueError("bad prompt")
            return torch.arange(len(text)).unsqueeze(0)

        def apply_chat_template(self, messages, **options):
            return torch.tensor([[9, 1, 2, 3, 8]])

        def batch_decode(self, values):
            return [
                "".join({1: "", 2: " b", 3: "x", 9: "<eos>"}.get(t, "") for t in row)
                for row in values
            ]

    params = SamplingParams(max_tokens=3)
    pending = iter(
        [
            TokenizeMsg(7, [{"role": "user", "content": "abc"}], params),
            TokenizeMsg(8, "xy", params),
            BatchTokenizerMsg(
                [
                    DetokenizeMsg(7, 1, False),
                    DetokenizeMsg(8, None, True),
                    DetokenizeMsg(7, 2, False),
                ]
            ),
            DetokenizeMsg(7, 9, True),
            TokenizeMsg(9, "zzzz", params),
            AbortMsg(9),
            DetokenizeMsg(9, 3, False),
            TokenizeMsg(7, "Q", params),
            DetokenizeMsg(7, 3, True),
            TokenizeMsg(10, "bad", params),
            DetokenizeMsg(10, 3, False),
        ]
    )
    sent = {"backend": [], "frontend": []}

    class Receive:
        def get(self):
            try:
                return next(pending)
            except StopIteration:
                raise KeyboardInterrupt

        def empty(self):
            return True

    monkeypatch.setattr(server, "load_tokenizer", lambda _: Tokenizer())
    monkeypatch.setattr(server, "ZmqPullQueue", lambda *a, **k: Receive())
    monkeypatch.setattr(
        server, "ZmqPushQueue", lambda addr, **k: SimpleNamespace(put=sent[addr].append)
    )
    server.tokenize_worker(
        tokenizer_path="fixture",
        addr="input",
        create=True,
        backend_addr="backend",
        frontend_addr="frontend",
        local_bs=1,
    )
    replies = [
        r
        for msg in sent["frontend"]
        for r in (msg.data if isinstance(msg, BatchFrontendMsg) else [msg])
    ]
    assert [(r.uid, r.prompt_tokens, r.completion_tokens) for r in replies] == [
        (7, 5, 1),
        (8, 2, 0),
        (7, 5, 2),
        (7, 5, 3),
        (7, 1, 1),
        (10, 0, 0),
    ]
    assert replies[0].incremental_output == ""  # A real token can decode to no text.
    assert replies[3].incremental_output == ""  # EOS still counts as a generated token.
    assert replies[-1].error == "bad prompt"
    assert not torch.cuda.is_initialized()


def test_usage_survives_message_serialization():
    reply = UserReply(4, "", True, prompt_tokens=7, completion_tokens=3)
    restored = BaseFrontendMsg.decoder(BaseFrontendMsg.encoder(reply))
    assert restored.usage == {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}


@pytest.mark.parametrize(
    "stream,include_usage,tokens",
    [
        (False, False, 3),
        (False, False, 0),
        (True, False, 3),
        (True, True, 3),
        (True, True, 0),
    ],
)
def test_chat_usage_counts_tokens_independently_of_text(monkeypatch, stream, include_usage, tokens):
    import json

    import minisgl.server.api_server as api

    queue = SimpleNamespace(stop=lambda: None)
    state = api.FrontendManager(
        SimpleNamespace(model_path="fixture"), queue, queue, initialized=True
    )

    async def feed(msg):
        state.ack_map[msg.uid] = (
            [
                UserReply(msg.uid, text, step == tokens, prompt_tokens=7, completion_tokens=step)
                for step, text in enumerate(["", "a", " b"][:tokens], 1)
            ]
            if tokens
            else [UserReply(msg.uid, "", True, prompt_tokens=7)]
        )
        state.event_map[msg.uid].set()

    state.send_one = feed
    monkeypatch.setattr(api, "_GLOBAL_STATE", state)
    payload = {"model": "fixture", "prompt": "p", "max_tokens": tokens, "stream": stream}
    if include_usage:
        payload["stream_options"] = {"include_usage": True}
    with TestClient(api.app) as client:
        response = client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        scrape = client.get("/metrics")
        assert scrape.status_code == 200
        assert "minisgl:completed_requests_total 1\n" in scrape.text
        assert "minisgl:completed_prompt_tokens_total 7\n" in scrape.text
        assert f"minisgl:completed_output_tokens_total {tokens}\n" in scrape.text
        assert "minisgl:active_requests 0\n" in scrape.text
        if stream:
            lines = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
            assert lines[-1] == "[DONE]"
            chunks = [json.loads(line) for line in lines[:-1]]
            assert all(c["object"] == "chat.completion.chunk" for c in chunks)
            measured = [c for c in chunks if "usage" in c]
            assert bool(measured) == include_usage
            if include_usage:
                assert len(measured) == 1 and measured[0]["choices"] == []
                result = measured[0]
            else:
                assert all(c["choices"] for c in chunks)
                return
        else:
            result = response.json()
        assert result["usage"] == {
            "prompt_tokens": 7,
            "completion_tokens": tokens,
            "total_tokens": 7 + tokens,
        }
        assert not state.ack_map and not state.event_map
