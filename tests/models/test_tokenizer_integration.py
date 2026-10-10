"""CPU coverage for segmented chat and generation-stop integration."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.llm.llm import LLM, RequestStatus
from minisgl.message import DetokenizeMsg
from minisgl.scheduler.scheduler import Scheduler
from minisgl.tokenizer.detokenize import DetokenizeManager
from minisgl.tokenizer.tokenize import TokenizeManager
from minisgl.utils.hf import get_stop_token_ids


def tokenizer():
    return SimpleNamespace(
        eos_token_id=163585,
        stop_token_ids=frozenset({163586}),
        batch_decode=lambda batches: [
            "".join({1: "hello", 163585: "[EOS]", 163586: "<|end_of_msg|>"}[i] for i in ids)
            for ids in batches
        ],
    )


def test_generation_stops_are_distinct_from_tokenizer_identity():
    tok = tokenizer()
    assert get_stop_token_ids(tok) == {163586}
    assert tok.eos_token_id == 163585
    assert get_stop_token_ids(SimpleNamespace(eos_token_id=2)) == {2}
    assert get_stop_token_ids(SimpleNamespace(eos_token_id=None)) == set()
    assert get_stop_token_ids(SimpleNamespace(eos_token_id=[2, 3])) == {2, 3}


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_scheduler_obeys_generation_stop_ids(ignore_eos):
    scheduler = object.__new__(Scheduler)
    scheduler.stop_token_ids = get_stop_token_ids(tokenizer())
    scheduler.eos_token_id = 163585
    scheduler.finished_reqs = set()
    scheduler.cache_manager = SimpleNamespace(lazy_free_region=nullcontext)
    freed, removed, replies = [], [], []
    scheduler.decode_manager = SimpleNamespace(remove_req=removed.append)
    scheduler._free_req_resources = freed.append
    scheduler.send_result = replies.extend
    req = Req(torch.tensor([1]), 0, 0, 5, 7, SamplingParams(ignore_eos=ignore_eos), None)
    req.complete_one()  # The engine has processed this forward.
    batch = Batch([req], "decode")
    data = (
        SimpleNamespace(batch=batch),
        (None, torch.tensor([163586]), SimpleNamespace(synchronize=lambda: None)),
    )
    scheduler._process_last_data(data)
    assert replies[0].finished is not ignore_eos
    assert len(freed) == len(removed) == (0 if ignore_eos else 1)


def test_detokenizer_suppresses_generation_terminal_marker():
    manager = DetokenizeManager(tokenizer())
    assert manager.detokenize([DetokenizeMsg(7, 1, False)]) == ["hello"]
    assert manager.detokenize([DetokenizeMsg(7, 163586, True)]) == [""]
    assert manager.decode_map == {}


def test_offline_llm_suppresses_same_terminal_marker():
    llm = object.__new__(LLM)
    llm.stop_token_ids = get_stop_token_ids(tokenizer())
    llm.status_map = {7: RequestStatus(7, [1], [])}
    llm.offline_send_result([DetokenizeMsg(7, 1, False), DetokenizeMsg(7, 163586, True)])
    assert llm.status_map[7].output_ids == [1]


def test_chat_uses_segmented_tokenizer_output_directly():
    calls = []

    def chat(messages, **kwargs):
        calls.append(kwargs)
        return torch.tensor([[100, 101]])

    def encode(text, **kwargs):
        assert text == "raw text", "Chat rendering must not be globally re-encoded"
        return torch.tensor([[1, 2]])

    manager = TokenizeManager(SimpleNamespace(apply_chat_template=chat, encode=encode))
    actual = manager.tokenize(
        [SimpleNamespace(text=[{"role": "user", "content": "x"}]), SimpleNamespace(text="raw text")]
    )
    assert [row.tolist() for row in actual] == [[100, 101], [1, 2]]
    assert calls == [{"tokenize": True, "add_generation_prompt": True, "return_tensors": "pt"}]


def test_standard_fast_tokenizer_chat_does_not_double_bos():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[BOS]": 1, "hello": 2}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", 1)]
    )
    tok = PreTrainedTokenizerFast(tokenizer_object=backend, bos_token="[BOS]", unk_token="[UNK]")
    tok.chat_template = "{{ bos_token }} hello"
    output = TokenizeManager(tok).tokenize(
        [SimpleNamespace(text=[{"role": "user", "content": "hello"}])]
    )
    assert output[0].tolist() == [1, 2]


@pytest.mark.parametrize("remote", [False, True])
def test_kimi_loader_uses_only_pinned_data_files(tmp_path, monkeypatch, remote):
    import json
    import sys

    import minisgl.utils.hf as hf

    metadata = {
        "tokenizer_class": "TikTokenTokenizer",
        "_commit_hash": "a" * 40,
        "added_tokens_decoder": {"163586": {"content": "<|end_of_msg|>"}},
    }
    monkeypatch.setattr(hf, "get_tokenizer_config", lambda _: metadata)
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 163586}))
    constructor, downloads = [], []

    def make(*args):
        constructor.append(args)
        return "local tokenizer"

    monkeypatch.setitem(sys.modules, "minisgl.tokenizer.kimi", SimpleNamespace(KimiTokenizer=make))

    def download(*args, **kwargs):
        downloads.append((args, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(hf, "snapshot_download", download)
    monkeypatch.setattr(
        hf.AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("Remote code path")
    )
    model = "moonshotai/Kimi-K3" if remote else str(tmp_path)
    assert hf.load_tokenizer(model) == "local tokenizer"
    assert constructor == [(str(tmp_path / "tiktoken.model"), metadata, {"eos_token_id": 163586})]
    if remote:
        assert downloads == [
            (
                ("moonshotai/Kimi-K3",),
                {
                    "revision": "a" * 40,
                    "allow_patterns": [
                        "tokenizer_config.json",
                        "tiktoken.model",
                        "generation_config.json",
                    ],
                },
            )
        ]
    else:
        assert not downloads
