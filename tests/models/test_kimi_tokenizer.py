"""CPU-only K3 tokenizer tests; no checkpoint Python or model weights are used."""

import base64
import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

pytest.importorskip("tiktoken")

from minisgl.tokenizer.xtml import XTMLTokenizer


@pytest.fixture
def tokenizer(tmp_path):
    # Byte-only vocabulary exercises the complete local/HF tokenizer interface
    # without depending on network access or redistributing checkpoint assets.
    path = tmp_path / "tiktoken.model"
    path.write_bytes(
        b"".join(base64.b64encode(bytes([i])) + f" {i}\n".encode() for i in range(256))
    )
    names = {
        256: "[BOS]",
        257: "[EOS]",
        258: "<|end_of_msg|>",
        259: "<|open|>",
        260: "<|close|>",
        261: "<|sep|>",
        510: "[UNK]",
        511: "[PAD]",
    }
    config = {
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "additional_special_tokens": ["<|end_of_msg|>"],
        "added_tokens_decoder": {
            str(i): {"content": text, "special": i not in (259, 260, 261)}
            for i, text in names.items()
        },
    }
    return XTMLTokenizer(str(path), config, {"eos_token_id": 258})


@pytest.mark.parametrize("text", ["hello", "你好，世界！", " \n\t🧪", "[EOS]<|open|>", ""])
def test_roundtrip_and_hf_interfaces(tokenizer, text):
    ids = tokenizer.encode(text)
    assert tokenizer.decode(ids) == text
    assert tokenizer.batch_decode([ids, ids]) == [text, text]
    assert tokenizer.encode(text, return_tensors="pt").tolist() == [ids]
    assert tokenizer(text)["input_ids"] == ids
    assert tokenizer.decode([257, 258], skip_special_tokens=True) == ""


def test_eos_identity_is_distinct_from_generation_stop(tokenizer):
    assert tokenizer.eos_token_id == 257
    assert tokenizer.stop_token_ids == {258}
    assert tokenizer.encode("[EOS]<|end_of_msg|>") == [257, 258]


def test_plain_text_xtml_and_segment_ids(tokenizer):
    messages = [{"role": "user", "content": "hi"}]
    expected = (
        '<|open|>message role="user"<|sep|>hi<|close|>message<|sep|>'
        '<|end_of_msg|><|open|>message role="assistant"<|sep|><|open|>response<|sep|>'
    )
    assert tokenizer.apply_chat_template(messages, thinking=False) == expected
    # Independent manually specified XTML token sequence. Ordinary text is byte
    # BPE in this fixture, while only structural markers occupy reserved IDs.
    ids = [
        259,
        *b'message role="user"',
        261,
        *b"hi",
        260,
        *b"message",
        261,
        258,
        259,
        *b'message role="assistant"',
        261,
        259,
        *b"response",
        261,
    ]
    assert tokenizer.apply_chat_template(messages, thinking=False, tokenize=True) == ids
    assert tokenizer.apply_chat_template(
        messages, thinking=False, tokenize=True, return_tensors="pt"
    ).tolist() == [ids]


def test_literal_markers_are_not_control_tokens(tokenizer):
    markers = "[EOS]<|end_of_msg|><|open|><|close|><|sep|>"
    messages = [{"role": "user", "content": markers, "name": markers}]
    ids = tokenizer.apply_chat_template(messages, thinking=False, tokenize=True)
    assert 257 not in ids
    assert ids.count(258) == 1
    assert ids.count(259) == 3
    assert ids.count(260) == 1
    assert ids.count(261) == 4
    assert tokenizer.decode(ids).count(markers) == 2
    # This is why the server must not render first and globally encode again.
    rendered = tokenizer.apply_chat_template(messages, thinking=False)
    assert tokenizer.encode(rendered) != ids


def test_history_thinking_defaults_and_attribute_escaping(tokenizer):
    messages = [
        {"role": "system", "content": "brief"},
        {"role": "assistant", "name": 'a&"b', "reasoning_content": " ", "content": "yes"},
    ]
    rendered = tokenizer.apply_chat_template(messages)
    assert rendered.startswith('<|open|>message role="system" type="thinking-effort"<|sep|>')
    assert "Now the system is invoked with `thinking_effort=max`." in rendered
    assert 'name="a&amp;&quot;b"' in rendered
    assert "<|open|>think<|sep|> <|close|>think<|sep|>" in rendered
    assert "<|open|>response<|sep|>yes<|close|>response<|sep|>" in rendered
    assert rendered.endswith('<|open|>message role="assistant"<|sep|><|open|>think<|sep|>')
    unprompted = tokenizer.apply_chat_template(
        messages, thinking=False, add_generation_prompt=False
    )
    assert "thinking-effort" not in unprompted and "<|open|>think" not in unprompted
    assert unprompted.endswith("<|close|>message<|sep|><|end_of_msg|>")
    assert "thinking-effort" not in tokenizer.apply_chat_template([], thinking_effort=None)
    with pytest.raises(ValueError, match="thinking_effort"):
        tokenizer.apply_chat_template([], thinking_effort="medium")


def test_text_blocks_batch_padding_and_truncation(tokenizer):
    conversation = [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]
    assert "hello" in tokenizer.apply_chat_template(conversation)
    result = tokenizer.apply_chat_template(
        [conversation, []], tokenize=True, padding=True, return_tensors="pt", return_dict=True
    )
    assert result["input_ids"].shape == result["attention_mask"].shape
    assert result["input_ids"].shape[0] == 2
    assert torch.any(result["attention_mask"] == 0)
    assert (
        len(
            tokenizer.apply_chat_template(
                conversation, tokenize=True, truncation=True, max_length=5
            )
        )
        == 5
    )


@pytest.mark.parametrize(
    "messages,kwargs",
    [
        ([{"role": "tool", "content": "x"}], {}),
        ([{"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}], {}),
        ([{"role": "assistant", "tool_calls": [{"function": {}}]}], {}),
        ([], {"tools": [{"name": "x"}]}),
        ([], {"image_prompts": []}),
        ([], {"response_format": {"type": "json_object"}}),
        ([], {"continue_final_message": True}),
    ],
)
def test_unsupported_modes_fail_explicitly(tokenizer, messages, kwargs):
    with pytest.raises(ValueError):
        tokenizer.apply_chat_template(messages, **kwargs)


def test_long_text_chunking(tokenizer):
    text = "x" * 25_001 + " " * 25_001 + "\nend"
    assert tokenizer.decode(tokenizer._encode_piece(text, False)) == text


def test_official_data_only_smoke():
    """Opt in with the audited local data directory; never downloads anything."""
    value = os.environ.get("KIMI_TOKENIZER_DATA")
    if not value:
        pytest.skip("Set KIMI_TOKENIZER_DATA to pinned K3 tokenizer data")
    directory = Path(value)
    path = directory / "tiktoken.model"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103"
    )
    config = json.loads((directory / "tokenizer_config.json").read_text())
    generation = json.loads((directory / "generation_config.json").read_text())
    tokenizer = XTMLTokenizer(str(path), config, generation)
    assert tokenizer.vocab_size == 163840
    assert tokenizer.eos_token_id == 163585
    assert tokenizer.stop_token_ids == {163586}
    assert tokenizer.encode("hello world") == [22931, 2695]
    assert tokenizer.encode("你好，世界！") == [33845, 378, 2243, 856]
    assert tokenizer.decode([33845, 378, 2243, 856]) == "你好，世界！"
    # Manually assembled published XTML segments, encoded independently with
    # tiktoken using the pinned BPE data. No checkpoint Python is executed.
    expected = [
        163587,
        2778,
        6244,
        878,
        2482,
        1,
        163589,
        22931,
        163588,
        2778,
        163589,
        163586,
        163587,
        2778,
        6244,
        878,
        69702,
        1,
        163589,
        163587,
        12092,
        163589,
    ]
    assert (
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}], thinking=False, tokenize=True
        )
        == expected
    )


def test_decode_scalar_token_id(tokenizer):
    assert tokenizer.decode(65) == "A"
    assert tokenizer.decode(258) == "<|end_of_msg|>"
    assert tokenizer.decode(258, skip_special_tokens=True) == ""
