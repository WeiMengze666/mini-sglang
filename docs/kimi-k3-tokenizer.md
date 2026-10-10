# Data-only K3 tokenizer verification

The released tokenizer requires custom Python when loaded through AutoTokenizer.
Mini instead reads local BPE/config data with a small `tiktoken` adapter. Remote
asset resolution is pinned to the commit reported by tokenizer metadata and
allows only `tokenizer_config.json`, `tiktoken.model`, and `generation_config.json`.
It neither downloads nor executes checkpoint Python during tokenizer loading.

The local formatter supports basic system/user/assistant text history, names,
reasoning text and thinking options. Its default is thinking enabled with maximum
effort, following the published formatter. Tools, images and other unsupported
formatting options raise errors. Structural XTML and ordinary text are encoded
as distinct segments, including individual attribute segments; rendering a
string and encoding it again would change boundaries and permit literal control
markers in user text. The serving tokenizer now consumes tokenized chat directly.

The tokenizer corpus EOS is `[EOS]` (163585); generation terminates at
`<|end_of_msg|>` (163586). Scheduler, streaming detokenizer and offline LLM share
that generation stop set. Standard tokenizers still fall back to their EOS.
Generated XTML thinking/response structure is not parsed into separate API
fields; this is basic text tokenization, not complete K3 chat-protocol support.

## Evidence and reproducibility

Official assets/source were inspected at revision
`f831ab66814297da540d832a5235f8e904f29d06` of
[moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/tree/f831ab66814297da540d832a5235f8e904f29d06).
Downloaded Python was read as source, never executed. The 2,795,286-byte BPE file
has SHA256 `b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103`.
The optional real-data test verifies that digest and compares token IDs against
an independently constructed tiktoken encoding and manually specified segments.
It does not execute the official formatter as an independent oracle.

```sh
# Synthetic byte vocabulary and serving regressions; no network.
PYTHONPATH=python .venv/bin/python -m pytest -o addopts='' \
  tests/models/test_kimi_tokenizer.py tests/models/test_tokenizer_integration.py -q

# Optional: locally supplied official tokenizer data, including the BPE file.
KIMI_TOKENIZER_DATA=/path/to/tokenizer-data PYTHONPATH=python \
  .venv/bin/python -m pytest -o addopts='' tests/models/test_kimi_tokenizer.py -q
```

Tests cover multilingual byte-stream round trips, HF batching/padding/scalar
decode, exact segmented XTML, marker escaping, thinking history, stop IDs,
unsupported-mode rejection, snapshot selection and a standard fast-tokenizer
chat regression. These establish tokenization behavior, not model generation
quality or end-to-end GPU serving correctness.
