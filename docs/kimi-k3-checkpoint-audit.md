# Released-checkpoint metadata audit

This audit checks safetensors headers, **not tensor values or inference**. It
never requests tensor payload ranges and does not open weight shards with a
safetensors tensor reader.

## Recorded result

Pinned model: `moonshotai/Kimi-K3` at
`f831ab66814297da540d832a5235f8e904f29d06`, checked on 2026-10-10.

- 96/96 shard headers, 497,220 stored tensors
- 249,756 normalized text tensors match declared model shapes
- 247,296 MXFP4 packed matrices with U8 E2M1 storage and U8 E8M0 scales
- Zero missing/unexpected keys, shape/packing errors, index/shard disagreement,
  dtype incompatibilities, or offset/byte-count/coverage errors
- 69 A_log tensors correctly narrow stored `[128]` to 96 heads
- 69 output-norm weights require the declared engine F32→BF16 conversion;
  these are reported casts, not hidden dtype mismatches
- Resulting parameter dtype counts: 437 F32 and 249,319 BF16
- Successful terminal HTTP 206 range-response bodies total **75,766,584 bytes**:
  75,765,816 header-JSON bytes plus 768 length-prefix bytes. This excludes
  unmetered historical redirect bodies; total HTTP-body consumption is unknown.
  Zero tensor payload ranges were requested.

The source config/index were verified against official responses at the same
revision: config Git-blob SHA1 matches its ETag, and the index SHA256 matches its
LFS X-Linked-ETag. The index alone is 59,764,096 bytes and was not downloaded
again for the header run. Exact hashes, per-shard hashes and accounting are
retained in the accompanying versioned validation evidence.

## Reproduce

Use the CPU test environment from [kimi-k3.md](kimi-k3.md), with locally supplied
config and index files from the pinned revision:

```sh
# Explicit network operation: only bounded prefixes and header JSON are fetched.
PYTHONPATH=python .venv/bin/python tests/models/audit_kimi_headers.py \
  /path/to/config.json /path/to/model.safetensors.index.json \
  /tmp/kimi-header-evidence --fetch

# Replay the same audit offline, checking cached hashes and provenance fields.
PYTHONPATH=python .venv/bin/python tests/models/audit_kimi_headers.py \
  /path/to/config.json /path/to/model.safetensors.index.json \
  /tmp/kimi-header-evidence

# Synthetic HTTP and malformed metadata checks; no network or weights.
PYTHONPATH=python .venv/bin/python -m pytest -o addopts='' \
  tests/models/test_kimi_headers.py -q
```

The reader caps each JSON header at 8 MiB and the total reserved header budget
at 128 MiB, uses at most four workers, follows only bounded redirects to verified
HF/CDN hosts and suppresses implicit netrc authentication. It rejects full
responses before reading their bodies and requires HTTP 206, exact Content-Range,
mandatory exact Content-Length, no content encoding and exact bounded reads.
Index paths and offline metadata/byte accounting are also validated.

### Historical reader policy

The complete 96-shard run used the first bounded reader, which requested one
extra byte as an overrun sentinel but received exactly the requested
prefix/header sizes. Requests also consumed redirect bodies internally despite
`allow_redirects=False`; those bodies were not metered. Its manifest therefore
records terminal-206 accounting only, not an aggregate network-byte guarantee.
The saved headers still support the complete offline shape audit.

The final reader dispatches directly through the Requests adapter, preserving
proxy/TLS settings while preventing automatic redirect-body consumption. It
requires exact Content-Length and performs exact reads without a sentinel.
Sixty synthetic safety/provenance tests pass, including a regression using a
real Requests session for the redirect-body issue.

A corrected fresh run stopped after an automatic-approval cancellation: 88/96
headers were saved, all matching the original hashes. Partial evidence records
71,638,720 saved prefix-plus-JSON bytes; interrupted aggregate HTTP-body counters
are unknown. No retry was performed. A complete 96-shard execution of the final
transport remains unverified; the historical full offline audit and partial
hardened-transport evidence are reported separately.

## What this does not prove

Headers describe shape, dtype and offsets; they do not establish valid numeric
contents, absence of NaNs, tensor integrity, dequantized accuracy, successful
full-model allocation/loading, tokenizer behavior or generation quality. There
is still no released-weight GPU run, unchanged full hybrid GPU-model logit
parity, or real distributed/EP/DCP/performance validation. The separate
[packed-storage audit](kimi-k3-capacity.md) now validates physical per-rank
resident tensors; the original audit continues to compare logical decoded shapes.
