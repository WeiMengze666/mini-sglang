# K3 capacity and tensor-parallel correctness path

The original eager baseline decoded every MXFP4 matrix into persistent dense
storage. This revision keeps packed bytes/scales resident and adds basic tensor
parallelism. These changes make a plausible capacity plan possible; they do not
establish GPU execution, generation quality or acceptable performance.

## Storage and loading

- Configuration selectors and ignore rules choose packed linears. K3's router,
  latent down/up projections and attention-residual score projections remain
  native dense, matching their actual checkpoint headers and upstream roles.
- Only weight-only, static symmetric E2M1/E8M0 MXFP4 with 32-column scale blocks
  is supported. Unsupported activation/KV quantization, auxiliary schemes and
  MLA `kv_b_proj` quantization fail explicitly.
- Packed linears retain U8 weights/scales. Each forward decodes at most 128 output
  rows, additionally capped at roughly one million elements, then discards that
  dense temporary. Temporary byte use is larger than the element count because
  decoding also builds integer codes and FP32 intermediates. No fused GEMM or
  throughput claim is made.
- Safetensors slices are selected before materialization. Row-parallel packed
  input slices must align with 32-column blocks; weight bytes and scale columns
  are sliced together, including cross-shard pairs. KDA `A_log` is narrowed and
  sliced using its live head count, including legacy four-dimensional storage.
- The engine assigns streamed tensors immediately into a meta-constructed model,
  validating names, duplicates, shapes, dtype and missing entries. Floating
  checkpoint tensors are cast to their declared runtime dtype; U8 storage is
  never implicitly cast. A failed load leaves an unusable incomplete model.
- Remote loading first resolves data-only metadata and pins the selected weight
  shards to that snapshot. Vision-only/MTP-only shards are excluded when an index
  exists. All selected shard files still need local disk space; TP does not mean
  a rank downloads only its byte ranges. No full checkpoint was downloaded here.

The released-header/meta audit found the following **model-tensor-only** bytes
per rank (BF16 native weights, FP32 declared special parameters):

| TP size | Bytes/rank | GiB/rank |
|---|---:|---:|
| 1 | 1,559,965,580,416 | 1,452.831 |
| 2 | 785,731,856,576 | 731.770 |
| 4 | 398,614,994,656 | 371.239 |
| 8 | 205,056,563,696 | 190.974 |

All configurations match 497,052 physical tensor keys and 247,296 packed matrices,
with zero key/shape/packed-dtype errors. The 69 previously documented F32 KDA
output-norm checkpoint weights are cast to declared BF16 storage. TP8 retains
11,498,132,736 bytes of replicated tensors per rank; its dense-equivalent total
would be 704,934,762,992 bytes per rank. These are static counts, not observed
GPU memory use or a hardware recommendation.

## TP layout

KDA and MLA attention heads, dense/shared FFN intermediate channels, and every
routed expert's intermediate channels are split evenly. Each rank still owns
all expert identities; this is **not expert parallelism**. Hidden states,
routers and low-rank/latent projections are replicated. Vocabulary rows shard
when divisible and otherwise remain replicated. KDA convolution/recurrent
states are rank-local; the compressed MLA token cache is replicated.

Attention and dense/shared FFN row projections reduce over the same TP group.
The routed mixture has one FP32 reduction after all local selected experts and
before nonlinear latent normalization/up-projection. There are no sparse
per-expert collectives. Graphs, prefix reuse and overlap remain disabled; K3 uses
native collectives rather than the unvalidated custom PyNCCL buffer path.

Admission preflight uses the least-free TP rank and declared weights, fixed
request state and minimum/overridden token pages. It rejects an impossible rank
before loading weights. After loading, each rank retains its own initial/final
free-memory pairing and fixed-state cost. The engine computes
`floor(ratio * local_before) - (local_before - local_after) - local_fixed_state`,
then MIN-reduces this final budget. All ranks allocate the same number of pages,
including one dummy page, and explicit page overrides must fit that budget.

Reducing the two snapshots separately is not equivalent. At ratio 0.9, initial
free memory `[100000, 101000]` MiB, final free memory `[60000, 59000]` MiB and
1000 MiB fixed state would give 48000 MiB using separate minima, versus the safe
paired minimum 47900 MiB. Both spreads pass the existing 2 GiB imbalance guard.
This consumes 100 MiB of intended headroom; it does not by itself exceed the
measured physical free memory. With equal fixed costs and ratio in [0,1], the
excess is bounded by `ceil((1-ratio) * initial_spread)` (up to byte rounding),
approximately 204.8 MiB at ratio 0.9 and the maximum accepted 2 GiB spread.
CPU regressions mock the budget collective, including changed limiting ranks,
heterogeneous fixed costs, page overrides and the unchanged non-Kimi path.
They do not constitute a real distributed or GPU allocation test.

These snapshot calculations are still not a peak-memory guarantee: activations,
collective buffers, allocator overhead, Python metadata, packed-decode temporaries
and concurrent external allocations require additional headroom.

## Evidence

The packed-storage auditor reuses all 96 saved, hash-verified official headers.
It creates meta models at representative last ranks for TP1/2/4/8 and compares
physical names, U8 packing, sliced shapes and declared dtype casts. Equal
contiguous partitions have equal storage on every rank. No network, process
group or weight payload is involved:

```sh
PYTHONPATH=python .venv/bin/python tests/models/audit_kimi_packed_storage.py \
  /path/to/config.json /path/to/model.safetensors.index.json \
  /path/to/saved-header-evidence --output /tmp/kimi-packed-storage.json
```

CPU tests cover dense/packed FP32/BF16 equivalence, sliced loading, strict stream
assignment, malformed quantization, and independent MoE reduction math. Two
separately constructed ranks run through test-only in-memory collectives,
including combined packed-checkpoint TP2 prefill/chunk/decode versus a dense TP1
twin, cache-state checks and collective order/shape/dtype checks.

**This is not a real distributed execution pass.** Actual Gloo initialization
was attempted and rejected by the cloud runtime's socket policy with `Operation
not permitted`, including an approved loopback retry. The opt-in Gloo test
remains runnable on a socket-enabled host. CUDA/NCCL and full-checkpoint memory
behavior remain untested. EP/DCP and performance work must wait for those basic
runtime and accuracy gates.
