# Attention layout

Engine owns one `Context.attn_backend`. The backend factory chooses an implementation
from the model's cache layout and the requested precision policy.

- `base.py`: the common interface and two independent compositions.
  `HybridAttnBackend` dispatches layer IDs using `is_kda_layer`;
  `PrefillDecodeAttnBackend` dispatches execution stages.
- `fa.py`, `fi.py`, `trtllm.py`: library backends.
  `fi.py` selects the FlashInfer wrapper appropriate to ordinary KV or latent KV.
- `kda/`: request-state execution, kernel registration, reference arithmetic,
  SGLang/FlashInfer recurrence adapters, and native short-convolution adapters.
- `mla/`: paged-cache execution, latent-attention reference arithmetic, and the
  adapters to FlashInfer's own MLA portfolios. `cutedsl.py` owns the dense
  physical-page/length buffers for native cyclic-DCP decode; the CSR wrapper
  remains a library fallback for other supported shapes.

Hybrid metadata keeps each child backend's metadata separate. The parent delegates
preparation, capture, replay, and cleanup to both children. KDA manages recurrent
state; MLA manages token-page plans. Neither child implements the other's algorithm.

Hybrid `auto` selects `sgl`, composing mature SGLang KDA and FlashInfer MLA.
Use `--attention-backend fi` to explicitly select the FlashInfer composition. `--bf16-precision strict`
selects the native SGLang KDA portfolio inside that backend; `default` uses FlashInfer's
KDA recurrence. The Torch reference composition is `--attention-backend torch`.
There is no separate Engine/Context KDA implementation setting.

An optional `--attention-provider module` exposes `register_attention_backends()`.
It can register a composition with any KDA kernel spec and MLA backend. Fused KDA
kernel specifications live in `kda/registry.py`; fused decode implementations stay
in their KDA kernel adapters. Communication fusion belongs to
`DistributedImpl`, outside attention and model definitions.

The FlashInfer MLA adapter does not implement an attention CUDA kernel. It supplies
compressed KV pages, absorbed queries, and capture-safe decode plans to the
library. Decode merges DCP latent partials before the value projection. Prefill
gathers cyclic KV history, selects TP-local heads, and calls the library's native
ragged attention. Unsupported
layouts use the portable reference. Prefill, eager decode and captured decode have
separate plan lifetimes.

`--attention-backend sgl` composes the mature SGLang KDA
kernels and FlashInfer MLA. Its KDA kernel spec can adapt projection dispatch,
recurrence, and complete one-token decode independently. Aligned skinny
projections use the SDK GEMM dispatch. Chunk prefill and packed recurrent decode
use the SDK directly; supported FP32-state decode also uses the SDK fused
convolution/KDA/gated-norm kernel after its own `covered()` check. BF16 state
retains the supported packed path. Weight-dependent plans are rebuilt after
loading. Explicit K,V <-> V,K conversion and owned update storage preserve
Mini's request state, snapshots and collectives.
There is no model-specific branch or new numerical kernel in this adapter.

The optional SDK is imported only on a supported CUDA execution path. CPU and
unsupported dimensions retain the portable reference. Integration is developed
against SGLang commit `baa9befa381511cd4d56f83980951e5a45fffc6d`; importing its
kernel package does not construct a SGLang scheduler or model runner. Component,
full-model numerical, and performance gates are separate.

Native SM100/SM103 decode with physical page size 64 and MLA512/64 uses
FlashInfer's `trtllm_batch_decode_with_kv_cache_mla(backend="cute-dsl")`, the
same public entry point used by the pinned SGLang DCP backend. The adapter
passes local and global sequence lengths separately. FP8 E4M3 KV storage also
requires FP8 absorbed queries with the official unit-scale NoPE conversion.
Empty local shards return neutral values/LSE for the distributed merge.

The allocator's logical page spans `page_size * dcp_size`; CLI `page_size` is
the physical page length on each rank. Metadata is prepared once per batch,
after physical-page relocation. Graph plans keep device buffers at stable
addresses and share a decode workspace; expanded prefill owns its workspace.

`--kv-cache-dtype` selects persistent token storage independently of model
activations, convolution history and `--mamba-ssm-dtype`. Supported FP8-KV MLA prefill follows the native FP8 Q/K/V
path with BF16 output. Kernel/precision/state semantics are recorded separately
from convergence performance and final model-quality acceptance.

KDA parameters and recurrent/convolution state keep their tensor-TP partition.
DCP expands MLA decode queries across cyclic KV shards. `ParallelLayout.mla_decode`
describes that expanded head partition; attention CP is a separate topology.
The scoped `context` communicator owns KV gathers and partial-readout all-to-all.
Each scope delegates to a `DistributedImpl`, joins graph capture, and releases
shared provider resources once.

Strict BF16 execution uses the optional SDK's batch-invariant operator policy,
the same policy selected by SGLang deterministic inference. Engine owns this
process-scoped policy before graph capture and restores any policy it enabled
on capture failure or shutdown. It also preserves an externally owned SDK mode.
Default execution and CPU references do not install these optional operators.
This policy includes native matrix multiplication and operator dispatch; setting
cuBLAS arithmetic flags alone does not select the same kernels.
