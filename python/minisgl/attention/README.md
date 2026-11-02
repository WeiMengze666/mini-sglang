# Attention layout

Engine owns one `Context.attn_backend`. The backend factory chooses an implementation
from the model's cache layout and the requested precision policy.

- `base.py`: the common interface and two independent compositions.
  `HybridAttnBackend` dispatches layer IDs using `is_kda_layer`;
  `PrefillDecodeAttnBackend` dispatches execution stages.
- `fa.py`, `fi.py`, `trtllm.py`: library backends.
  `fi.py` selects the FlashInfer wrapper appropriate to ordinary KV or latent KV.
- `kda/`: request-state execution, kernel registration, reference arithmetic,
  Triton/FlashInfer recurrence, and the short-convolution kernel.
- `mla/`: paged-cache execution, latent-attention reference arithmetic, and the
  adapter to FlashInfer's own `BatchMLAPagedAttentionWrapper`.

Hybrid metadata keeps each child backend's metadata separate. The parent delegates
preparation, capture, replay, and cleanup to both children. KDA manages recurrent
state; MLA manages token-page plans. Neither child implements the other's algorithm.

Use `--attention-backend fi` for FlashInfer serving. `--bf16-precision strict`
selects the strict KDA recurrence inside that backend; `default` uses FlashInfer's
KDA recurrence. The Torch reference composition is `--attention-backend torch`.
There is no separate Engine/Context KDA implementation setting.

An optional `--attention-provider module` exposes `register_attention_backends()`.
It can register a composition with any KDA kernel spec and MLA backend. Fused KDA
decode belongs to `kda/registry.py`; communication fusion belongs to
`DistributedImpl`, outside attention and model definitions.

The FlashInfer MLA adapter does not implement an attention CUDA kernel. It supplies
compressed KV pages, absorbed queries, interleaved DCP causal subsequences, and
capture-safe plans to the library, then applies the value projection. Unsupported
layouts use the portable reference. Prefill, eager decode and captured decode have
separate plan lifetimes.
