# Attention backends

The engine builds one attention backend from the model configuration. Ordinary attention layers call a QKV backend. Layers that own projections and recurrent state call a layer backend, which can fuse the projections and state update. Both share the metadata and CUDA Graph lifecycle in `base.py`.

| Module | Responsibility |
| --- | --- |
| `factory.py` | Resolve the requested implementation and construct the backend. An optional attention provider can register another implementation. |
| `flashattention.py`, `flashinfer.py`, `trtllm.py` | Run ordinary MHA or GQA with the selected library. A comma-separated pair selects prefill and decode independently. |
| `base.py` | `HybridAttnBackend` selects KDA or MLA using the model's layer predicate. It passes the shared query offsets and the corresponding child metadata to each layer. |
| `kda/` | Prepare request state indices and prefix checkpoints, and select a recurrence kernel for each phase. The cache owns the working and checkpoint slots. |
| `mla/` | Plan latent attention over paged KV, pack the inputs for native kernels, and merge DCP partials before the value projection. |

For Kimi-K3 on B300, `--attention-backend auto` selects the following kernels. The server logs the resolved implementations at startup.

| Layer | Prefill | Decode |
| --- | --- | --- |
| KDA | FlashInfer CuTe DSL for supported chunk shapes | Indexed Triton recurrence for BF16 state |
| MLA | FlashInfer ragged attention and paged prefix reads | FlashInfer paged MLA with DCP |

The KDA adapter can dispatch a fused CUDA decode for supported FP32 state and convolution layouts. Single-token or unbounded-gate prefill has a Triton path. `--attention-backend torch` selects the hybrid reference implementation.

`--kv-cache-dtype` controls latent KV storage. `--mamba-ssm-dtype` controls the KDA state. KDA heads follow tensor parallelism. MLA's DCP group owns the cyclic KV pages and exchanges partial attention outputs and their log-sum-exp values. The communication group and its resources live in `distributed/`.

See the [native kernel source map](../kernel/licenses/UPSTREAM.txt) and the [serving configuration](../../../docs/kimi-k3.md).
