# MoE backends

A model supplies the router scores, expert weights and any shared branch. `factory.py` selects an executor using the checkpoint format, activation precision and device. The engine prepares communication resources before sizing the KV cache.

| Step | Owner | Inputs and outputs |
| --- | --- | --- |
| Select experts | `routing.py` | Turn router logits into expert IDs and weights. A native executor may fuse routing with activation quantization. |
| Dispatch and combine | `dispatch.py`, `ep.py` | Send token rows to the ranks that own their experts, then return weighted outputs in token order. The executor declares the payload layout. |
| Compute local experts | `experts/` | The Torch, Triton, FlashInfer and Marlin adapters own weight preparation and local computation. FlashInfer tactic selection lives alongside its executor. |
| Finish the layer | `epilogue.py` | Combine routed and shared outputs, perform any outstanding reduction and apply the model's projection or residual. |

`--moe-backend fused` selects local computation using the weight format. `marlin` uses packed MXFP4 weights and BF16 activations. `flashinfer-ep` adds expert exchange through the configured process groups. All three use the same routing semantics.

An executor may declare its input dtype. For example, a FlashInfer MXFP8 executor can quantize FP32 latent projections directly, while Marlin consumes BF16. The model's KV cache and recurrent-state dtypes are independent of this selection. See the [kernel provenance](../kernel/licenses/UPSTREAM.txt) and [distributed backends](../../../docs/distributed-backends.md).
