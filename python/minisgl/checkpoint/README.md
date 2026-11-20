# Checkpoint loading

A model selects a checkpoint format with its `checkpoint_type` class attribute. The default [Hugging Face format](huggingface.py) serves Llama and Qwen. K3 supplies its [text checkpoint format](../models/kimi_k3/checkpoint.py). Both engine loading modes consume the same stream of named local tensors.

| Operation | Owner |
| --- | --- |
| Resolve a snapshot, validate its index and prefetch files | [Loader](loader.py) and [safetensors reader](safetensors.py) |
| Translate or ignore an external key | The format’s `normalize` method |
| Select experts owned by this rank | `Checkpoint.owns` uses the expert partition from `ParallelLayout` |
| Select a slice of a logical parameter | The format’s `shard` method calls the owning layer’s partition rule. `None` means replicated. |
| Read packed bytes and block scales, including pairs in different files | The reader translates the logical `ShardSpec` before materializing either tensor. |
| Combine separately stored parameters | The format’s `assemble` method can fuse Q/K/V or stack local experts before the model binds them. |

A model with the existing Hugging Face names and projection layout inherits the default format. For another checkpoint convention, override the steps that differ and select that format on the model class. Keep geometry with the layer that allocates the parameter, including the distinction between expert TP and ordinary TP. Execution backends prepare their native weight layouts after binding.
