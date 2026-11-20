# Distributed backends

`DistributedImpl` owns collective execution. Models use their parallel groups
and do not select a transport. Torch/PyNCCL and optional native providers
implement the same all-reduce, all-gather, capture and teardown interface.

## Native symmetric-memory collectives

Select the native MNNVL provider on a supported multicast fabric.

```bash
python -m minisgl --model-path /data/models/Kimi-K3 --tp-size 8 \
  --distributed-backend mnnvl \
  --distributed-provider minisgl.distributed.multimem.provider
```

Add the remaining model and serving arguments for the intended workload. This
provider is independent of the model and attention backend. Mini packages the upstream symmetric-memory allocator, topology checks,
algorithm selection and collective kernels. Unsupported all-reduce inputs and all-gather use Torch/NCCL
on the same ordered ranks. TVM FFI launches follow the current Torch stream,
including graph capture and subsequent pointer registration.


The built-in native provider leaves its workspace cap and tuned thresholds at
the upstream defaults (16 MiB for this revision, with environment overrides).
The generic factory's expected collective-size hint is separate from this
workspace budget. Larger inputs use the existing same-rank NCCL fallback;
architecture selection, graph registration and resource lifetime are unchanged.

The provider has two initialization steps:

1. Optional `prepare(tp_info, device)` runs before process-group creation.
2. `create_provider(tp_info, cpu_group, device_group, device, max_bytes)` runs
   after the bootstrap and device groups exist.

Existing external providers only need the second step. Mini verifies that the
native provider's CPU/device groups have matching coordinates and ordered ranks.
The provider probes peer access before starting the process groups. Workers serialize
the probe and share a cache keyed by local device UUIDs, CUDA and driver versions.
Set `CUDA_VISIBLE_DEVICES` to the local allocation so that the probe covers the
same devices as the communicator.

The packaged implementation follows upstream revision `baa9bef`. Eight-B300
component tests cover eager collectives, graph replay and provider lifetime.
The full-model EP2 and EP8 quality runs select MNNVL explicitly. The standalone
AgentX profile leaves TP transport on `auto`, which selects PyNCCL with this
server configuration. [Component checks](validation/native-mnnvl.json) ·
[Serving measurements](kimi-k3-performance.md)

## Collective resource lifetime

Torch/NCCL groups allocate collective resources before the engine measures
memory for cache capacity. Context groups are prepared only on member ranks;
Torch tensor groups and the native provider's NCCL fallback use the same helper.
This keeps late communicator allocation out of the memory pressure created by a
large prefill. It changes neither model precision nor the requested cache-memory
fraction. The [controlled qualification](validation/native-collective-bootstrap.json)
records the frozen R2 cold failure, successful 65,537-token warm control on all
eight GPUs, repeated native-provider graph checks, and CPU/Gloo coverage.

The frontend retains owned worker handles for its lifetime. If a scheduler or
tokenizer exits after readiness, it finishes pending responses with an error and
rejects new requests/readiness checks with HTTP 503. Backend teardown remains
scoped to those owned processes on normal shutdown or frontend failure.

The builtin native provider declares its SDK input-copy alignment (32 bytes on
Blackwell, 16 on earlier devices). The generic adapter materializes contiguous,
aligned payloads locally before capability selection, so different view offsets
or strides on peers do not select different collectives. Dtype is preserved.
The [alignment regression](validation/native-collective-alignment.json) records
947 CPU/Gloo passes and 256 exact eight-GPU eager/graph results with mixed
rank layouts, together with the retained native misaligned-view failure.
