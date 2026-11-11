# Distributed backends

`DistributedImpl` owns collective execution. Models use their parallel groups
and do not select a transport. The default remains Torch/PyNCCL; optional
providers can use mature hardware-specific implementations behind the same
all-reduce, all-gather, capture and teardown interface.

## Native symmetric-memory collectives

Use SGLang's custom all-reduce v2 through the regular MNNVL provider:

```bash
python -m minisgl --model-path /data/models/Kimi-K3 --tp-size 8 \
  --distributed-backend mnnvl \
  --distributed-provider minisgl.distributed.sgl
```

Add the remaining model and serving arguments for the intended workload. This
provider is independent of the model and attention backend. SGLang supplies the
symmetric-memory allocation, topology checks, algorithm selection, multicast and
collective kernels. Unsupported all-reduce inputs and all-gather use Torch/NCCL
on the same ordered ranks. TVM FFI launches follow the current Torch stream,
including graph capture and subsequent pointer registration.

The provider has two initialization steps:

1. Optional `prepare(tp_info, device)` runs before process-group creation.
2. `create_provider(tp_info, cpu_group, device_group, device, max_bytes)` runs
   after the bootstrap and device groups exist.

Existing external providers only need the second step. Mini verifies that the
native provider's CPU/device groups have matching coordinates and ordered ranks.
The native adapter warms the SDK's actual P2P cache before distributed
initialization because the pinned SDK's cold check otherwise consults its
framework-wide coordinator. Local workers serialize this probe using a private
cache keyed by device UUIDs, CUDA and driver versions. The caller's cache setting
is restored even if the probe fails. No SGLang coordinator is initialized, and
the peer test is not skipped. Restrict `CUDA_VISIBLE_DEVICES` to the local worker
allocation; the bootstrap verifies this boundary.

The tested SDK is SGLang `baa9befa381511cd4d56f83980951e5a45fffc6d` with Torch
2.14.1+cu130 on eight B300 devices. The [qualification record](validation/native-mnnvl.json)
separates host lifecycle tests, real eager/graph collectives and pending model
validation. It establishes neither multi-node qualification nor a performance
improvement. Whole-model quality, cache lifecycle and a controlled serving
comparison remain necessary before enabling it for a performance candidate.

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
