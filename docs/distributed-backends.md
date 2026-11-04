# Distributed backends

`DistributedImpl` owns collective execution. Models use their parallel groups
and do not select a transport. The default remains Torch/PyNCCL; optional
providers can use mature hardware-specific implementations behind the same
all-reduce, all-gather, capture and teardown interface.

## Native symmetric-memory collectives

Use SGLang's custom all-reduce v2 through the regular MNNVL provider:

```bash
python -m minisgl --model /data/models/Kimi-K3 --tp-size 8 \
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
