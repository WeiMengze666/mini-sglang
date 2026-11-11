# Native scheduling alignment: CPU status

This isolated change builds on canonical V1 (c2ba9fb708470c03099dde7b8bfcc8fe12e34f13).
It adds a shared 4096-token generation forecast, reserves only the current
chunk's actual KV pages plus its applicable future allowance, and requeues
requests safely when decode needs more KV capacity.

Decode pressure first reclaims unlocked prefix-cache capacity. If that is
insufficient, retraction waits for the preceding forward's completion and host
output delivery, releases only privately owned allocated pages/state/table
ownership, and requeues the complete committed token history with the remaining
generation limit. Already delivered tokens and cached-prompt accounting survive
requeue. Impossible readmission ends with an explicit error.

The implementation uses the existing cache, pending-request, and scheduler
interfaces. It adds no model-specific scheduling path, numerical kernel,
manager, or CLI option. Existing canonical V1 configuration, kernels, cache
policies, validation artifacts, and historical documentation remain unchanged.

CPU evidence for the exact source bytes:
- Decode/retraction and associated lifecycle checks: 20 passed, 1 existing skip.
- Prefill/future-budget and associated checks: 51 passed, 0 skipped.
These suites overlap; their counts must not be added.
A fresh publication-checkout rerun of the three changed test files passed
20 tests with no skips (34.93 seconds); this also overlaps the suites above.
The CPU fixtures do not qualify CUDA stream/DMA behavior or multi-rank serving.

No GPU, whole-model quality, or serving-performance acceptance is claimed.
The canonical ignore_eos=true case uses forecast ratio 1. This change does not
add the native adaptive ratio tracker for other sampling configurations.
