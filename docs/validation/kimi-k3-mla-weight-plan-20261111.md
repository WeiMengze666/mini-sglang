# Native MLA weight lifecycle topic (2026-11-11)

MLA checkpoint Q/Qb and KVb weights remain TP-local. after_load prepares
native-layout local Wk/Wv. A generic layer-owned prepare_inference hook runs
after cache allocation and before graphs, preparing decode Q/Wk replicas across
the existing DCP group; Wv stays local. Prefill uses local queries. Packed or
non-16-bit Q keeps its normal Linear executor and gathers activations.
Arbitrary quantized KVb support is not claimed.

Necessary CPU checks passed **285 tests**, with **2 existing skips**, including
53 new layout/head-order/reload/query-fallback controls. Actual metadata confirms
**93 layers:24 MLA and69 KDA**, with TP8 Qb[2304,1536] and KVb[3072,512].
The derived pre-budget reduction is **1,642,070,016 bytes/rank**; late replicas add
**1,660,944,384 bytes/rank**. This changes allocation order and does not establish
a comparable final resident-memory saving.

Fixed SDK model_runner.py:1141 directly gathers contiguous [H,nope,latent] Wk;
forward_mla.py:503–507 consumes it directly in bmm, matching the late-replica
stride here. Local pre-budget Wk retains its native transposed layout.

GPU smoke, actual cache capacities and serving performance remain pending.
This isolated topic does not promote the default branch. The
[machine-readable record](kimi-k3-mla-weight-plan-20261111.json) binds exact
source, CPU tests, independent review and native-capacity evidence.
