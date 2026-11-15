"""Opt-in real FI intermediate h/conv snapshots and restored continuation.

One GPU, two ragged requests and two KDA layers. No mocked numerical calls.
"""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="queued native FI checkpoint gate"
)

LAYERS, TABLES, FORKS, ENDS = (0, 1), (3, 1), (128, 256), (323, 389)
HIDDEN, HEADS, WIDTH, PREFIX = 64, 12, 128, 64


def _allocate_to(manager, req, end):
    old = getattr(req, "_allocated_pages", 0)
    new = (end + manager.page_size - 1) // manager.page_size
    if new > old:
        indices = manager._page_to_token(manager._allocate(new - old))
        manager.page_table[req.table_idx, old * manager.page_size : new * manager.page_size] = (
            indices
        )
    req._allocated_pages = new


def _layer(config, layer_id):
    from minisgl.layers.kda import KDALayer

    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("cuda"):
            layer = KDALayer(config)
    finally:
        torch.set_default_dtype(old_dtype)
    for value in layer.state_dict().values():
        value.zero_()
    # One nonzero per GEMM output makes projections exact dyadic sums even
    # when the seed-prefix and tracked full-prefill physical M differ.
    for op, shift in (
        (layer.q_proj, 1),
        (layer.k_proj, 3),
        (layer.v_proj, 5),
        (layer.g_proj, 7),
        (layer.o_proj, 9),
    ):
        rows = torch.arange(op.weight.shape[0], device="cuda")
        columns = (rows * (shift + 2) + shift + layer_id) % op.weight.shape[1]
        op.weight[rows, columns] = 0.125
    for conv in (layer.q_conv1d, layer.k_conv1d, layer.v_conv1d):
        conv.weight[:, 0, :] = torch.tensor([0.125, 0.0, 0.25, 1.0], device="cuda")
    layer.A_log.fill_(-1)
    layer.o_norm.weight.fill_(1)
    layer.after_load()
    return layer


def _tensor_metadata(value):
    return {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype)}


def _execute(backend, layers, states, reqs, packed, *, checkpoint=False):
    """Invoke the actual backend and preserve the ordinary all-layer lease protocol."""
    from minisgl.core import Batch

    batch = Batch(reqs, "prefill")
    if checkpoint:
        manager, order = checkpoint
        plans = manager.plan_checkpoints(batch, backend)
        assert len(plans) == len(reqs)
        batch.checkpoint_plan = tuple(plans[index] for index in order)
    backend.prepare_metadata(batch)  # Calls real prepare_checkpoints for nonempty plans.
    for req in reqs:
        states.begin_request(req.table_idx, req.uid, req.cached_len)
    outputs = {}
    for layer_id, layer in layers.items():
        outputs[layer_id] = backend.forward_layer(
            layer, layer_id, packed[layer_id], batch, backend.kvcache, None
        )
        assert outputs[layer_id].is_cuda
    for req in reqs:
        states.finish_request(req.table_idx, req.uid, req.device_len)
        req.cached_len = req.device_len
    return batch, outputs


def test_real_fi_interior_forks_keep_h_axes_conv_and_restored_continuation(
    monkeypatch, request, tmp_path
):
    assert os.environ.get("MINISGL_GPUQ") == "1", "GPUq ownership marker required"
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    assert torch.cuda.get_device_capability(device) in ((10, 0), (10, 3))
    import minisgl.core as core
    import minisgl.distributed.info as info
    import minisgl.distributed.layout as topology
    from minisgl.attention.config import KDAConfig
    from minisgl.attention.kda import sgl
    from minisgl.attention.kda.backend import KDAAttentionBackend
    from minisgl.attention.kda.registry import get_kda_kernel
    from minisgl.core import Req, SamplingParams
    from minisgl.distributed.info import DistributedInfo
    from minisgl.kvcache.config import HybridCacheConfig
    from minisgl.kvcache.recurrent_pool import RecurrentStatePool
    from minisgl.scheduler.cache import CacheManager
    from minisgl.scheduler.prefill import PrefillAdder

    monkeypatch.setattr(info, "_TP_INFO", DistributedInfo(0, 1))
    monkeypatch.setattr(topology, "_DCP_SIZE", 1)
    monkeypatch.setattr(topology, "_EP_SIZE", 1)
    config = KDAConfig(HIDDEN, HEADS, WIDTH, 4, gate_lower_bound=-5, full_rank_gate=True)
    layout = HybridCacheConfig(2, (), LAYERS, 1, 0, HEADS, WIDTH, 4)
    with torch.inference_mode():
        states = RecurrentStatePool(
            LAYERS,
            6,
            HEADS,
            WIDTH,
            4,
            torch.bfloat16,
            device,
            state_slot_capacity=24,
            state_dtype=torch.bfloat16,
        )
        for layer_id in LAYERS:
            for value in states.indexed_kda_state(layer_id):
                value.zero_()
        cache = SimpleNamespace(
            recurrent=states,
            device=device,
            arena=None,
            supports_overlap=False,
            release_request=states.release_request,
            release_staged=states.release_staged,
        )
        context = core.Context(page_size=PREFIX, bf16_precision="default")
        context.kv_cache = cache
        monkeypatch.setattr(core, "_GLOBAL_CTX", context)
        backend = KDAAttentionBackend(
            SimpleNamespace(cache_config=layout), get_kda_kernel("flashinfer"), cache=cache
        )
        backend.prepare_model(tuple((layer_id, config) for layer_id in LAYERS))
        quantum = backend.checkpoint_quantum()
        assert quantum == 64
        manager = CacheManager(
            128,
            PREFIX,
            torch.full((6, 512), -1, dtype=torch.int32, device=device),
            "hybrid",
            cache,
            checkpoint_quantum=quantum,
        )
        layers = {layer_id: _layer(config, layer_id) for layer_id in LAYERS}
        inputs = {}
        for layer_id in LAYERS:
            inputs[layer_id] = []
            for row, end in enumerate(ENDS):
                t = torch.arange(end, device=device)[:, None]
                c = torch.arange(HIDDEN, device=device)[None, :]
                inputs[layer_id].append(
                    (((t * (c % 7 + 1) + row * 17 + layer_id * 11) % 29 - 14) / 16).to(
                        torch.bfloat16
                    )
                )
        ids = [torch.arange(end, dtype=torch.int32) + row * 10000 for row, end in enumerate(ENDS)]

        def admit(row, uid, end):
            handle = manager.prefix_cache.match_prefix(ids[row][:-1]).cuda_handle
            table = TABLES[row]
            manager.lock(handle)
            manager.restore_state(table, uid, handle)
            req = Req(ids[row], table, handle.cached_len, 2, uid, SamplingParams(), handle)
            req.device_len = end
            if handle.cached_len:
                manager.page_table[table, : handle.cached_len].copy_(handle.get_matched_indices())
            req._allocated_pages = handle.cached_len // PREFIX
            _allocate_to(manager, req, end)
            return req

        # Seed an actual reusable state at64 and Full-KV-only hits at128/256.
        seed = [admit(row, 10 + row, PREFIX) for row in range(2)]
        packed = {
            layer: torch.cat([inputs[layer][row][:PREFIX] for row in range(2)]) for layer in LAYERS
        }
        _execute(backend, layers, states, seed, packed)
        for req in seed:
            manager.cache_req(req, finished=False)
        for row, req in enumerate(seed):
            req.device_len = FORKS[row]
            _allocate_to(manager, req, req.device_len)
        packed = {
            layer: torch.cat([inputs[layer][row][PREFIX : FORKS[row]] for row in range(2)])
            for layer in LAYERS
        }
        _execute(backend, layers, states, seed, packed)
        references = []
        for row, req in enumerate(seed):
            cp = states.snapshot(req.table_idx, req.uid, staged=True, expected_len=FORKS[row])
            states.adopt_staged(cp, finished=True)
            references.append(cp)  # independent native truncated-prefix reference
            manager.cache_req(req, finished=True, frontier=FORKS[row])
            manager.release_state(req.table_idx, req.uid)

        reqs = [admit(row, 100 + row, ENDS[row]) for row in range(2)]
        for row, req in enumerate(reqs):
            handle = manager.current_handle(req)
            assert (req.cached_len, handle.replay_until) == (PREFIX, FORKS[row])
            adder = PrefillAdder(1024, 0, manager, None)
            assert (
                adder._chunk_size(SimpleNamespace(input_len=ENDS[row]), handle, PREFIX)
                == ENDS[row] - PREFIX
            )
        packed = {
            layer: torch.cat([inputs[layer][row][PREFIX : ENDS[row]] for row in range(2)])
            for layer in LAYERS
        }
        raw = {}
        for layer_id, layer in layers.items():
            q, k, v, _, _ = sgl.project(layer, packed[layer_id])
            raw[layer_id] = torch.cat((q, k, v), -1)
        batch, _ = _execute(backend, layers, states, reqs, packed, checkpoint=(manager, (1, 0)))
        tracked = batch.attn_metadata.checkpoints
        checkpoints = {
            plan.table_idx: states.seal_checkpoint(plan) for plan in batch.checkpoint_plan
        }
        conv_expected, raw_checkpoints = {}, {}
        for row, req in enumerate(reqs):
            query_offset = 0 if row == 0 else ENDS[0] - PREFIX
            relative = FORKS[row] - PREFIX
            conv_expected[row] = torch.stack(
                [
                    raw[layer_id][query_offset + relative - 3 : query_offset + relative]
                    .view(3, 3, HEADS * WIDTH)
                    .permute(1, 2, 0)
                    for layer_id in LAYERS
                ]
            )
            cp = checkpoints[req.table_idx]
            captured = states._checkpoints[cp.slot]
            reference = states._checkpoints[references[row].slot]
            raw_checkpoints[row] = {
                "checkpoint_slot": cp.slot,
                "reference_slot": references[row].slot,
                "captured_conv": captured[0].cpu(),
                "captured_ssm_KV": captured[1].cpu(),
                "reference_conv": reference[0].cpu(),
                "reference_ssm_KV": reference[1].cpu(),
                "raw_projection_conv_oracle": conv_expected[row].cpu(),
                "captured_metadata": [_tensor_metadata(value) for value in captured],
                "reference_metadata": [_tensor_metadata(value) for value in reference],
            }
        checkpoint_path = tmp_path / "native-checkpoint-raw.pt"
        torch.save(
            {
                "scope": "Persisted BF16 K,V checkpoint after native FP32 V,K h conversion; no weights",
                "forks": FORKS,
                "ends": ENDS,
                "rows": tracked.rows,
                "cu_starts": tracked.cu_starts.cpu(),
                "destinations": tracked.destinations.cpu(),
                "conv_indices": tracked.conv_indices.cpu(),
                "tracking_metadata": {
                    name: _tensor_metadata(getattr(tracked, name))
                    for name in ("cu_starts", "destinations", "conv_indices", "track_rows")
                },
                "checkpoints": raw_checkpoints,
            },
            checkpoint_path,
        )
        request.node.user_properties.append(("native_checkpoint_raw", str(checkpoint_path)))
        assert tracked.rows == (1, 0) and tracked.cu_starts.tolist() == [0, 4, 9]
        assert tracked.destinations.tolist() == [1, -1, -1, -1, -1, -1, 0, -1, -1]
        for row, req in enumerate(reqs):
            cp = checkpoints[req.table_idx]
            saved_conv, saved_state = states._checkpoints[cp.slot]
            expected_conv, expected_state = states._checkpoints[references[row].slot]
            torch.testing.assert_close(saved_state, expected_state, rtol=0, atol=0)
            torch.testing.assert_close(saved_conv, expected_conv, rtol=0, atol=0)
            assert not torch.equal(expected_state, expected_state.transpose(-1, -2))
            torch.testing.assert_close(saved_conv, conv_expected[row], rtol=0, atol=0)
            manager.cache_req(req, finished=False, frontier=FORKS[row], checkpoint=cp)
            assert req.cached_len == ENDS[row]  # Capturing the fork never rewinds working state.

        # Restore each real tree checkpoint and its independent native prefix
        # reference into different physical working rows. Execute all4 lanes
        # together so each pair uses identical native shape/rounding.
        continuation, handles, lane_rows = [], [], (0, 1, 1, 0)
        lane_tables, from_tree = (0, 4, 2, 5), (True, False, True, False)
        for lane, (row, table, tree) in enumerate(zip(lane_rows, lane_tables, from_tree)):
            uid = 500 + lane
            handle = manager.prefix_cache.match_prefix(ids[row][: FORKS[row]]).cuda_handle
            assert handle.cached_len == FORKS[row]
            if tree:
                manager.lock(handle)
                manager.restore_state(table, uid, handle)
                handles.append(handle)
            else:
                states.restore(table, uid, references[row])
            continuation.append(Req(ids[row], table, FORKS[row], 2, uid, SamplingParams(), handle))
        packed = {
            layer: torch.cat([inputs[layer][row][FORKS[row] : ENDS[row]] for row in lane_rows])
            for layer in LAYERS
        }
        _, outputs = _execute(backend, layers, states, continuation, packed)
        lengths = [ENDS[row] - FORKS[row] for row in lane_rows]
        continuation_path = tmp_path / "native-continuation-raw.pt"
        torch.save(
            {
                "lane_rows": lane_rows,
                "lane_tables": lane_tables,
                "lengths": lengths,
                "from_tree": from_tree,
                "outputs": {layer: value.cpu() for layer, value in outputs.items()},
                "states": {
                    table: [value.cpu() for value in states._working(table)]
                    for table in lane_tables
                },
                "state_metadata": {
                    table: [_tensor_metadata(value) for value in states._working(table)]
                    for table in lane_tables
                },
            },
            continuation_path,
        )
        request.node.user_properties.append(("native_continuation_raw", str(continuation_path)))
        for layer_id, output in outputs.items():
            parts = output.split(lengths)
            torch.testing.assert_close(parts[0], parts[3], rtol=0, atol=0)
            torch.testing.assert_close(parts[2], parts[1], rtol=0, atol=0)
            assert bool(torch.isfinite(output).all()) and bool(output.ne(0).any())
        for left, right in ((0, 3), (2, 1)):
            a = states._working(lane_tables[left])
            b = states._working(lane_tables[right])
            for actual, expected in zip(a, b):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for lane, req in enumerate(continuation):
            if from_tree[lane]:
                manager.release_state(req.table_idx, req.uid)
            else:
                states.release_request(req.table_idx, req.uid)
        for handle in handles:
            manager.unlock(handle)
        for req in reqs:
            manager.discard_req(req)
        for cp in references:
            states.drop_checkpoint(cp)
        manager._free(manager.prefix_cache.evict(manager.prefix_cache.size_info.evictable_size))
        assert not states._pending and not states._checkpoints and states.is_idle
        manager.check_integrity()
        assert states.state_usage["free"] == states.slot_capacity
        torch.cuda.synchronize(device)
        request.node.user_properties.append(
            (
                "native_checkpoint_scope",
                f"FI indexed_prefill2requests/2layers; forks{FORKS}; destinations1,0; h+conv exact;4lane restored continuation exact",
            )
        )
