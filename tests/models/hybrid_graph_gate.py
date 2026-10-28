"""Queued Engine graph/eager equivalence with reusable hybrid cache layouts.

Create the bounded synthetic checkpoint on CPU with --create-fixture first;
run the numerical gate with torchrun under gpuq. This is not model accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig


def create_fixture(path):
    import run_kimi_runtime_gate as runtime

    config = runtime.tiny_config()
    config.hidden_size = config.vocab_size = 128
    config.intermediate_size = config.moe_intermediate_size = 128
    config.routed_expert_hidden_size = 128
    config.q_lora_rank = 32
    config.kv_lora_rank, config.qk_nope_head_dim = 512, 192
    config.qk_rope_head_dim, config.v_head_dim = 64, 128
    config.linear_attn_config["head_dim"] = 128
    runtime.tiny_config = lambda: config
    runtime.generate_fixtures(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--create-fixture", action="store_true")
    parser.add_argument("--unified", action="store_true")
    args = parser.parse_args()
    if args.create_fixture:
        create_fixture(args.fixture)
        return
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    engine = Engine(
        EngineConfig(
            str(args.fixture / "packed"),
            DistributedInfo(rank, world),
            torch.bfloat16,
            dcp_size=2,
            ep_size=world,
            attention_backend="recurrent-fi",
            mxfp4_backend="triton",
            cuda_graph_bs=[1, 2, 4],
            max_running_req=8,
            max_cached_states=2,
            page_size=8,
            max_seq_len_override=64,
            num_page_override=256,
            enable_unified_memory=args.unified,
            distributed_timeout=180,
        )
    )
    pool = engine.kv_cache
    assert set(engine.graph_runner.graph_map) == {1, 2, 4}
    assert all(owner is None for owner in pool.recurrent._owners)
    if pool.arena is not None:
        assert not pool.arena._states and not pool.arena._pages
    table = torch.arange(512, device=engine.device, dtype=torch.int32).view(8, 64)
    engine.page_table[:8, :64].copy_(table)
    uid = 0

    def run(slots, graphs):
        nonlocal uid
        uid += 10
        reqs = [
            Req(
                torch.arange(5, 12 + slot, dtype=torch.int32),
                slot,
                0,
                8,
                uid + slot,
                SamplingParams(),
                None,
            )
            for slot in slots
        ]
        if pool.arena is not None:
            pages = sorted(
                {
                    (req.table_idx * 64 + pos) // 8
                    for req in reqs
                    for pos in range(req.device_len + 4)
                }
            )
            assert 7 not in pages
            pool.arena.allocate_pages([7, *pages])
        outputs = []
        for step in range(5):
            if step == 2:
                reqs.reverse()
                if pool.arena is not None:
                    pool.arena.free_pages_by_id([7])  # Move a live physical KV page.
            batch = Batch(reqs, "prefill" if step == 0 else "decode")
            engine.graph_runner.pad_batch(batch)
            batch.input_ids = torch.cat([req.input_ids[req.cached_len :] for req in reqs]).to(
                engine.device
            )
            batch.positions = torch.cat(
                [torch.arange(req.cached_len, req.device_len, device=engine.device) for req in reqs]
            )
            batch.out_loc = torch.cat(
                [table[req.table_idx, req.cached_len : req.device_len] for req in reqs]
            )
            engine.attn_backend.prepare_metadata(batch)
            use_graph = graphs and engine.graph_runner.can_use_cuda_graph(batch)
            assert use_graph == (graphs and step > 0 and len(slots) in (1, 2, 4))
            with engine.ctx.forward_batch(batch):
                logits = engine.graph_runner.replay(batch) if use_graph else engine.model.forward()
            outputs.append(
                torch.stack(
                    [
                        logits[i].cpu()
                        for i, req in sorted(enumerate(reqs), key=lambda pair: pair[1].table_idx)
                    ]
                )
            )
            for req in reqs:
                req.complete_one()
                req.append_host(torch.tensor([20 + req.table_idx + step], dtype=torch.int32))
        states = {
            req.table_idx: tuple(
                part.cpu().clone() for part in pool.recurrent._working(req.table_idx)
            )
            for req in reqs
        }
        for req in reqs:
            pool.release_request(req.table_idx, req.uid)
        if pool.arena is not None:
            pool.arena.free_pages_by_id(list(pool.arena._pages))
            pool.arena.check_integrity()
        return torch.stack(outputs), states

    results = []
    try:
        for slots in ([3, 0, 6, 1], [6, 2], [4], [5, 2, 0]):
            expected, reference_states = run(slots, False)
            actual, states = run(slots, True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for slot in slots:
                for a, b in zip(states[slot], reference_states[slot]):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
            results.append(
                {
                    "slots": slots,
                    "graph_decode": len(slots) in (1, 2, 4),
                    "max_abs_logits": float((actual - expected).abs().max()),
                    "states_bitwise_equal": True,
                }
            )
    finally:
        engine.shutdown()
    if rank == 0:
        report = {"unified": args.unified, "world_size": world, "cases": results}
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
