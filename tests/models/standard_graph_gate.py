"""Queued compatibility check for ordinary dense/MoE models and padded graphs."""

import argparse
import json
import os
from pathlib import Path

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig


def create_fixture(path):
    for moe in (False, True):
        config = {
            "model_type": "qwen3_moe" if moe else "qwen3",
            "architectures": ["Qwen3MoeForCausalLM" if moe else "Qwen3ForCausalLM"],
            "hidden_size": 256,
            "intermediate_size": 512,
            "vocab_size": 256,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "head_dim": 64,
            "max_position_embeddings": 64,
            "rope_theta": 1_000_000,
            "rms_norm_eps": 1e-6,
            "hidden_act": "silu",
            "tie_word_embeddings": False,
        }
        if moe:
            config.update(
                num_experts=4, num_experts_per_tok=2, moe_intermediate_size=128, norm_topk_prob=True
            )
        target = path / ("moe" if moe else "dense")
        target.mkdir(parents=True)
        (target / "config.json").write_text(json.dumps(config, indent=2) + "\n")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--create-fixture", action="store_true")
    parser.add_argument("--moe", action="store_true")
    args = parser.parse_args()
    if args.create_fixture:
        return create_fixture(args.fixture)
    if args.output is None:
        parser.error("Inference requires --output")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    engine = Engine(
        EngineConfig(
            str(args.fixture / ("moe" if args.moe else "dense")),
            DistributedInfo(rank, world),
            torch.bfloat16,
            ep_size=world if args.moe else 1,
            attention_backend="fi",
            moe_backend="fused",
            use_dummy_weight=True,
            cuda_graph_bs=[1, 2, 4],
            max_running_req=8,
            page_size=1,
            max_seq_len_override=64,
            num_page_override=512,
            distributed_timeout=180,
        )
    )
    assert set(engine.graph_runner.graph_map) == {1, 2, 4}
    assert engine.attn_backend.supports_graph_padding
    engine.page_table[:8, :64].copy_(
        torch.arange(512, device=engine.device, dtype=torch.int32).view(8, 64)
    )
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
        outputs = []
        for step in range(5):
            if step == 2:
                reqs.reverse()
            batch = Batch(reqs, "prefill" if step == 0 else "decode")
            engine.graph_runner.pad_batch(batch)
            batch.input_ids = torch.cat(
                [r.input_ids[r.cached_len :] for r in batch.padded_reqs]
            ).to(engine.device)
            batch.positions = torch.cat(
                [
                    torch.arange(r.cached_len, r.device_len, device=engine.device)
                    for r in batch.padded_reqs
                ]
            )
            batch.out_loc = torch.cat(
                [
                    engine.page_table[r.table_idx, r.cached_len : r.device_len]
                    for r in batch.padded_reqs
                ]
            )
            engine.attn_backend.prepare_metadata(batch)
            use_graph = graphs and engine.graph_runner.can_use_cuda_graph(batch)
            assert use_graph == (graphs and step > 0)
            with engine.ctx.forward_batch(batch):
                logits = engine.graph_runner.replay(batch) if use_graph else engine.model.forward()
            outputs.append(
                torch.stack(
                    [
                        logits[i].float().cpu()
                        for i, r in sorted(enumerate(reqs), key=lambda pair: pair[1].table_idx)
                    ]
                )
            )
            for req in reqs:
                req.complete_one()
                req.append_host(torch.tensor([20 + req.table_idx + step], dtype=torch.int32))
        return torch.stack(outputs)

    results = []
    try:
        for slots in ([3, 0, 6, 1], [6, 2], [4], [5, 2, 0]):
            expected, actual = run(slots, False), run(slots, True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            results.append(
                {"slots": slots, "max_abs_logits": float((actual - expected).abs().max())}
            )
    finally:
        engine.shutdown()
    if rank == 0:
        report = {
            "scope": "synthetic model compatibility",
            "comparison": "exact FP32 values; BF16 eager outputs are converted losslessly",
            "moe": args.moe,
            "world_size": world,
            "ep_size": world if args.moe else 1,
            "cases": results,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
