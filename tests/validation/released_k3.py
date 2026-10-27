"""Opt-in, local released-checkpoint gate. Launch with torchrun under gpuq.

Writes full-vocabulary logits for sequential independent-engine comparison.
A successful run establishes runtime consistency, not independent model accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.utils import load_tokenizer


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--memory-ratio", type=float, default=0.9)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--dcp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--unified-memory", action="store_true")
    parser.add_argument("--num-pages", type=int, default=512)
    parser.add_argument("--mxfp4-backend", choices=["torch", "triton"], default="torch")
    parser.add_argument(
        "--attention-backend", choices=["recurrent", "recurrent-fi"], default="recurrent"
    )
    parser.add_argument("--reference-logits", type=Path)
    parser.add_argument("--prompt", default="请只回答：你好。")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    rank, size = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if not args.model.is_dir() or args.tokens < 1:
        raise ValueError("A local checkpoint and positive output length are required")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Rank {rank}: loading released checkpoint", flush=True)
    started = time.monotonic()
    engine = Engine(
        EngineConfig(
            str(args.model),
            DistributedInfo(rank, size),
            getattr(torch, args.dtype),
            dcp_size=args.dcp_size,
            ep_size=args.ep_size,
            page_size=max(1, args.dcp_size),
            enable_unified_memory=args.unified_memory,
            mxfp4_backend=args.mxfp4_backend,
            attention_backend=args.attention_backend,
            max_running_req=2,
            max_seq_len_override=256,
            num_page_override=args.num_pages,
            distributed_timeout=1800,
            memory_ratio=args.memory_ratio,
        )
    )
    print(f"Rank {rank}: loaded in {time.monotonic() - started:.1f}s", flush=True)
    try:
        tokenizer = load_tokenizer(str(args.model))
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": args.prompt}], tokenize=True, thinking=False
        )
        prompt = torch.tensor(prompt, dtype=torch.int32)
        if len(prompt) + args.tokens > 256:
            raise ValueError("Prompt and generation exceed the bounded validation context")
        table = torch.arange(512, dtype=torch.int32, device=engine.device).reshape(2, 256)
        engine.page_table[:2, :256].copy_(table)
        if engine.kv_cache.arena is not None:
            engine.kv_cache.arena.allocate_pages(list(range(512 // args.dcp_size)))
        req = Req(prompt, 0, 0, args.tokens, 100, SamplingParams(max_tokens=args.tokens), None)
        outputs, step_seconds = [], []
        traces = {"full": [], "chunked": []}
        trace_mode = ["full"]
        for index, layer in enumerate(engine.model.model.layers.op_list if args.trace else []):
            original = layer.forward

            def traced(*values, _original=original, _index=index, **kwargs):
                output = _original(*values, **kwargs)
                batch = values[2]
                if rank == 0 and batch.reqs[-1].device_len == len(prompt):
                    traces[trace_mode[0]].append((_index, output[-1].detach().cpu()))
                return output

            layer.forward = traced
        reference_logits = None
        for step in range(args.tokens):
            batch = Batch([req], "prefill" if step == 0 else "decode")
            batch.padded_reqs = [req]
            batch.input_ids = req.input_ids[req.cached_len :].to(engine.device)
            batch.positions = torch.arange(req.cached_len, req.device_len, device=engine.device)
            batch.out_loc = table[0, req.cached_len : req.device_len]
            engine.attn_backend.prepare_metadata(batch)
            tick = time.monotonic()
            with engine.ctx.forward_batch(batch):
                logits = engine.model.forward()
            if not torch.isfinite(logits).all():
                raise AssertionError("Released checkpoint produced non-finite logits")
            token = logits.argmax(-1).to(torch.int32).cpu()
            step_seconds.append(time.monotonic() - tick)
            if step == 0:
                reference_logits = logits.cpu()
                if rank == 0:
                    torch.save(
                        {"input_ids": prompt, "logits": reference_logits},
                        args.output / "prefill.pt",
                    )
            outputs.append(int(token.item()))
            req.complete_one()
            req.append_host(token)
            if rank == 0:
                print(
                    f"step={step} seconds={step_seconds[-1]:.3f} token={outputs[-1]} text={tokenizer.decode(outputs)!r}",
                    flush=True,
                )
        if rank == 0:
            (args.output / "generation.json").write_text(
                json.dumps(
                    {
                        "token_ids": outputs,
                        "text": tokenizer.decode(outputs),
                        "step_seconds": step_seconds,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n"
            )
        if args.reference_logits:
            reference = torch.load(args.reference_logits, map_location="cpu", weights_only=True)
            torch.testing.assert_close(prompt, reference["input_ids"], rtol=0, atol=0)
            expected = reference["logits"].float()
            error = reference_logits - expected
            if rank == 0:
                (args.output / "layout-comparison.json").write_text(
                    json.dumps(
                        {
                            "reference": str(args.reference_logits),
                            "max_abs": float(error.abs().max()),
                            "relative_l2": float(error.norm() / expected.norm()),
                            "greedy_match": int(reference_logits.argmax(-1))
                            == int(expected.argmax(-1)),
                        },
                        indent=2,
                    )
                    + "\n"
                )
            tolerance = (1e-4, 1e-4) if args.dtype == "float32" else (0.05, 0.1)
            torch.testing.assert_close(
                reference_logits, expected, rtol=tolerance[0], atol=tolerance[1]
            )
        trace_mode[0] = "chunked"
        # Replay the same prompt in two chunks, from a clean request state.
        engine.kv_cache.release_request(0, 100)
        split = max(1, len(prompt) // 2)
        last = None
        for start, stop in ((0, split), (split, len(prompt))):
            if start == stop:
                continue
            chunk = Req(prompt[:stop], 0, start, 1, 101, SamplingParams(max_tokens=1), None)
            batch = Batch([chunk], "prefill")
            batch.padded_reqs = [chunk]
            batch.input_ids = prompt[start:stop].to(engine.device)
            batch.positions = torch.arange(start, stop, device=engine.device)
            batch.out_loc = table[0, start:stop]
            with engine.ctx.forward_batch(batch):
                last = engine.model.forward()
        if rank == 0:
            chunked = last.cpu()
            torch.save({"logits": chunked, "traces": traces}, args.output / "chunked.pt")
            error = chunked - reference_logits
            diagnostics = {
                "max_abs": float(error.abs().max()),
                "relative_l2": float(error.norm() / reference_logits.norm()),
                "cosine": float(
                    torch.nn.functional.cosine_similarity(chunked, reference_logits).item()
                ),
                "greedy_match": int(chunked.argmax(-1)) == int(reference_logits.argmax(-1)),
                "layers": [
                    {"layer": i, "relative_l2": float((a - b).float().norm() / a.float().norm())}
                    for (i, a), (_, b) in zip(traces["full"], traces["chunked"])
                ],
            }
            (args.output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
            print(json.dumps(diagnostics), flush=True)
        torch.testing.assert_close(last.cpu(), reference_logits, rtol=0.05, atol=0.1)
        if int(last.argmax(-1).item()) != int(reference_logits.argmax(-1).item()):
            raise AssertionError("Released-checkpoint chunked prefill changed the greedy token")
        report = {
            "status": "passed",
            "scope": "released_checkpoint_runtime_consistency",
            "independent_accuracy": "not_tested",
            "tp_size": size,
            "dcp_size": args.dcp_size,
            "ep_size": args.ep_size,
            "token_ids": outputs,
            "text": tokenizer.decode(outputs),
            "step_seconds": step_seconds,
            "chunked_prefill_max_abs": float((last.cpu() - reference_logits).abs().max()),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(engine.device),
        }
        if rank == 0:
            (args.output / "report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
