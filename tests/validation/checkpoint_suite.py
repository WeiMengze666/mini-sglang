"""Released-weight accuracy/performance suite. Run torchrun under gpuq.

Outputs teacher-forced full-vocabulary logits for independent engine comparison.
Repeated warm decode timings cover model execution and greedy token transfer;
loading and first-use compilation are recorded separately.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from minisgl.core import Batch, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine import Engine, EngineConfig
from minisgl.utils import get_stop_token_ids, load_tokenizer
from provenance import execution_identity


def assert_chunked_prefill(full, chunked, dtype):
    tolerance = (1e-4, 1e-4) if dtype == "float32" else (0.05, 0.1)
    torch.testing.assert_close(
        full["logits"], chunked["logits"], rtol=tolerance[0], atol=tolerance[1]
    )
    if full["token_ids"][0] != chunked["token_ids"][0]:
        raise AssertionError("Full and chunked prefill chose different greedy tokens")


def cases(tokenizer):
    prompts = [
        ("greeting", "请只回答：你好。"),
        ("arithmetic", "计算 17 × 23，只输出结果。"),
        ("translation", "Translate into English: 春天来了，河水渐渐变暖。"),
        ("python", "Write a Python function that returns the sum of even numbers in a list."),
        (
            "reasoning",
            "有三枚硬币，其中一枚两面都是正面。随机取一枚抛出正面，这枚是双面正面硬币的概率是多少？",
        ),
        (
            "explanation",
            "Explain why a KV cache speeds up autoregressive inference in two sentences.",
        ),
        ("json", 'Return JSON with keys "name" and "count", values "apple" and 3. No other text.'),
        ("unicode", "依次写出中文、English、日本語各一句问候。"),
    ]
    for count in (16, 64, 128):
        records = "\n".join(
            f"记录 {i}: 项目代号 P{i:04d}，数量 {i * 7 + 3}。" for i in range(count)
        )
        prompts.append((f"retrieval_{count}", records + f"\n请只回答记录 {count // 2} 的数量。"))
    return [
        {
            "name": name,
            "input_ids": tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=True, thinking=False
            ),
        }
        for name, text in prompts
    ]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attention-backend", default="recurrent-fi")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--dcp-size", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--strict-chunked-prefill", action="store_true")
    parser.add_argument("--cuda-graph-max-bs", type=int)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    args = parser.parse_args()
    if args.tokens < 1 or args.max_context < 1 or min(args.batch_sizes) < 1:
        raise ValueError("tokens, max-context and batch sizes must be positive")
    max_batch = max(args.batch_sizes)
    stride = (args.max_context + args.dcp_size - 1) // args.dcp_size * args.dcp_size
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = load_tokenizer(args.model)
    stop_ids = get_stop_token_ids(tokenizer)

    def through_stop(tokens):
        for i, token in enumerate(tokens):
            if token in stop_ids:
                return tokens[: i + 1]
        return tokens

    dataset = cases(tokenizer)
    if args.only:
        dataset = [case for case in dataset if case["name"] in args.only]
    if not dataset or max(len(c["input_ids"]) + args.tokens for c in dataset) > args.max_context:
        raise ValueError("Empty dataset or insufficient context capacity")
    args.output.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        torch.save(dataset, args.output / "cases.pt")
    if rank == 0:
        identity = execution_identity("minisgl")
        identity["configuration"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        (args.output / "provenance.json").write_text(json.dumps(identity, indent=2) + "\n")
    engine = Engine(
        EngineConfig(
            args.model,
            DistributedInfo(rank, world),
            getattr(torch, args.dtype),
            attention_backend=args.attention_backend,
            cuda_graph_max_bs=args.cuda_graph_max_bs,
            mxfp4_backend="triton",
            dcp_size=args.dcp_size,
            ep_size=args.ep_size,
            page_size=args.dcp_size,
            max_running_req=max(2, max_batch),
            max_cached_states=0,
            max_seq_len_override=args.max_context,
            num_page_override=max(args.max_context, max_batch * stride // args.dcp_size),
            memory_ratio=0.9,
            distributed_timeout=1800,
        )
    )
    table = torch.arange(max_batch * stride, dtype=torch.int32, device=engine.device)
    table = table.view(max_batch, stride)[:, : args.max_context]
    engine.page_table[:max_batch, : args.max_context].copy_(table)
    uid = 0

    def run(prompt, tokens, chunk=None, batch_size=1):
        nonlocal uid
        prompt = torch.tensor(prompt, dtype=torch.int32)
        reqs = [
            Req(
                prompt,
                slot,
                0,
                tokens,
                uid + slot + 1,
                SamplingParams(max_tokens=tokens, ignore_eos=True),
                None,
            )
            for slot in range(batch_size)
        ]
        uid += batch_size
        outputs, seconds, first = [[] for _ in reqs], [], None
        for step in range(tokens):
            stops = (
                list(range(chunk, len(prompt), chunk)) + [len(prompt)]
                if step == 0 and chunk
                else [reqs[0].device_len]
            )
            tick = time.monotonic()
            for stop in stops:
                parts = [
                    Req(
                        req.input_ids[:stop],
                        req.table_idx,
                        req.cached_len,
                        1,
                        req.uid,
                        SamplingParams(max_tokens=1, ignore_eos=True),
                        None,
                    )
                    for req in reqs
                ]
                batch = Batch(parts, "prefill" if step == 0 else "decode")
                batch.padded_reqs = parts
                batch.input_ids = torch.cat([p.input_ids[p.cached_len :] for p in parts]).to(
                    engine.device
                )
                batch.positions = torch.cat(
                    [torch.arange(p.cached_len, stop, device=engine.device) for p in parts]
                )
                batch.out_loc = torch.cat([table[p.table_idx, p.cached_len : stop] for p in parts])
                engine.attn_backend.prepare_metadata(batch)
                with engine.ctx.forward_batch(batch):
                    logits = (
                        engine.graph_runner.replay(batch)
                        if engine.graph_runner.can_use_cuda_graph(batch)
                        else engine.model.forward()
                    )
                for req in reqs:
                    req.cached_len = stop
            token = logits.argmax(-1).to(torch.int32).cpu()
            seconds.append(time.monotonic() - tick)
            if not torch.isfinite(logits).all():
                raise AssertionError("Non-finite released-model logits")
            if step == 0:
                first = logits.cpu()
            for slot, req in enumerate(reqs):
                outputs[slot].append(int(token[slot].item()))
                req.complete_one()
                req.append_host(token[slot : slot + 1])
        for req in reqs:
            engine.kv_cache.release_request(req.table_idx, req.uid)
        # Fixed-length timing continues past EOS; stopped responses remain the gate.
        raw_match = all(output == outputs[0] for output in outputs)
        stopped_match = all(through_stop(output) == through_stop(outputs[0]) for output in outputs)
        if not stopped_match:
            if rank == 0:
                torch.save(
                    {"logits": first, "token_ids": outputs, "step_seconds": seconds},
                    args.output / f"batch-mismatch-{uid}.pt",
                )
            raise AssertionError("Identical batch requests diverged before the configured stop")
        return {
            "logits": first,
            "token_ids": outputs[0],
            "all_token_ids": outputs,
            "step_seconds": seconds,
            "duplicate_raw_match": raw_match,
            "duplicate_through_stop_match": stopped_match,
        }

    try:
        results = []
        for case in dataset:
            full = run(case["input_ids"], args.tokens)
            chunk_size = min(64, max(1, len(case["input_ids"]) // 2))
            chunked = run(case["input_ids"], 1, chunk=chunk_size)
            if rank == 0:
                torch.save(
                    {"input_ids": case["input_ids"], "full": full, "chunked": chunked},
                    args.output / (case["name"] + ".pt"),
                )
                results.append(
                    {
                        "name": case["name"],
                        "prompt_tokens": len(case["input_ids"]),
                        "token_ids": full["token_ids"],
                        "text": tokenizer.decode(full["token_ids"]),
                        "step_seconds": full["step_seconds"],
                        "chunked_first_token": chunked["token_ids"][0],
                        "chunked_max_abs_logits": float(
                            (full["logits"] - chunked["logits"]).abs().max()
                        ),
                    }
                )
                (args.output / "report.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=2) + "\n"
                )
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
            if args.strict_chunked_prefill:
                assert_chunked_prefill(full, chunked, args.dtype)
        # Warm repetitions use the same prompt, with private state reset each time.
        batches = []
        for batch_size in dict.fromkeys(args.batch_sizes):
            run(dataset[0]["input_ids"], args.tokens, batch_size=batch_size)
            warm = [
                run(dataset[0]["input_ids"], args.tokens, batch_size=batch_size) for _ in range(3)
            ]
            decode = (
                statistics.median(s for r in warm for s in r["step_seconds"][1:])
                if args.tokens > 1
                else None
            )
            batches.append(
                {
                    "batch_size": batch_size,
                    "warm_step_seconds": [r["step_seconds"] for r in warm],
                    "warm_token_ids": [r["all_token_ids"] for r in warm],
                    "duplicate_raw_match": all(r["duplicate_raw_match"] for r in warm),
                    "duplicate_through_stop_match": all(
                        r["duplicate_through_stop_match"] for r in warm
                    ),
                    "median_decode_seconds": decode,
                    "decode_tokens_per_second": batch_size / decode if decode else None,
                    "greedy_matches_first_case": (
                        all(r["token_ids"] == results[0]["token_ids"] for r in warm)
                        if rank == 0
                        else None
                    ),
                    "greedy_through_stop_matches_first_case": (
                        all(
                            through_stop(r["token_ids"]) == through_stop(results[0]["token_ids"])
                            for r in warm
                        )
                        if rank == 0
                        else None
                    ),
                }
            )
            if rank == 0:
                single = next((r for r in batches if r["batch_size"] == 1), {})
                (args.output / "benchmark.json").write_text(
                    json.dumps(
                        {
                            "boundary": "batch construction, model execution (eager or CUDA Graph), argmax and token transfer; excludes scheduler",
                            "complete": len(batches) == len(set(args.batch_sizes)),
                            "warm_step_seconds": single.get("warm_step_seconds"),
                            "median_decode_seconds": single.get("median_decode_seconds"),
                            "batches": batches,
                        },
                        indent=2,
                    )
                    + "\n"
                )
        if args.profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as profile:
                run(dataset[0]["input_ids"], 4)
            if rank == 0:
                profile.export_chrome_trace(str(args.output / "trace.json"))
                (args.output / "profile.txt").write_text(
                    profile.key_averages().table(sort_by="self_device_time_total", row_limit=50)
                )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
