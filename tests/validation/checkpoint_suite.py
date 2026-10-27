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
from minisgl.utils import load_tokenizer
from provenance import execution_identity


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
    args = parser.parse_args()
    if args.tokens < 1 or args.max_context < 1:
        raise ValueError("tokens and max-context must be positive")
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = load_tokenizer(args.model)
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
            mxfp4_backend="triton",
            dcp_size=args.dcp_size,
            ep_size=args.ep_size,
            page_size=args.dcp_size,
            max_running_req=2,
            max_cached_states=0,
            max_seq_len_override=args.max_context,
            num_page_override=args.max_context,
            memory_ratio=0.9,
            distributed_timeout=1800,
        )
    )
    table = torch.arange(args.max_context, dtype=torch.int32, device=engine.device)
    engine.page_table[0, : args.max_context].copy_(table)
    uid = 0

    def run(prompt, tokens, chunk=None):
        nonlocal uid
        uid += 1
        prompt = torch.tensor(prompt, dtype=torch.int32)
        req = Req(prompt, 0, 0, tokens, uid, SamplingParams(max_tokens=tokens), None)
        outputs, seconds, first = [], [], None
        for step in range(tokens):
            stops = (
                list(range(chunk, len(prompt), chunk)) + [len(prompt)]
                if step == 0 and chunk
                else [req.device_len]
            )
            tick = time.monotonic()
            for stop in stops:
                part = Req(
                    req.input_ids[:stop],
                    0,
                    req.cached_len,
                    1,
                    uid,
                    SamplingParams(max_tokens=1),
                    None,
                )
                batch = Batch([part], "prefill" if step == 0 else "decode")
                batch.padded_reqs = [part]
                batch.input_ids = part.input_ids[part.cached_len :].to(engine.device)
                batch.positions = torch.arange(part.cached_len, stop, device=engine.device)
                batch.out_loc = table[part.cached_len : stop]
                engine.attn_backend.prepare_metadata(batch)
                with engine.ctx.forward_batch(batch):
                    logits = engine.model.forward()
                req.cached_len = stop
            token = logits.argmax(-1).to(torch.int32).cpu()
            seconds.append(time.monotonic() - tick)
            if not torch.isfinite(logits).all():
                raise AssertionError("Non-finite released-model logits")
            if step == 0:
                first = logits.cpu()
            outputs.append(int(token.item()))
            req.complete_one()
            req.append_host(token)
        engine.kv_cache.release_request(0, uid)
        return {"logits": first, "token_ids": outputs, "step_seconds": seconds}

    try:
        results = []
        for case in dataset:
            full = run(case["input_ids"], args.tokens)
            chunked = run(case["input_ids"], 1, chunk=64)
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
                    }
                )
                (args.output / "report.json").write_text(
                    json.dumps(results, ensure_ascii=False, indent=2) + "\n"
                )
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
        # Warm repetitions use the same prompt, with private state reset each time.
        warm = []
        for _ in range(3):
            sample = run(dataset[0]["input_ids"], args.tokens)
            warm.append(sample["step_seconds"])
        if rank == 0:
            (args.output / "benchmark.json").write_text(
                json.dumps(
                    {
                        "boundary": "model.forward, argmax and token transfer; excludes scheduler",
                        "warm_step_seconds": warm,
                        "median_decode_seconds": statistics.median(s for r in warm for s in r[1:])
                        if args.tokens > 1
                        else None,
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
