"""Paired, zero-shot GSM8K validation with identical saved tokenized prompts.

Prepare cases on CPU from the pinned official test file. Run mini with torchrun
or SGLang in its own environment, always through gpuq. Inference never downloads
datasets. This measures task accuracy, separately from decode microbenchmarks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from decimal import Decimal
from pathlib import Path

GSM8K_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
GSM8K_URL = f"https://raw.githubusercontent.com/openai/grade-school-math/{GSM8K_REVISION}/grade_school_math/data/test.jsonl"


def answer_number(text):
    matches = re.findall(r"####\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)(?![0-9,/+*eE-]|\.[0-9])", text)
    if not matches:
        return None
    value = Decimal(matches[-1].replace(",", ""))
    return str(value.normalize()) if value else "0"


def prepare(args):
    from minisgl.utils import get_stop_token_ids, load_tokenizer

    raw = args.dataset.read_bytes()
    if hashlib.sha256(raw).hexdigest() != GSM8K_SHA256:
        raise ValueError("Dataset SHA256 differs from the requested pinned file")
    examples = [json.loads(line) for line in raw.decode().splitlines()]
    if len(examples) != 1319:
        raise ValueError("Expected the complete official GSM8K test split")
    tokenizer = load_tokenizer(args.model)
    stop_ids = set(get_stop_token_ids(tokenizer))
    eos = tokenizer.eos_token_id
    if eos is not None:
        stop_ids.update([eos] if isinstance(eos, int) else eos)
    cases = []
    for index, example in enumerate(examples):
        gold = answer_number(example["answer"])
        if gold is None:
            raise ValueError(f"Missing gold answer at dataset row {index}")
        prompt = (
            "Solve this math problem. Show your reasoning, and end with a line "
            "in the form #### <number>.\n\n" + example["question"]
        )
        cases.append(
            {
                "id": index,
                "gold": gold,
                "input_ids": tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    thinking=False,
                ),
            }
        )
    if args.cases.exists():
        raise FileExistsError("Choose a fresh cases file; saved tokenization is immutable")
    args.cases.write_text(
        json.dumps(
            {
                "dataset_url": GSM8K_URL,
                "dataset_sha256": GSM8K_SHA256,
                "protocol": "zero-shot, thinking=False, final #### number; generation stops plus tokenizer EOS",
                "stop_token_ids": sorted(stop_ids),
                "examples": cases,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    print(
        json.dumps(
            {"cases": len(cases), "max_prompt_tokens": max(len(c["input_ids"]) for c in cases)}
        )
    )


def record(case, token_ids, text, truncated):
    predicted = answer_number(text)
    return {
        "id": case["id"],
        "gold": case["gold"],
        "predicted": predicted,
        "correct": predicted == case["gold"],
        "truncated": truncated,
        "token_ids": token_ids,
        "text": text,
    }


def save_progress(args, rows, total, started):
    report = {
        "complete": len(rows) == total,
        "engine": args.engine,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "completed": len(rows),
        "expected": total,
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "invalid_answers": sum(row["predicted"] is None for row in rows),
        "truncated": sum(row["truncated"] for row in rows),
        "generated_tokens": sum(len(row["token_ids"]) for row in rows),
        "evaluation_seconds": time.monotonic() - started,
        "timing_boundary": "task evaluation including prefill, decode, CPU transfer and state reset; excludes model loading",
    }
    temporary = args.output / "report.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output / "report.json")
    print(
        json.dumps({key: report[key] for key in ("completed", "correct", "accuracy", "truncated")}),
        flush=True,
    )


def run_mini(args, cases, stop_ids):
    import torch
    from minisgl.core import Batch, Req, SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig
    from minisgl.utils import load_tokenizer
    from provenance import execution_identity

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != args.tp_size:
        raise ValueError("torchrun world size differs from --tp-size")
    torch.backends.cuda.matmul.allow_tf32 = False
    engine = Engine(
        EngineConfig(
            args.model,
            DistributedInfo(rank, world),
            torch.bfloat16,
            attention_backend=args.attention_backend,
            mxfp4_backend="triton",
            dcp_size=args.dcp_size,
            ep_size=args.ep_size,
            max_running_req=args.batch_size,
            max_cached_states=0,
            max_seq_len_override=args.max_context,
            page_size=64,
            num_page_override=max(2048, args.batch_size * args.max_context // 64),
            cuda_graph_bs=list(range(1, args.batch_size + 1)) if args.cuda_graph else [],
            distributed_timeout=1800,
        )
    )
    tokenizer = load_tokenizer(args.model)
    table = torch.arange(
        args.batch_size * args.max_context, dtype=torch.int32, device=engine.device
    ).view(args.batch_size, args.max_context)
    engine.page_table[: args.batch_size, : args.max_context].copy_(table)
    if rank == 0:
        (args.output / "provenance.json").write_text(
            json.dumps(execution_identity("minisgl"), indent=2) + "\n"
        )
    rows, started = [], time.monotonic()
    try:
        with torch.inference_mode():
            for begin in range(0, len(cases), args.batch_size):
                group = cases[begin : begin + args.batch_size]
                reqs = [
                    Req(
                        torch.tensor(case["input_ids"], dtype=torch.int32),
                        slot,
                        0,
                        args.max_new_tokens,
                        begin + slot + 1,
                        SamplingParams(),
                        None,
                    )
                    for slot, case in enumerate(group)
                ]
                generated = [[] for _ in reqs]
                active = list(reqs)
                while active:
                    batch = Batch(active, "prefill" if active[0].cached_len == 0 else "decode")
                    engine.graph_runner.pad_batch(batch)
                    batch.input_ids = torch.cat(
                        [req.input_ids[req.cached_len :] for req in active]
                    ).to(engine.device)
                    batch.positions = torch.cat(
                        [
                            torch.arange(req.cached_len, req.device_len, device=engine.device)
                            for req in active
                        ]
                    )
                    batch.out_loc = torch.cat(
                        [table[req.table_idx, req.cached_len : req.device_len] for req in active]
                    )
                    engine.attn_backend.prepare_metadata(batch)
                    with engine.ctx.forward_batch(batch):
                        logits = (
                            engine.graph_runner.replay(batch)
                            if engine.graph_runner.can_use_cuda_graph(batch)
                            else engine.model.forward()
                        )
                    tokens = logits.argmax(-1).to(torch.int32).cpu()
                    if not torch.isfinite(logits).all():
                        raise AssertionError("Task evaluation produced non-finite logits")
                    remaining = []
                    for index, req in enumerate(active):
                        token = int(tokens[index])
                        generated[req.table_idx].append(token)
                        req.complete_one()
                        req.append_host(tokens[index : index + 1])
                        if (
                            token in stop_ids
                            or len(generated[req.table_idx]) == args.max_new_tokens
                        ):
                            if hasattr(engine.kv_cache, "release_request"):
                                engine.kv_cache.release_request(req.table_idx, req.uid)
                        else:
                            remaining.append(req)
                    active = remaining
                if rank == 0:
                    new = [
                        record(
                            case,
                            output,
                            tokenizer.decode(output),
                            len(output) == args.max_new_tokens and output[-1] not in stop_ids,
                        )
                        for case, output in zip(group, generated)
                    ]
                    with (args.output / "responses.jsonl").open("a") as stream:
                        for row in new:
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.extend(new)
                    save_progress(args, rows, len(cases), started)
    finally:
        engine.shutdown()


def run_sglang(args, cases, stop_ids):
    from provenance import execution_identity
    from sglang import Engine

    identity = execution_identity("sglang")
    if (
        identity["revision"] != "baa9befa381511cd4d56f83980951e5a45fffc6d"
        or identity["tracked_changes"]
    ):
        raise RuntimeError("Official reference differs from the pinned clean source")
    (args.output / "provenance.json").write_text(json.dumps(identity, indent=2) + "\n")
    engine = Engine(
        model_path=args.model,
        tp_size=args.tp_size,
        dcp_size=args.dcp_size,
        ep_size=args.ep_size,
        dtype="bfloat16",
        trust_remote_code=True,
        context_length=args.max_context,
        mem_fraction_static=0.9,
        max_running_requests=args.batch_size,
        disable_radix_cache=True,
        mamba_ssm_dtype="float32",
        chunked_prefill_size=4096,
        page_size=64,
        flashinfer_autotune_skip_ops=args.flashinfer_autotune_skip_ops,
        log_level="info",
    )
    rows, started = [], time.monotonic()
    try:
        for begin in range(0, len(cases), args.batch_size):
            group = cases[begin : begin + args.batch_size]
            results = engine.generate(
                input_ids=[case["input_ids"] for case in group],
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": args.max_new_tokens,
                    "stop_token_ids": list(stop_ids),
                },
            )
            if isinstance(results, dict):
                results = [results]
            if len(results) != len(group):
                raise AssertionError("Official engine returned an incomplete batch")
            new = [
                record(
                    case,
                    result["output_ids"],
                    result["text"],
                    result["meta_info"]["finish_reason"]["type"] == "length",
                )
                for case, result in zip(group, results)
            ]
            with (args.output / "responses.jsonl").open("a") as stream:
                for row in new:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.extend(new)
            save_progress(args, rows, len(cases), started)
    finally:
        engine.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine", choices=["mini", "sglang"], default="mini")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--dcp-size", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--attention-backend", default="recurrent-fi")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--flashinfer-autotune-skip-ops", nargs="*", default=[])
    args = parser.parse_args()
    if args.prepare:
        if args.dataset is None:
            parser.error("Preparation requires --dataset")
        return prepare(args)
    if (
        args.output is None
        or min(args.batch_size, args.max_new_tokens, args.max_context) < 1
        or args.limit < 0
    ):
        parser.error("Inference requires an output directory and positive capacities")
    if (args.output / "report.json").exists() or (args.output / "responses.jsonl").exists():
        raise FileExistsError("Choose a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    saved = json.loads(args.cases.read_text())
    cases = saved["examples"][: args.limit or None]
    if (
        not cases
        or max(len(case["input_ids"]) + args.max_new_tokens for case in cases) > args.max_context
    ):
        raise ValueError("Empty cases or insufficient context")
    runner = run_mini if args.engine == "mini" else run_sglang
    runner(args, cases, set(saved["stop_token_ids"]))


if __name__ == "__main__":
    main()
