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
from contextlib import contextmanager
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


def execution_contract(args):
    """Effective precision and layout, independent of backend naming conventions."""
    return {
        "model_dtype": "bfloat16",
        "moe_activation_precision": args.activation_precision,
        "kv_cache_dtype": args.kv_cache_dtype,
        "recurrent_state_dtype": args.recurrent_state_dtype,
        "conv_state_dtype": "bfloat16",
        "tp_size": args.tp_size,
        "dcp_size": args.dcp_size,
        "ep_size": args.ep_size,
        "page_size": 64,
    }


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
        "execution_contract": execution_contract(args),
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
        "workload_boundary": "offline numerical/task-quality evaluation in fixed-size request groups; prefix caching is disabled; this is not the canonical online serving benchmark",
    }
    temporary = args.output / "report.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output / "report.json")
    print(
        json.dumps({key: report[key] for key in ("completed", "correct", "accuracy", "truncated")}),
        flush=True,
    )


def save_failure(args, error, *, stage, group, generated=(), batch=None, logits=None, graph=False):
    """Retain CPU token history and device metadata without reading a failed GPU."""
    import dataclasses
    import traceback

    import torch

    target = args.output / ("failure-rank" + os.environ.get("RANK", "0") + ".json")
    if target.exists():
        return target  # nested failure handlers retain the first, closest context

    def describe(value, depth=0):
        if isinstance(value, torch.Tensor):
            result = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "device": str(value.device),
                "storage_offset": value.storage_offset(),
            }
            if value.device.type == "cpu" and value.numel() <= 4096:
                result["values"] = value.tolist()
            return result
        if depth > 4:
            return {"type": type(value).__name__}
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                f.name: describe(getattr(value, f.name), depth + 1)
                for f in dataclasses.fields(value)
                if hasattr(value, f.name)
            }
        if isinstance(value, dict):
            return {str(k): describe(v, depth + 1) for k, v in list(value.items())[:64]}
        if isinstance(value, (list, tuple)):
            return [describe(v, depth + 1) for v in value[:64]]
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        return str(value) if isinstance(value, Path) else {"type": type(value).__name__}

    requests = [] if batch is None else batch.reqs
    result = {
        "complete": False,
        "observed_unix_ns": time.time_ns(),
        "stage": stage,
        "graph_replay": graph,
        "error": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        "configuration": describe(vars(args)),
        "case_ids": [case["id"] for case in group],
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
        "generated_ids": generated,
        "requests": [
            {
                name: describe(getattr(req, name))
                for name in ("uid", "table_idx", "cached_len", "device_len", "input_ids")
            }
            for req in requests
        ],
        "batch": None
        if batch is None
        else {
            name: describe(getattr(batch, name))
            for name in ("phase", "input_ids", "positions", "out_loc", "attn_metadata")
            if hasattr(batch, name)
        },
        "logits": describe(logits),
        "limitation": "Device tensors retain metadata only. Generated tokens and small CPU inputs/logical indices are saved; large prompts remain in the immutable cases file. Device values and kernel traces must be captured before a fault.",
    }
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(target)
    return target


def shutdown_engine(engine, primary_error=None):
    try:
        engine.shutdown()
    except Exception as cleanup:
        if primary_error is None:
            raise
        import sys

        print(
            f"Cleanup also failed while handling {type(primary_error).__name__}: {cleanup}",
            file=sys.stderr,
            flush=True,
        )


def quality_batch(engine, active):
    """Use real requests for ownership and physical rows for graph inputs."""
    import torch
    from minisgl.core import Batch

    batch = Batch(active, "prefill" if active[0].cached_len == 0 else "decode")
    engine.graph_runner.pad_batch(batch)
    batch.input_ids = torch.cat(
        [req.input_ids[req.cached_len : req.device_len] for req in batch.padded_reqs]
    ).to(engine.device)
    batch.positions = torch.cat(
        [
            torch.arange(req.cached_len, req.device_len, dtype=torch.int32, device=engine.device)
            for req in batch.padded_reqs
        ]
    )
    batch.out_loc = torch.cat(
        [
            engine.page_table[req.table_idx, req.cached_len : req.device_len]
            for req in batch.padded_reqs
        ]
    )
    engine.attn_backend.prepare_metadata(batch)
    return batch


@contextmanager
def finite_sampling(engine):
    """Quality-only observer; delegate real sampling once and restore exactly."""
    import torch

    sampler = engine.sampler
    original = sampler.sample
    missing = object()
    previous = vars(sampler).get("sample", missing)

    def checked(logits, sampling_args):
        # Engine publishes real rows to its sampler; undefined padding is excluded.
        if not torch.isfinite(logits).all():
            raise AssertionError(
                f"Task evaluation produced non-finite real logits: "
                f"dtype={logits.dtype}, shape={tuple(logits.shape)}, stride={logits.stride()}"
            )
        return original(logits, sampling_args)

    sampler.sample = checked
    try:
        yield
    finally:
        if previous is missing:
            del sampler.sample
        else:
            sampler.sample = previous


def checkpoint_loader_options(args):
    keys = ["weight_loader_prefetch_checkpoints", "weight_loader_prefetch_num_threads"]
    if args.engine == "sglang":
        keys.append("model_loader_extra_config")
    return {key: getattr(args, key) for key in keys if getattr(args, key, None) is not None}


def run_mini(args, cases, stop_ids, *, worker_owner=None):
    import torch
    from minisgl.core import Req, SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig
    from minisgl.utils import load_tokenizer
    from provenance import execution_identity

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    if world != args.tp_size:
        raise ValueError("torchrun world size differs from --tp-size")
    engine = Engine(
        EngineConfig(
            args.model,
            DistributedInfo(rank, world),
            torch.bfloat16,
            attention_backend=args.attention_backend,
            distributed_backend=args.distributed_backend,
            distributed_provider=args.distributed_provider,
            mxfp4_backend=args.mxfp4_backend,
            moe_backend=args.moe_backend,
            mxfp4_activation_precision=args.activation_precision,
            kv_cache_dtype=(
                torch.bfloat16 if args.kv_cache_dtype == "bfloat16" else torch.float8_e4m3fn
            ),
            recurrent_state_dtype=getattr(torch, args.recurrent_state_dtype),
            dcp_size=args.dcp_size,
            ep_size=args.ep_size,
            max_running_req=args.max_running_requests,
            max_seq_len_override=args.max_context,
            page_size=64,
            num_page_override=(args.batch_size * args.max_context + 64 * args.dcp_size - 1)
            // (64 * args.dcp_size),
            cuda_graph_bs=None if args.cuda_graph else [],
            cuda_graph_max_bs=args.cuda_graph_max_bs,
            distributed_timeout=1800,
            **checkpoint_loader_options(args),
        ),
        worker_owner=worker_owner,
    )
    tokenizer = load_tokenizer(args.model)
    table = torch.arange(
        args.batch_size * args.max_context, dtype=torch.int32, device=engine.device
    ).view(args.batch_size, args.max_context)
    engine.page_table[: args.batch_size, : args.max_context].copy_(table)
    if rank == 0:
        identity = execution_identity("minisgl")
        identity["quality_observer"] = {
            "finite_real_logits_before_sampler": True,
            "delegates_original_sampler_once": True,
            "changes_seed_or_resamples": False,
            "scope": "Validation-only; restored in finally; absent from serving performance.",
        }
        (args.output / "provenance.json").write_text(json.dumps(identity, indent=2) + "\n")
    rows, started = [], time.monotonic()
    group, generated, batch, logits = [], [], None, None
    failure, stage, graph = None, "setup", False
    try:
        with torch.inference_mode(), finite_sampling(engine):
            for begin in range(0, len(cases), args.batch_size):
                group = cases[begin : begin + args.batch_size]
                generated, batch, logits = [], None, None
                stage = "request_init"
                reqs = [
                    Req(
                        torch.tensor(case["input_ids"], dtype=torch.int32),
                        slot,
                        0,
                        args.max_new_tokens,
                        begin + slot + 1,
                        SamplingParams(max_tokens=args.max_new_tokens),
                        None,
                    )
                    for slot, case in enumerate(group)
                ]
                generated = [[] for _ in reqs]
                active = list(reqs)
                while active:
                    stage, graph = "prepare_batch", False
                    batch = quality_batch(engine, active)
                    sampling_args = engine.sampler.prepare(batch)
                    stage = "model_forward"
                    graph = engine.graph_runner.can_use_cuda_graph(batch)
                    forward_output = engine.forward_batch(batch, sampling_args)
                    stage = "token_transfer"
                    forward_output.copy_done_event.synchronize()
                    tokens = forward_output.next_tokens_cpu
                    stage = "token_bookkeeping"
                    remaining = []
                    for index, req in enumerate(active):
                        token = int(tokens[index])
                        generated[req.table_idx].append(token)
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
                    for row, req in zip(new, reqs):
                        row["logical_request"] = {
                            "uid": req.uid,
                            "table_idx": req.table_idx,
                            "cached_len": req.cached_len,
                            "device_len": req.device_len,
                        }
                    with (args.output / "responses.jsonl").open("a") as stream:
                        for row in new:
                            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows.extend(new)
                    save_progress(args, rows, len(cases), started)
    except BaseException as exc:
        failure = exc
        try:
            save_failure(
                args,
                exc,
                stage=stage,
                group=group,
                generated=generated,
                batch=batch,
                logits=logits,
                graph=graph,
            )
        except Exception as recording:
            import sys

            print(f"Failure recording also failed: {recording}", file=sys.stderr, flush=True)
        if worker_owner is not None:
            worker_owner.abort(exc)
        raise
    finally:
        shutdown_engine(engine, failure)


def run_sglang(args, cases, stop_ids):
    from provenance import execution_identity
    from sglang import Engine

    identity = execution_identity("sglang")
    if identity["revision"] != args.expected_reference_revision or identity["tracked_changes"]:
        raise RuntimeError("Official reference differs from the pinned clean source")
    (args.output / "provenance.json").write_text(json.dumps(identity, indent=2) + "\n")
    attention_options = {
        key: getattr(args, "sglang_" + key)
        for key in ("attention_backend", "prefill_attention_backend", "decode_attention_backend")
        if getattr(args, "sglang_" + key) is not None
    }
    loader_options = checkpoint_loader_options(args)
    engine = Engine(
        model_path=args.model,
        tp_size=args.tp_size,
        dcp_size=args.dcp_size,
        ep_size=args.ep_size,
        dtype="bfloat16",
        trust_remote_code=True,
        context_length=args.max_context,
        mem_fraction_static=0.9,
        max_running_requests=args.max_running_requests,
        disable_radix_cache=True,
        disable_cuda_graph=not args.cuda_graph,
        cuda_graph_max_bs_decode=args.cuda_graph_max_bs,
        mamba_ssm_dtype=args.recurrent_state_dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        moe_runner_backend=args.moe_backend,
        flashinfer_mxfp4_moe_precision=(
            "default" if args.activation_precision == "mxfp8" else "bf16"
        ),
        **attention_options,
        **loader_options,
        chunked_prefill_size=4096,
        page_size=64,
        flashinfer_autotune_skip_ops=args.flashinfer_autotune_skip_ops,
        log_level="info",
    )
    rows, started = [], time.monotonic()
    group, results, failure, stage = [], None, None, "setup"
    try:
        for begin in range(0, len(cases), args.batch_size):
            group = cases[begin : begin + args.batch_size]
            stage, results = "native_generate", None
            results = engine.generate(
                input_ids=[case["input_ids"] for case in group],
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": args.max_new_tokens,
                    "stop_token_ids": list(stop_ids),
                },
            )
            stage = "native_result"
            if isinstance(results, dict):
                results = [results]
            if len(results) != len(group):
                raise AssertionError("Official engine returned an incomplete batch")
            finish_reasons = [result["meta_info"]["finish_reason"] for result in results]
            if any(reason.get("type") not in {"stop", "length"} for reason in finish_reasons):
                (args.output / "native-terminal-error.json").write_text(
                    json.dumps(
                        {"case_ids": [case["id"] for case in group], "results": results},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
                raise AssertionError(f"Official engine returned non-completion: {finish_reasons!r}")
            new = [
                record(
                    case,
                    result["output_ids"],
                    result["text"],
                    result["meta_info"]["finish_reason"]["type"] == "length",
                )
                for case, result in zip(group, results)
            ]
            for row, result in zip(new, results):
                row["native_meta_info"] = result["meta_info"]
            with (args.output / "responses.jsonl").open("a") as stream:
                for row in new:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.extend(new)
            save_progress(args, rows, len(cases), started)
    except BaseException as exc:
        failure = exc
        try:
            generated = (
                [r.get("output_ids", []) for r in results] if isinstance(results, list) else []
            )
            save_failure(args, exc, stage=stage, group=group, generated=generated)
        except Exception as recording:
            import sys

            print(f"Failure recording also failed: {recording}", file=sys.stderr, flush=True)
        raise
    finally:
        shutdown_engine(engine, failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--engine", choices=["mini", "sglang"], default="mini")
    parser.add_argument(
        "--expected-reference-revision",
        default="baa9befa381511cd4d56f83980951e5a45fffc6d",
        help="Full Git revision of the official reference checkout.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-running-requests", type=int, help="Engine capacity; defaults to --batch-size."
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--dcp-size", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument(
        "--distributed-backend", choices=["auto", "torch", "pynccl", "mnnvl"], default="auto"
    )
    parser.add_argument("--distributed-provider", help="Mini communication provider module.")
    parser.add_argument(
        "--moe-backend", choices=["auto", "marlin", "flashinfer-ep"], default="marlin"
    )
    parser.add_argument("--mxfp4-backend", choices=["torch", "flashinfer"], default="torch")
    parser.add_argument("--activation-precision", choices=["bf16", "mxfp8"], default="bf16")
    parser.add_argument("--kv-cache-dtype", choices=["bfloat16", "fp8_e4m3"], default="bfloat16")
    parser.add_argument(
        "--recurrent-state-dtype", choices=["float32", "bfloat16"], default="float32"
    )
    for stage in ("", "prefill-", "decode-"):
        parser.add_argument(f"--sglang-{stage}attention-backend")
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument(
        "--cuda-graph-max-bs",
        type=int,
        help="Capture capacity; requires --cuda-graph and defaults to --batch-size.",
    )
    parser.add_argument("--flashinfer-autotune-skip-ops", nargs="*", default=[])
    parser.add_argument("--weight-loader-prefetch-checkpoints", action="store_true", default=None)
    parser.add_argument("--weight-loader-prefetch-num-threads", type=int)
    parser.add_argument("--model-loader-extra-config")
    args = parser.parse_args()
    if (
        args.weight_loader_prefetch_num_threads is not None
        and args.weight_loader_prefetch_num_threads < 1
    ):
        parser.error("--weight-loader-prefetch-num-threads must be positive")
    if args.engine == "mini" and args.model_loader_extra_config is not None:
        parser.error("--model-loader-extra-config applies only to --engine sglang")
    if args.moe_backend == "flashinfer-ep" and (args.engine != "mini" or args.ep_size <= 1):
        parser.error("flashinfer-ep requires --engine mini and --ep-size greater than 1")
    if args.engine == "sglang" and not re.fullmatch(
        r"[0-9a-f]{40}", args.expected_reference_revision
    ):
        parser.error("--expected-reference-revision requires a full commit hash")
    if args.moe_backend == "marlin" and args.activation_precision != "bf16":
        parser.error("Marlin requires BF16 expert activations")
    if (
        args.engine == "mini"
        and args.activation_precision == "mxfp8"
        and args.mxfp4_backend != "flashinfer"
    ):
        parser.error("MXFP8 evaluation requires the native FlashInfer expert backend")
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
    if args.max_running_requests is None:
        args.max_running_requests = args.batch_size
    if args.max_running_requests < args.batch_size:
        parser.error("--max-running-requests must be at least --batch-size")
    if args.cuda_graph_max_bs is not None and not args.cuda_graph:
        parser.error("--cuda-graph-max-bs requires --cuda-graph")
    if args.cuda_graph_max_bs is None:
        args.cuda_graph_max_bs = args.batch_size if args.cuda_graph else 0
    if args.cuda_graph and not 1 <= args.cuda_graph_max_bs <= args.max_running_requests:
        parser.error("Graph capacity must be positive and no larger than --max-running-requests")
    if args.distributed_backend == "mnnvl" and not args.distributed_provider:
        parser.error("The mnnvl backend requires --distributed-provider")
    if args.engine != "mini" and (
        args.distributed_backend != "auto" or args.distributed_provider is not None
    ):
        parser.error("Communication backend/provider options apply only to --engine mini")
    if (
        (args.output / "report.json").exists()
        or (args.output / "responses.jsonl").exists()
        or any(args.output.glob("failure-rank*.json"))
    ):
        raise FileExistsError("Choose a fresh output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    saved = json.loads(args.cases.read_text())
    cases = saved["examples"][: args.limit or None]
    if (
        not cases
        or max(len(case["input_ids"]) + args.max_new_tokens for case in cases) > args.max_context
    ):
        raise ValueError("Empty cases or insufficient context")
    stop_ids = set(saved["stop_token_ids"])
    if args.engine == "mini":
        from minisgl.engine.engine import WorkerOwner
        from minisgl.server.launch import _abort_worker

        # The CLI runs in each fresh torchrun worker, as in the serving launcher.
        owner = WorkerOwner(fresh_worker=True, abort_callback=_abort_worker)
        run_mini(args, cases, stop_ids, worker_owner=owner)
    else:
        run_sglang(args, cases, stop_ids)


if __name__ == "__main__":
    main()
