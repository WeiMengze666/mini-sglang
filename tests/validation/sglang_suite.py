"""Independent released-weight suite; run in the official SGLang environment."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from provenance import execution_identity


def benchmark(engine, prompt, tokens, batch_sizes, output, repeats):
    """Warm fixed-length Engine calls, including scheduler and IPC overhead."""
    batches = []
    for size in dict.fromkeys(batch_sizes):

        def generate():
            return engine.generate(
                input_ids=[prompt] * size,
                sampling_params={"temperature": 0, "max_new_tokens": tokens, "ignore_eos": True},
            )

        generate()
        seconds, outputs = [], []
        for _ in range(repeats):
            start = time.monotonic()
            results = generate()
            seconds.append(time.monotonic() - start)
            if isinstance(results, dict):
                results = [results]
            if len(results) != size or any(len(r["output_ids"]) != tokens for r in results):
                raise AssertionError("Fixed-length benchmark returned an incomplete batch")
            outputs.append([r["output_ids"] for r in results])
        median = statistics.median(seconds)
        batches.append(
            {
                "batch_size": size,
                "warm_total_seconds": seconds,
                "warm_token_ids": outputs,
                "median_total_seconds": median,
                "generation_tokens_per_second": size * tokens / median,
            }
        )
        output.write_text(
            json.dumps(
                {
                    "boundary": "synchronous Engine.generate including prefill, decode, scheduler, sampling and IPC; no logprobs",
                    "complete": len(batches) == len(set(batch_sizes)),
                    "tokens_per_request": tokens,
                    "prompt_tokens": len(prompt),
                    "warmup_per_batch": 1,
                    "repeats": repeats,
                    "batches": batches,
                },
                indent=2,
            )
            + "\n"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-context", type=int, default=4096)
    parser.add_argument("--benchmark-batch-sizes", type=int, nargs="*", default=[])
    parser.add_argument("--benchmark-repeats", type=int, default=3)
    parser.add_argument("--flashinfer-autotune-skip-ops", nargs="*", default=[])
    args = parser.parse_args()
    if (
        args.tokens < 1
        or args.benchmark_repeats < 1
        or any(size < 1 for size in args.benchmark_batch_sizes)
    ):
        parser.error("tokens, benchmark repeats and batch sizes must be positive")
    from sglang import Engine

    args.output.mkdir(parents=True, exist_ok=True)
    identity = execution_identity("sglang")
    if (
        identity["revision"] != "baa9befa381511cd4d56f83980951e5a45fffc6d"
        or identity["tracked_changes"]
    ):
        raise RuntimeError("Official reference checkout differs from the pinned clean source")
    identity["configuration"] = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    (args.output / "provenance.json").write_text(json.dumps(identity, indent=2) + "\n")
    dataset = torch.load(args.cases, map_location="cpu", weights_only=True)
    engine = Engine(
        model_path=args.model,
        tp_size=8,
        dcp_size=2,
        ep_size=8,
        dtype="bfloat16",
        trust_remote_code=True,
        context_length=args.max_context,
        mem_fraction_static=0.9,
        max_running_requests=max(4, max(args.benchmark_batch_sizes, default=1)),
        disable_radix_cache=True,
        mamba_ssm_dtype="float32",
        chunked_prefill_size=args.chunk_size,
        page_size=64,
        flashinfer_autotune_skip_ops=args.flashinfer_autotune_skip_ops,
        log_level="info",
    )
    try:
        results = []
        config = json.loads((Path(args.model) / "config.json").read_text())
        vocab = int(config.get("text_config", config)["vocab_size"])
        for case in dataset:
            first = engine.generate(
                input_ids=case["input_ids"],
                sampling_params={"temperature": 0, "max_new_tokens": 1, "ignore_eos": True},
                return_logprob=True,
                token_ids_logprob=list(range(vocab)),
            )
            values = first["meta_info"]["output_token_ids_logprobs"][0]
            if len(values) != vocab:
                raise AssertionError("Official engine did not return the complete vocabulary")
            logprobs = torch.full((vocab,), torch.nan, dtype=torch.float32)
            for value, index, _ in values:
                logprobs[index] = value
            torch.save(
                {"input_ids": case["input_ids"], "logprobs": logprobs},
                args.output / (case["name"] + ".pt"),
            )
            result = engine.generate(
                input_ids=case["input_ids"],
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": args.tokens,
                    "ignore_eos": True,
                },
                return_logprob=True,
                top_logprobs_num=20,
            )
            results.append(
                {
                    "name": case["name"],
                    "prompt_tokens": len(case["input_ids"]),
                    "token_ids": result["output_ids"],
                    "text": result["text"],
                    "meta_info": result["meta_info"],
                }
            )
            (args.output / "report.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        key: results[-1][key]
                        for key in ("name", "prompt_tokens", "token_ids", "text")
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if args.benchmark_batch_sizes:
            benchmark(
                engine,
                dataset[0]["input_ids"],
                args.tokens,
                args.benchmark_batch_sizes,
                args.output / "benchmark.json",
                args.benchmark_repeats,
            )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
