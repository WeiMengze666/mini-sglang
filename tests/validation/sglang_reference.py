"""Sequential official-engine reference for a saved released_k3.py prompt.

Run in SGLang's own environment under gpuq, after the mini engine has exited.
No inference-engine runtime code is imported by the mini framework.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=64)
    parser.add_argument("--full-vocab", action="store_true")
    parser.add_argument("--dcp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--enable-graphs", action="store_true")
    parser.add_argument("--enable-overlap", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 0 or args.tokens < 1:
        raise ValueError("repeat/tokens must be positive and warmup must be nonnegative")
    from sglang import Engine

    data = torch.load(args.prompt_file, map_location="cpu", weights_only=True)
    engine = Engine(
        model_path=args.model,
        tp_size=8,
        dcp_size=args.dcp_size,
        ep_size=args.ep_size,
        dtype="bfloat16",
        trust_remote_code=True,
        context_length=256,
        mem_fraction_static=0.9,
        max_running_requests=4,
        disable_cuda_graph=not args.enable_graphs,
        disable_overlap_schedule=not args.enable_overlap,
        disable_radix_cache=True,
        mamba_ssm_dtype="float32",
        chunked_prefill_size=args.chunk_size,
        page_size=args.page_size,
        log_level="info",
    )
    try:
        tokens = 1 if args.full_vocab else args.tokens

        def generate(count):
            return engine.generate(
                input_ids=data["input_ids"].tolist(),
                sampling_params={"temperature": 0, "max_new_tokens": count, "ignore_eos": True},
                return_logprob=True,
                top_logprobs_num=20,
                token_ids_logprob=list(range(data["logits"].shape[-1]))
                if args.full_vocab
                else None,
            )

        for _ in range(args.warmup):
            generate(tokens)
        runs = []
        for _ in range(args.repeat):
            prefill = None
            if args.repeat > 1 and tokens > 1:
                tick = time.monotonic()
                generate(1)
                prefill = time.monotonic() - tick
            tick = time.monotonic()
            result = generate(tokens)
            total = time.monotonic() - tick
            runs.append(
                {
                    "total_seconds": total,
                    "prefill_seconds": prefill,
                    "decode_seconds_per_token_estimate": (total - prefill) / (tokens - 1)
                    if prefill is not None
                    else None,
                }
            )
        result["benchmark"] = {
            "boundary": "synchronous Engine.generate including scheduler and logprobs",
            "graphs": args.enable_graphs,
            "overlap": args.enable_overlap,
            "dcp_size": args.dcp_size,
            "ep_size": args.ep_size,
            "tokens": tokens,
            "warmup": args.warmup,
            "runs": runs,
            "median_total_seconds": statistics.median(r["total_seconds"] for r in runs),
        }
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n")
        print(
            json.dumps(
                {
                    "text": result["text"],
                    "output_ids": result["output_ids"],
                    "meta_keys": list(result["meta_info"]),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
