"""Sequential official-engine reference for a saved released_k3.py prompt.

Run in SGLang's own environment under gpuq, after the mini engine has exited.
No inference-engine runtime code is imported by the mini framework.
"""

import argparse
import json
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
    args = parser.parse_args()
    from sglang import Engine

    data = torch.load(args.prompt_file, map_location="cpu", weights_only=True)
    engine = Engine(
        model_path=args.model,
        tp_size=8,
        dtype="bfloat16",
        trust_remote_code=True,
        context_length=256,
        mem_fraction_static=0.9,
        max_running_requests=4,
        disable_cuda_graph=True,
        disable_overlap_schedule=True,
        disable_radix_cache=True,
        mamba_ssm_dtype="float32",
        chunked_prefill_size=args.chunk_size,
        page_size=args.page_size,
        log_level="info",
    )
    try:
        result = engine.generate(
            input_ids=data["input_ids"].tolist(),
            sampling_params={
                "temperature": 0,
                "max_new_tokens": 1 if args.full_vocab else 8,
                "ignore_eos": True,
            },
            return_logprob=True,
            top_logprobs_num=20,
            token_ids_logprob=list(range(data["logits"].shape[-1])) if args.full_vocab else None,
        )
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
