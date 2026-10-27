"""Compare independently saved full-vocabulary logprobs and greedy generations."""

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mini", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mini = json.loads((args.mini / "report.json").read_text())
    official = {
        case["name"]: case for case in json.loads((args.reference / "report.json").read_text())
    }
    results = []
    for case in mini:
        name = case["name"]
        actual = torch.load(args.mini / (name + ".pt"), map_location="cpu", weights_only=True)
        expected = torch.load(
            args.reference / (name + ".pt"), map_location="cpu", weights_only=True
        )
        assert actual["input_ids"] == expected["input_ids"]
        logits = actual["full"]["logits"].float().reshape(-1)
        logp = logits.log_softmax(-1)
        ref = expected["logprobs"]
        if (
            logp.shape != ref.shape
            or not torch.isfinite(logp).all()
            or not torch.isfinite(ref).all()
        ):
            raise AssertionError("Invalid full-vocabulary comparison")
        chunkp = actual["chunked"]["logits"].float().reshape(-1).log_softmax(-1)
        tokens = official[name]["token_ids"]
        prefix = 0
        for a, b in zip(case["token_ids"], tokens):
            if a != b:
                break
            prefix += 1
        results.append(
            {
                "name": name,
                "prompt_tokens": case["prompt_tokens"],
                "first_token_match": int(logp.argmax()) == int(ref.argmax()),
                "generation_common_prefix": prefix,
                "generation_exact_match": case["token_ids"] == tokens,
                "kl_reference_to_mini": float((ref.exp() * (ref - logp)).sum()),
                "max_logprob_difference": float((ref - logp).abs().max()),
                "top20_overlap": len(
                    set(logp.topk(20).indices.tolist()) & set(ref.topk(20).indices.tolist())
                ),
                "chunked_first_token_match": int(logp.argmax()) == int(chunkp.argmax()),
                "kl_full_to_chunked": float((logp.exp() * (logp - chunkp)).sum()),
            }
        )
    report = {
        "scope": "greedy generation and full-vocabulary numerical comparison; not a benchmark accuracy score",
        "cases": results,
        "first_token_matches": sum(r["first_token_match"] for r in results),
        "generation_exact_matches": sum(r["generation_exact_match"] for r in results),
        "case_count": len(results),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
