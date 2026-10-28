"""Compare paired task runs; the CLI requires the full GSM8K split by default."""

import argparse
import json
from pathlib import Path


def read_run(folder):
    summary = json.loads((folder / "report.json").read_text())
    rows = [json.loads(line) for line in (folder / "responses.jsonl").read_text().splitlines()]
    indexed = {row["id"]: row for row in rows}
    if not summary["complete"] or len(indexed) != len(rows) or len(rows) != summary["expected"]:
        raise ValueError(f"Incomplete or duplicated task results: {folder}")
    if sum(row["correct"] for row in rows) != summary["correct"]:
        raise ValueError(f"Task summary differs from its saved responses: {folder}")
    return summary, indexed


def compare(left, right, *, expected_examples=None):
    ls, lr = read_run(left)
    rs, rr = read_run(right)
    if ls["cases_sha256"] != rs["cases_sha256"] or lr.keys() != rr.keys():
        raise ValueError("Task runs do not contain the same tokenized examples")
    if expected_examples is not None and len(lr) != expected_examples:
        raise ValueError(f"Expected {expected_examples} examples, got {len(lr)}")
    for field in ("batch_size", "max_new_tokens", "max_context"):
        left_value = ls.get("configuration", {}).get(field)
        right_value = rs.get("configuration", {}).get(field)
        if left_value is None or right_value is None or left_value != right_value:
            raise ValueError(f"Task generation protocol differs or is missing: {field}")
    if any(lr[key]["gold"] != rr[key]["gold"] for key in lr):
        raise ValueError("Task gold answers differ")
    left_only = [key for key in lr if lr[key]["correct"] and not rr[key]["correct"]]
    right_only = [key for key in lr if rr[key]["correct"] and not lr[key]["correct"]]
    return {
        "left": str(left),
        "right": str(right),
        "examples": len(lr),
        "cases_sha256": ls["cases_sha256"],
        "left_accuracy": ls["accuracy"],
        "right_accuracy": rs["accuracy"],
        "accuracy_delta_percentage_points": (ls["correct"] - rs["correct"]) / len(lr) * 100,
        "both_correct": sum(lr[key]["correct"] and rr[key]["correct"] for key in lr),
        "both_wrong": sum(not lr[key]["correct"] and not rr[key]["correct"] for key in lr),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "same_valid_answer": sum(
            lr[key]["predicted"] is not None and lr[key]["predicted"] == rr[key]["predicted"]
            for key in lr
        ),
        "identical_token_ids": sum(lr[key]["token_ids"] == rr[key]["token_ids"] for key in lr),
        "left_invalid": ls["invalid_answers"],
        "right_invalid": rs["invalid_answers"],
        "left_truncated": ls["truncated"],
        "right_truncated": rs["truncated"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-examples",
        type=int,
        default=1319,
        help="Required paired case count; defaults to the complete GSM8K test split",
    )
    args = parser.parse_args()
    if args.expected_examples < 1:
        parser.error("expected-examples must be positive")
    result = compare(args.left, args.right, expected_examples=args.expected_examples)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
