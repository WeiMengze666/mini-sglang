"""Compare completed AIPerf exports without changing their statistics or time boundaries."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path

METRICS = (
    "output_token_throughput",
    "request_throughput",
    "time_to_first_token",
    "inter_token_latency",
    "request_latency",
    "request_error_rate",
    "completed_request_count",
    "effective_concurrency",
    "benchmark_duration",
    "total_usage_completion_tokens",
    "total_usage_prompt_tokens",
    "total_usage_prompt_cache_read_tokens",
    "overall_usage_prompt_cache_read_pct",
)


def summarize_run(directory: Path, *, prompt_token_offset: int = 0):
    directory = Path(directory).resolve()
    run_path = directory / "run.json"
    run = json.loads(run_path.read_text())
    if run.get("status") != "completed" or run.get("benchmark_returncode") != 0:
        raise ValueError(f"Run did not complete successfully: {directory}")
    files = list(directory.rglob("profile_export_aiperf.json"))
    if len(files) != 1:
        raise ValueError(f"Expected exactly one AIPerf export in {directory}; found {len(files)}")
    export_path = files[0]
    export = json.loads(export_path.read_text())
    if not export.get("is_complete") or not export.get("metadata", {}).get("submission_valid"):
        raise ValueError(f"AIPerf export is incomplete or invalid: {export_path}")
    metrics = {key: export[key] for key in METRICS if key in export}
    accounting = {"prompt_token_offset_per_completed_request": prompt_token_offset}
    if "total_usage_prompt_tokens" in metrics and "completed_request_count" in metrics:
        total = (
            metrics["total_usage_prompt_tokens"]["avg"]
            + prompt_token_offset * metrics["completed_request_count"]["avg"]
        )
        accounting["model_prompt_tokens"] = total
        if total > 0 and "total_usage_prompt_cache_read_tokens" in metrics:
            accounting["cache_read_pct_of_model_tokens"] = (
                100 * metrics["total_usage_prompt_cache_read_tokens"]["avg"] / total
            )
    records = None
    log = directory / "aiperf.log"
    if log.exists():
        for line in log.read_text().splitlines():
            if "PhaseRecordsStats(phase=CreditPhase.PROFILING" in line:
                fields = dict(
                    re.findall(
                        r"(final_requests_completed|final_requests_cancelled|final_request_errors)=(\d+)",
                        line,
                    )
                )
                if len(fields) == 3:
                    records = {key: int(value) for key, value in fields.items()}
    workload = copy.deepcopy(export["input_config"])
    workload.get("artifacts", {}).pop("dir", None)
    return {
        "directory": str(directory),
        "benchmark_id": export["benchmark_id"],
        "is_complete": export["is_complete"],
        "submission_valid": export["metadata"]["submission_valid"],
        "was_cancelled": export.get("was_cancelled"),
        "error_summary": export.get("error_summary"),
        "metrics": metrics,
        "input_accounting": accounting,
        "profiling_phase_records": records,
        "warmup_metrics": export.get("warmup_metrics"),
        "benchmark_argv": run["benchmark_argv"],
        "server_argv": run["server_argv"],
        "packages": run.get("packages"),
        "resolved_workload": workload,
        "export_path": str(export_path),
        "export_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
        "run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        "profile_sha256": run["profile_sha256"],
    }


def compare_runs(reference, candidate, *, reference_offset=0, candidate_offset=0):
    left = summarize_run(reference, prompt_token_offset=reference_offset)
    right = summarize_run(candidate, prompt_token_offset=candidate_offset)
    for key in ("benchmark_argv", "resolved_workload"):
        if left[key] != right[key]:
            raise ValueError(f"Benchmark mismatch: {key}")
    return {
        "scope": __doc__,
        "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "same_benchmark_argv": True,
        "same_resolved_workload_except_artifact_directory": True,
        "same_packages": (
            left["packages"] == right["packages"]
            if left["packages"] is not None and right["packages"] is not None
            else None
        ),
        "reference": left,
        "candidate": right,
        "acceptance": "Workload equality only. Model quality, precision, state/cache policy and hardware parity require separate evidence. No fair performance ratio is inferred.",
        "timing": "Metric objects are copied from AIPerf. Nominal duration is not substituted for the tool's measurement boundary. Deadline request cancellations remain separate from HTTP errors.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference-prompt-token-offset", type=int, default=0)
    parser.add_argument("--candidate-prompt-token-offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    report = compare_runs(
        args.reference,
        args.candidate,
        reference_offset=args.reference_prompt_token_offset,
        candidate_offset=args.candidate_prompt_token_offset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
