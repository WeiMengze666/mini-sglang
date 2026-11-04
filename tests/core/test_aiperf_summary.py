"""Serving reports must preserve tool boundaries, invalid runs and workload differences."""

import json
import runpy
from pathlib import Path

import pytest

module = runpy.run_path(str(Path(__file__).parents[2] / "benchmark/online/summarize_aiperf.py"))
summarize_run, compare_runs = module["summarize_run"], module["compare_runs"]


def make_run(tmp_path, name, *, complete=True, valid=True, seed=42):
    root = tmp_path / name
    root.mkdir()
    run = {
        "status": "completed",
        "benchmark_returncode": 0,
        "benchmark_argv": ["aiperf", "profile", "--benchmark-duration", "900"],
        "server_argv": [name, "serve"],
        "packages": {"aiperf": "0.13.0"},
        "profile_sha256": "0" * 64,
    }
    export = {
        "benchmark_id": name,
        "is_complete": complete,
        "metadata": {"submission_valid": valid},
        "was_cancelled": False,
        "error_summary": {},
        "input_config": {"seed": seed, "artifacts": {"dir": name, "level": "raw"}},
        "output_token_throughput": {"avg": 22.84357349486434, "unit": "tokens/sec"},
        "benchmark_duration": {"avg": 926.866, "unit": "seconds"},
        "completed_request_count": {"avg": 88},
        "total_usage_completion_tokens": {"avg": 21473},
        "total_usage_prompt_tokens": {"avg": 100},
        "total_usage_prompt_cache_read_tokens": {"avg": 50},
        "request_error_rate": {"avg": 0},
    }
    (root / "run.json").write_text(json.dumps(run))
    (root / "profile_export_aiperf.json").write_text(json.dumps(export))
    (root / "aiperf.log").write_text(
        "PhaseRecordsStats(phase=CreditPhase.PROFILING, final_requests_completed=88, "
        "final_requests_cancelled=20, final_request_errors=0)\n"
    )
    return root


def test_summary_keeps_original_metrics_and_cancellation_boundaries(tmp_path):
    root = make_run(tmp_path, "reference")
    result = summarize_run(root, prompt_token_offset=3)
    assert result["metrics"]["output_token_throughput"]["avg"] == 22.84357349486434
    assert result["metrics"]["benchmark_duration"]["avg"] == 926.866
    assert result["metrics"]["total_usage_prompt_tokens"]["avg"] == 100
    assert result["input_accounting"]["model_prompt_tokens"] == 364
    assert result["input_accounting"]["cache_read_pct_of_model_tokens"] == pytest.approx(5000 / 364)
    assert result["profiling_phase_records"]["final_requests_cancelled"] == 20
    assert result["profiling_phase_records"]["final_request_errors"] == 0
    assert (
        json.loads((root / "profile_export_aiperf.json").read_text())["input_config"]["artifacts"][
            "dir"
        ]
        == "reference"
    )


@pytest.mark.parametrize("complete,valid", [(False, True), (True, False)])
def test_incomplete_or_invalid_exports_are_rejected(tmp_path, complete, valid):
    root = make_run(tmp_path, "bad", complete=complete, valid=valid)
    with pytest.raises(ValueError, match="incomplete or invalid"):
        summarize_run(root)


def test_failed_run_is_rejected_before_export(tmp_path):
    root = make_run(tmp_path, "failed")
    data = json.loads((root / "run.json").read_text())
    data["benchmark_returncode"] = 1
    (root / "run.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="did not complete"):
        summarize_run(root)


def test_workload_comparison_only_ignores_output_directory(tmp_path):
    left, right = make_run(tmp_path, "left"), make_run(tmp_path, "right")
    result = compare_runs(left, right)
    assert result["same_resolved_workload_except_artifact_directory"]
    assert "separate evidence" in result["acceptance"]
    data = json.loads((right / "profile_export_aiperf.json").read_text())
    data["input_config"]["seed"] = 43
    (right / "profile_export_aiperf.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="resolved_workload"):
        compare_runs(left, right)


def test_missing_phase_records_are_unknown_not_zero_and_duplicate_exports_rejected(tmp_path):
    root = make_run(tmp_path, "records")
    (root / "aiperf.log").write_text("")
    assert summarize_run(root)["profiling_phase_records"] is None
    nested = root / "extra"
    nested.mkdir()
    (nested / "profile_export_aiperf.json").write_text("{}")
    with pytest.raises(ValueError, match="exactly one"):
        summarize_run(root)


def test_missing_package_metadata_is_unknown(tmp_path):
    left, right = make_run(tmp_path, "left"), make_run(tmp_path, "right")
    for root in (left, right):
        path = root / "run.json"
        data = json.loads(path.read_text())
        data.pop("packages")
        path.write_text(json.dumps(data))
    assert compare_runs(left, right)["same_packages"] is None
