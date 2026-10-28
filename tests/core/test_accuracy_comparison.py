import json
import runpy
from pathlib import Path

import pytest

compare = runpy.run_path(str(Path(__file__).parents[1] / "validation/compare_accuracy.py"))[
    "compare"
]


def write_run(folder, answers, *, digest="same", complete=True):
    folder.mkdir()
    rows = [
        {
            "id": index,
            "gold": "1",
            "predicted": answer,
            "correct": answer == "1",
            "token_ids": [index],
            "truncated": False,
        }
        for index, answer in enumerate(answers)
    ]
    summary = {
        "complete": complete,
        "expected": len(rows),
        "cases_sha256": digest,
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "invalid_answers": answers.count(None),
        "truncated": 0,
        "configuration": {"batch_size": 8, "max_new_tokens": 1024, "max_context": 4096},
    }
    (folder / "report.json").write_text(json.dumps(summary))
    (folder / "responses.jsonl").write_text("\n".join(json.dumps(row) for row in rows))


def test_paired_accuracy_counts_each_outcome_once(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    write_run(left, ["1", "1", "2", None])
    write_run(right, ["1", "2", "1", None])
    result = compare(left, right)
    assert result["examples"] == 4
    assert result["both_correct"] == result["both_wrong"] == 1
    assert result["left_only_correct"] == [1] and result["right_only_correct"] == [2]
    assert result["accuracy_delta_percentage_points"] == 0
    assert result["same_valid_answer"] == 1


@pytest.mark.parametrize("digest,complete", [("different", True), ("same", False)])
def test_different_or_incomplete_cases_are_not_comparable(tmp_path, digest, complete):
    left, right = tmp_path / "left", tmp_path / "right"
    write_run(left, ["1"])
    write_run(right, ["1"], digest=digest, complete=complete)
    with pytest.raises(ValueError):
        compare(left, right)


def test_matching_partial_runs_cannot_satisfy_full_split_gate(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    write_run(left, ["1"] * 8)
    write_run(right, ["1"] * 8)
    with pytest.raises(ValueError, match="Expected 1319 examples"):
        compare(left, right, expected_examples=1319)


@pytest.mark.parametrize("field", ["batch_size", "max_new_tokens", "max_context"])
def test_changed_generation_protocol_is_rejected(tmp_path, field):
    left, right = tmp_path / "left", tmp_path / "right"
    write_run(left, ["1"])
    write_run(right, ["1"])
    path = right / "report.json"
    summary = json.loads(path.read_text())
    summary["configuration"][field] //= 2
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="generation protocol"):
        compare(left, right)
