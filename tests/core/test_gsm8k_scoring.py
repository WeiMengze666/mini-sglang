import runpy
from pathlib import Path

import pytest

answer_number = runpy.run_path(str(Path(__file__).parents[1] / "validation/gsm8k_suite.py"))[
    "answer_number"
]


@pytest.mark.parametrize(
    "text,expected",
    [
        ("#### 1,000", "1E+3"),
        ("#### 1000.0", "1E+3"),
        ("#### -0.0", "0"),
        ("#### -12.5<|end_of_msg|>", "-12.5"),
        ("#### 1\nCorrection:\n#### 2", "2"),
        ("Working: 17 * 23 = 391", None),
        ("#### 1/2", None),
        ("#### 10-2", None),
        ("#### 5.0.2", None),
        ("#### 42.", "42"),
    ],
)
def test_final_explicit_answer_and_numeric_normalization(text, expected):
    assert answer_number(text) == expected
