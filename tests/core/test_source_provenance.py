"""An archived source tree must not inherit its enclosing repository's HEAD."""

import hashlib
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.mark.parametrize("marker", [None, "directory", "worktree_file"])
def test_source_identity_requires_its_own_checkout_root(tmp_path, monkeypatch, marker):
    import subprocess

    outer = tmp_path / "enclosing-repository"
    (outer / ".git").mkdir(parents=True)
    root = outer / "archive"
    package = root / "python" / "fixture_package"
    package.mkdir(parents=True)
    source = package / "__init__.py"
    source.write_text("VERSION = 1\n")
    if marker == "directory":
        (root / ".git").mkdir()
    elif marker == "worktree_file":
        (root / ".git").write_text("gitdir: elsewhere\n")
    module = ModuleType("fixture_package")
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    calls = []

    def git(argv, **kwargs):
        assert marker is not None, "Archived package inherited enclosing Git metadata"
        assert argv[:3] == ["git", "-C", str(root)]
        calls.append(argv[3:])
        return "package-commit\n" if argv[3] == "rev-parse" else ""

    monkeypatch.setattr(subprocess, "check_output", git)
    identity = runpy.run_path(str(Path(__file__).parents[1] / "validation/provenance.py"))[
        "execution_identity"
    ](module.__name__)
    assert identity["revision"] == ("package-commit" if marker is not None else None)
    assert identity["tracked_changes"] == ("" if marker is not None else None)
    assert identity["source_sha256"] == {
        "__init__.py": hashlib.sha256(source.read_bytes()).hexdigest()
    }
    assert len(calls) == (2 if marker is not None else 0)
