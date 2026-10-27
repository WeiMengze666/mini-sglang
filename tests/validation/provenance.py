"""Identify actual sources and libraries in exported or dirty validation runs."""

import hashlib
import importlib
import importlib.metadata
import subprocess
from pathlib import Path


def execution_identity(module_name):
    module = importlib.import_module(module_name)
    package = (
        Path(module.__file__).resolve().parent
        if module.__file__
        else Path(next(iter(module.__path__))).resolve()
    )
    root = package.parent.parent

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    libraries = {}
    for name in ("torch", "transformers", "flashinfer-python", "triton", "safetensors"):
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libraries[name] = None
    return {
        "source_root": str(root),
        "revision": git("rev-parse", "HEAD"),
        "tracked_changes": git("status", "--porcelain", "--untracked-files=no"),
        "libraries": libraries,
        "source_sha256": {
            str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(package.rglob("*.py"))
        },
    }
