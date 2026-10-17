"""Execution identity for the offline acceptance harness (no network probes)."""

import hashlib
import importlib.metadata
import platform
import subprocess
from pathlib import Path

import torch


def provenance(device, *, references_run=False, executed=True):
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    versions = {}
    for package in ("torch", "transformers", "safetensors", "triton", "flashinfer-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    # Hash actual execution sources too: HEAD alone cannot identify dirty/exported runs.
    paths = sorted((root / "python/minisgl").rglob("*.py"))
    paths += sorted((root / "tests/models").glob("*kimi*.py"))
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    device = torch.device(device)
    cpu_model = platform.processor() or platform.machine()
    if device.type == "cpu" and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    backend = {"kda": "minisgl_eager_sequential_fp32_recurrence", "mla": "minisgl_eager_pytorch"}
    return {
        "implementation": {
            "revision": git("rev-parse", "HEAD"),
            "worktree_status": git("status", "--porcelain"),
            "source_sha256": sources,
        },
        "references": {
            "executed": references_run,
            "hf_revision": "f831ab66814297da540d832a5235f8e904f29d06",
            "fla_revision": "14bef3d9ec98ee359c9a9c7860795d91675dd091",
            "scope": "pinned_operator_checks" if references_run else "not_run",
        },
        "backends": {
            "executed": executed,
            "prefill": dict(backend),
            "decode": dict(backend),
            "sglang_gpu_kernels_executed": False,
        },
        "libraries": versions,
        "python": platform.python_version(),
        "device": {
            "requested": str(device),
            "model": (torch.cuda.get_device_name(device) if executed else None)
            if device.type == "cuda"
            else cpu_model,
            "cuda_runtime": torch.version.cuda,
        },
    }
