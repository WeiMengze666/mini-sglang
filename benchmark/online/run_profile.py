"""Run a saved server/AIPerf profile in one GPUq job, retaining raw artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


def stop_process_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def snapshot(url, path):
    try:
        with urlopen(url, timeout=10) as response:
            path.write_bytes(response.read())
    except Exception as exc:
        path.with_suffix(".error.txt").write_text(str(exc))


def run_profile(
    profile_path, server_key, output, *, readiness_timeout=1800, benchmark_timeout=3600
):
    profile_path, output = Path(profile_path).resolve(), Path(output).resolve()
    profile = json.loads(profile_path.read_text())
    server_command = profile[server_key]["argv"]
    benchmark_command = profile["benchmark"]["argv"]
    if not server_command or not benchmark_command:
        raise ValueError("Server and benchmark commands must be explicit argument arrays")
    options = benchmark_command
    url = options[options.index("--url") + 1].rstrip("/")
    address = urlparse(url)
    try:
        with socket.create_connection((address.hostname, address.port or 80), timeout=1):
            raise RuntimeError(f"Benchmark port is already in use: {url}")
    except OSError:
        pass
    output.mkdir(parents=True, exist_ok=False)
    packages = {}
    for name in ("aiperf", "torch", "triton", "flashinfer-python", "transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result = {
        "status": "starting",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
        "server_key": server_key,
        "server_argv": server_command,
        "benchmark_argv": benchmark_command,
        "packages": packages,
        "readiness_timeout": readiness_timeout,
        "benchmark_timeout": benchmark_timeout,
    }
    (output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    server = benchmark = None
    ready = False
    try:
        with (output / "server.log").open("w") as server_log:
            server = subprocess.Popen(
                server_command, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            deadline = time.monotonic() + readiness_timeout
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited with {server.returncode}; see server.log")
                try:
                    with urlopen(url + "/v1/models", timeout=1) as response:
                        if response.status == 200:
                            ready = True
                            break
                except OSError:
                    pass
                time.sleep(0.5)
            if not ready:
                raise TimeoutError("Server readiness timed out")
            snapshot(url + "/get_server_info", output / "server-info.json")
            snapshot(url + "/metrics", output / "metrics-before.txt")
            result["ready_utc"] = datetime.now(timezone.utc).isoformat()
            result["status"] = "benchmarking"
            (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")
            with (output / "aiperf.log").open("w") as benchmark_log:
                benchmark = subprocess.Popen(
                    benchmark_command,
                    cwd=output,
                    stdout=benchmark_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                result["benchmark_returncode"] = benchmark.wait(timeout=benchmark_timeout)
            if result["benchmark_returncode"]:
                raise RuntimeError(
                    f"AIPerf exited with {result['benchmark_returncode']}; see aiperf.log"
                )
            result["status"] = "completed"
            return result
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop_process_group(benchmark)
        if ready:
            snapshot(url + "/metrics", output / "metrics-after.txt")
        stop_process_group(server)
        result["finished_utc"] = datetime.now(timezone.utc).isoformat()
        (output / "run.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--server-key", default="reference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--readiness-timeout", type=float, default=1800)
    parser.add_argument("--benchmark-timeout", type=float, default=3600)
    args = parser.parse_args()
    if os.environ.get("MINISGL_GPUQ") != "1":
        parser.error("Launch this server/benchmark bundle through GPUq (MINISGL_GPUQ=1)")

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    run_profile(
        args.profile,
        args.server_key,
        args.output,
        readiness_timeout=args.readiness_timeout,
        benchmark_timeout=args.benchmark_timeout,
    )


if __name__ == "__main__":
    main()
