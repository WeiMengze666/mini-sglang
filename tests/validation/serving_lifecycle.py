"""Opt-in synthetic HTTP/scheduler gate. Run through gpuq; cleans only its own server."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx


def fixture(root):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
    import run_kimi_runtime_gate as runtime

    config = runtime.tiny_config()
    config.vocab_size = 512
    runtime.tiny_config = lambda: config
    runtime.generate_fixtures(root)
    model = root / "dense"
    (model / "tiktoken.model").write_bytes(
        b"".join(base64.b64encode(bytes([i])) + f" {i}\n".encode() for i in range(256))
    )
    names = {
        256: "[BOS]",
        257: "[EOS]",
        258: "<|end_of_msg|>",
        259: "<|open|>",
        260: "<|close|>",
        261: "<|sep|>",
        510: "[UNK]",
        511: "[PAD]",
    }
    metadata = {
        "tokenizer_class": "TikTokenTokenizer",
        "bos_token": "[BOS]",
        "eos_token": "[EOS]",
        "unk_token": "[UNK]",
        "pad_token": "[PAD]",
        "additional_special_tokens": ["<|end_of_msg|>"],
        "added_tokens_decoder": {
            str(i): {"content": text, "special": i not in (259, 260, 261)}
            for i, text in names.items()
        },
    }
    (model / "tokenizer_config.json").write_text(json.dumps(metadata))
    (model / "generation_config.json").write_text(json.dumps({"eos_token_id": 258}))
    return model


def run(model, output, unified):
    for _ in range(20):
        with socket.socket() as first, socket.socket() as second:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            try:
                second.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            break
    else:
        raise RuntimeError("No free pair of validation ports")
    command = [
        sys.executable,
        "-m",
        "minisgl",
        "--model",
        str(model),
        "--port",
        str(port),
        "--dtype",
        "float32",
        "--max-running-requests",
        "2",
        "--num-pages",
        "128",
        "--page-size",
        "2",
        "--max-extend-length",
        "4",
        "--max-cached-states",
        "8",
    ]
    if unified:
        command.append("--enable-unified-memory")
    log = output / ("http-unified.log" if unified else "http-static.log")
    with log.open("w") as stream:
        server = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited; see {log}")
                try:
                    if httpx.get(url + "/v1/models", timeout=1).status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                time.sleep(0.2)
            else:
                raise TimeoutError(f"Server did not become ready; see {log}")

            def complete(prompt="hello", tokens=8):
                response = httpx.post(
                    url + "/v1/chat/completions",
                    json={
                        "model": "synthetic",
                        "prompt": prompt,
                        "temperature": 0,
                        "max_tokens": tokens,
                        "ignore_eos": True,
                    },
                    timeout=60,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            baseline = complete()
            for _ in range(3):
                assert complete() == baseline, "Prefix restore or slot reuse changed the generation"
            complete("help")  # aligned radix fork and recurrent replay
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                assert all(value == baseline for value in pool.map(lambda _: complete(), range(4)))
            with httpx.stream(
                "POST",
                url + "/v1/chat/completions",
                json={
                    "model": "synthetic",
                    "prompt": "cancel",
                    "temperature": 0,
                    "max_tokens": 30,
                    "ignore_eos": True,
                    "stream": True,
                },
                timeout=60,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith("data:") and "[DONE]" not in line:
                        break
                else:
                    raise AssertionError("No stream event received")
            assert complete() == baseline, "A cancelled request contaminated a reused state slot"
            assert complete(tokens=0) == "", "Zero-token request did not complete"
            bad = httpx.post(
                url + "/v1/chat/completions",
                json={"model": "synthetic", "prompt": "hi", "max_tokens": -1},
                timeout=10,
            )
            assert bad.status_code == 400
            return {
                "status": "passed",
                "unified_memory": unified,
                "scope": "synthetic_http_scheduler_prefix_reuse_and_cancellation",
            }
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = fixture(args.output / "fixture")
    report = [run(model, args.output, unified) for unified in (False, True)]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
