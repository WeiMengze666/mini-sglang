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


def fixture(root, args):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
    import run_kimi_runtime_gate as runtime

    config = runtime.tiny_config()
    config.vocab_size = 512
    if args.packed:
        config.linear_attn_config["head_dim"] = 32
        config.v_head_dim = 32
        config.intermediate_size = config.moe_intermediate_size = 32 * args.tp_size
    runtime.tiny_config = lambda: config
    runtime.generate_fixtures(root)
    model = root / ("packed" if args.packed else "dense")
    (model / "tiktoken.model").write_bytes(
        b"".join(base64.b64encode(bytes([i])) + f" {i}\n".encode() for i in range(256))
    )
    names = {i: f"<extra_{i}>" for i in range(256, 512)}
    names.update(
        {
            256: "[BOS]",
            257: "[EOS]",
            258: "<|end_of_msg|>",
            259: "<|open|>",
            260: "<|close|>",
            261: "<|sep|>",
            510: "[UNK]",
            511: "[PAD]",
        }
    )
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


def run(model, output, unified, args):
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
        args.dtype,
        "--max-running-requests",
        "2",
        "--num-pages",
        str(args.num_pages),
        "--page-size",
        str(args.page_size),
        "--max-extend-length",
        str(args.max_extend_tokens),
        "--max-cached-states",
        "8",
        "--tp-size",
        str(args.tp_size),
        "--dcp-size",
        str(args.dcp_size),
        "--ep-size",
        str(args.ep_size),
        "--mxfp4-backend",
        args.mxfp4_backend,
        "--attention-backend",
        args.attention_backend,
    ]
    if args.max_seq_len is not None:
        command.extend(("--max-seq-len-override", str(args.max_seq_len)))
    if unified:
        command.append("--enable-unified-memory")
    log = output / ("http-unified.log" if unified else "http-static.log")
    with log.open("w") as stream:
        server = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + args.readiness_timeout
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

            def complete(prompt=args.prompt, tokens=8, timeout=60):
                response = httpx.post(
                    url + "/v1/chat/completions",
                    json={
                        "model": str(model),
                        "prompt": prompt,
                        "temperature": 0,
                        "max_tokens": tokens,
                        "ignore_eos": True,
                    },
                    timeout=timeout,
                )
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

            baseline = complete(timeout=180)  # first-use CUDA compilation is outside latency gates
            for _ in range(3):
                assert complete() == baseline, "Prefix restore or slot reuse changed the generation"
            complete(args.prompt[:-2] + "different ending")  # radix fork and recurrent replay
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                assert all(value == baseline for value in pool.map(lambda _: complete(), range(4)))
            with httpx.stream(
                "POST",
                url + "/v1/chat/completions",
                json={
                    "model": str(model),
                    "prompt": args.prompt + " cancellation",
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
                json={"model": str(model), "prompt": "hi", "max_tokens": -1},
                timeout=10,
            )
            assert bad.status_code == 400
            return {
                "status": "passed",
                "unified_memory": unified,
                "tp_size": args.tp_size,
                "dcp_size": args.dcp_size,
                "ep_size": args.ep_size,
                "packed": args.packed,
                "prompt_tokens": args.prompt_tokens,
                "page_size": args.page_size,
                "scope": "synthetic_http_scheduler_prefix_reuse_and_cancellation"
                if args.model is None
                else "released_checkpoint_http_scheduler_prefix_reuse_and_cancellation",
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
    parser.add_argument(
        "--model",
        type=Path,
        help="Use an existing local checkpoint instead of generating a fixture",
    )
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--num-pages", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=2)
    parser.add_argument("--max-extend-tokens", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--readiness-timeout", type=float, default=180)
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--dcp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--mxfp4-backend", choices=["torch", "triton"], default="torch")
    parser.add_argument("--attention-backend", default="recurrent")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    model = (
        args.model.resolve() if args.model is not None else fixture(args.output / "fixture", args)
    )
    if not model.is_dir():
        raise ValueError("The serving gate requires an existing local checkpoint")
    from minisgl.utils import load_tokenizer

    args.prompt_tokens = len(load_tokenizer(str(model)).encode(args.prompt))
    if args.prompt_tokens < 2 * args.page_size:
        raise ValueError("Prompt must span at least two pages to exercise prefix reuse")
    report = [run(model, args.output, unified, args) for unified in (False, True)]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
