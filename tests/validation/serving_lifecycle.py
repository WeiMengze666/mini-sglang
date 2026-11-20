"""HTTP lifecycle checks: attach to an owned server or launch the opt-in synthetic gate."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import secrets
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
    if args.attention_backend.split(",")[-1] == "flashinfer":
        config.hidden_size = config.routed_expert_hidden_size = 128
        config.q_lora_rank = 32
        config.kv_lora_rank, config.qk_nope_head_dim = 512, 192
        config.qk_rope_head_dim, config.v_head_dim = 64, 128
        config.linear_attn_config["head_dim"] = 128
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


def _free_port_pair():
    # Outgoing readiness/store connections can claim a probed ephemeral port
    # before a worker binds it. Keep both listeners outside that kernel range.
    ephemeral = Path("/proc/sys/net/ipv4/ip_local_port_range")
    lower, upper = map(int, ephemeral.read_text().split()) if ephemeral.exists() else (32768, 65535)
    for _ in range(100):
        port = 1024 + secrets.randbelow(65535 - 1024)
        if lower - 1 <= port <= upper:
            continue
        with socket.socket() as first, socket.socket() as second:
            try:
                first.bind(("127.0.0.1", port))
                second.bind(("127.0.0.1", port + 1))
            except OSError:
                continue
            return port
    raise RuntimeError("No free pair of validation ports outside the ephemeral range")


def _text_parts(message):
    parts = tuple(message.get(name) for name in ("content", "reasoning_content"))
    if any(part is not None and not isinstance(part, str) for part in parts):
        raise ValueError("Expected textual content/reasoning_content in chat response")
    return tuple(part or "" for part in parts)


def _prepare_prompt(prompt, tokenizer, minimum):
    if minimum < 0:
        raise ValueError("min_prefill_tokens cannot be negative")
    if minimum and tokenizer is None:
        raise ValueError("A real tokenizer is required to verify min_prefill_tokens")
    count = len(tokenizer.encode(prompt)) if tokenizer is not None else None
    if minimum:
        unit = prompt + " "
        unit_tokens = len(tokenizer.encode(unit))
        if not unit_tokens:
            raise ValueError("Prompt padding must encode at least one token")
        while count < minimum:
            copies = max(1, (minimum - count + unit_tokens - 1) // unit_tokens)
            prompt += unit * copies
            updated = len(tokenizer.encode(prompt))
            if updated <= count:
                raise ValueError("Tokenizer did not make progress while padding the prompt")
            count = updated
    return prompt, count


def exercise_http(
    url,
    served_model,
    prompt,
    *,
    concurrency=4,
    timeout=300,
    tokenizer=None,
    min_prefill_tokens=0,
    strict_protocol=False,
):
    """Return functional HTTP evidence; never own or stop the attached server.

    The standalone gate selects ``strict_protocol`` to retain its strict synthetic text
    and per-token event checks. Attach mode uses standard chat messages and
    accepts content/reasoning text differences without claiming model quality.
    Concurrency describes client requests, not an observed server batch size.
    Token floors describe logical prompts; raw usage/cache fields retain the server semantics.
    """
    if concurrency < 1 or timeout <= 0:
        raise ValueError("concurrency and timeout must be positive")
    url = url.rstrip("/")
    prompt, prompt_tokens = _prepare_prompt(prompt, tokenizer, min_prefill_tokens)
    evidence = {}

    def payload(text, tokens, *, streaming=False):
        body = {"model": served_model, "temperature": 0, "max_tokens": tokens, "ignore_eos": True}
        body.update(
            {"prompt": text}
            if strict_protocol
            else {"messages": [{"role": "user", "content": text}]}
        )
        if streaming:
            body["stream"] = True
        return body

    def complete(text=prompt, tokens=8, request_timeout=timeout):
        response = httpx.post(
            url + "/v1/chat/completions", json=payload(text, tokens), timeout=request_timeout
        )
        response.raise_for_status()
        body = response.json()
        content, reasoning = _text_parts(body["choices"][0]["message"])
        usage = body.get("usage")
        if min_prefill_tokens:
            actual = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            if (
                not isinstance(actual, int)
                or isinstance(actual, bool)
                or actual < min_prefill_tokens
            ):
                raise AssertionError(
                    "Server usage did not confirm the required prefill token count"
                )
        return {
            "status_code": response.status_code,
            "text": content if strict_protocol else reasoning + content,
            "content": content,
            "reasoning_content": reasoning,
            "usage": usage,
            "response": body,
        }

    baseline = complete(request_timeout=max(180, timeout))
    evidence["baseline"] = baseline
    warm_seconds, repeated = [], []
    for _ in range(3):
        start = time.perf_counter() if strict_protocol else None
        value = complete()
        if strict_protocol:
            assert value["text"] == baseline["text"], (
                "Prefix restore or slot reuse changed the generation"
            )
            warm_seconds.append(time.perf_counter() - start)
        repeated.append(value)
    evidence["prefix_reuse"] = repeated
    fork, _ = _prepare_prompt(prompt[:-2] + "different ending", tokenizer, min_prefill_tokens)
    evidence["prefix_fork"] = complete(fork)

    def concurrent_reuse():
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda _: complete(), range(concurrency)))
        if strict_protocol:
            assert all(value["text"] == baseline["text"] for value in results)
        return results

    evidence["concurrent"] = concurrent_reuse()
    cancel_prompt, _ = _prepare_prompt(prompt + " cancellation", tokenizer, min_prefill_tokens)
    events = []
    with httpx.stream(
        "POST",
        url + "/v1/chat/completions",
        json=payload(cancel_prompt, 30, streaming=True),
        timeout=timeout,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            event = json.loads(line[5:])
            events.append(event)
            choices = event.get("choices", [])
            delta = choices[0].get("delta", {}) if choices else {}
            content, reasoning = _text_parts(delta)
            if strict_protocol or content or reasoning:
                break
        else:
            raise AssertionError("No actual content chunk received before stream completion")
    evidence["cancellation"] = {
        "stage": "after_first_event" if strict_protocol else "after_content_chunk",
        "events": events,
        "status_code": response.status_code,
    }
    reused = complete()
    if strict_protocol:
        assert reused["text"] == baseline["text"], (
            "A cancelled request contaminated a reused state slot"
        )
    evidence["after_cancel"] = reused
    if not strict_protocol:
        evidence["concurrent_after_cancel"] = concurrent_reuse()
    else:
        assert complete(tokens=0)["text"] == "", "Zero-token request did not complete"
        for budget in (1, 2, 5):
            with httpx.stream(
                "POST",
                url + "/v1/chat/completions",
                json=payload(prompt, budget, streaming=True),
                timeout=timeout,
            ) as response:
                response.raise_for_status()
                streamed = [
                    json.loads(line[5:])
                    for line in response.iter_lines()
                    if line.startswith("data:") and "[DONE]" not in line
                ]
            assert sum(event["choices"][0]["finish_reason"] is None for event in streamed) == budget
            assert streamed[-1]["choices"][0]["finish_reason"] == "length"
        bad = httpx.post(url + "/v1/chat/completions", json=payload("hi", -1), timeout=10)
        assert bad.status_code == 400
    report = {
        "status": "passed",
        "url": url,
        "served_model": served_model,
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "client_concurrency": concurrency,
        "min_prefill_tokens": min_prefill_tokens,
        "request_format": "prompt" if strict_protocol else "messages",
        "baseline": baseline["text"],
        "evidence": evidence,
        "scope": "HTTP prefix/cancel/reuse functional checks; no quality or performance claim",
    }
    if strict_protocol:
        report.update(warm_eight_token_request_seconds=warm_seconds, stream_token_budgets=[1, 2, 5])
    return report


def run(model, output, unified, args):
    port = _free_port_pair()
    command = [
        sys.executable,
        "-m",
        "minisgl",
        "--model-path",
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
        "--max-prefill-tokens",
        str(args.max_extend_tokens),
        "--chunked-prefill-size",
        str(args.max_extend_tokens),
        "--mamba-full-memory-ratio",
        str(args.mamba_full_memory_ratio),
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
    if args.served_model_name is not None:
        command.extend(("--served-model-name", args.served_model_name))
    if args.max_seq_len is not None:
        command.extend(("--context-length", str(args.max_seq_len)))
    if args.cuda_graph_max_bs is not None:
        command.extend(("--cuda-graph-max-bs-decode", str(args.cuda_graph_max_bs)))
    if not args.enable_cache_overlap:
        command.append("--disable-overlap-schedule")
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

            report = exercise_http(
                url,
                args.served_model_name or str(model),
                args.prompt,
                concurrency=args.concurrency if args.concurrency is not None else 4,
                timeout=args.request_timeout if args.request_timeout is not None else 60,
                tokenizer=args.tokenizer_instance,
                min_prefill_tokens=args.min_prefill_tokens or 0,
                strict_protocol=True,
            )
            return {
                **report,
                "unified_memory": unified,
                "tp_size": args.tp_size,
                "dcp_size": args.dcp_size,
                "ep_size": args.ep_size,
                "packed": args.packed,
                "page_size": args.page_size,
                "cache_overlap": args.enable_cache_overlap,
                "cuda_graph_max_bs": args.cuda_graph_max_bs,
                "scope": (
                    "synthetic_http_scheduler_prefix_reuse_and_cancellation"
                    if args.model is None
                    else "released_checkpoint_http_scheduler_prefix_reuse_and_cancellation"
                ),
            }
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        help="Use an existing local checkpoint instead of generating a fixture",
    )
    parser.add_argument("--url", help="Attach without starting or stopping a server")
    parser.add_argument("--served-model-name", help="Chat API model name")
    parser.add_argument(
        "--tokenizer", type=Path, help="Explicit local tokenizer path; required in attach mode"
    )
    parser.add_argument("--concurrency", type=int, help="Client requests: attach48, standalone4")
    parser.add_argument(
        "--request-timeout", type=float, help="HTTP seconds: attach300, standalone60"
    )
    parser.add_argument(
        "--min-prefill-tokens", type=int, help="Real tokenizer minimum: attach16384, standalone0"
    )
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument(
        "--num-pages", type=int, default=2048, help="Token-page cache budget for the small fixture."
    )
    parser.add_argument(
        "--mamba-full-memory-ratio",
        type=float,
        default=0.31,
        help="State/KV memory ratio for the small fixture's cache budget.",
    )
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
    parser.add_argument("--attention-backend", default="torch")
    parser.add_argument("--cuda-graph-max-bs", type=int)
    parser.add_argument("--enable-cache-overlap", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.url:
        path = args.tokenizer
        served_model = args.served_model_name or (str(args.model) if args.model else None)
        if path is None or not path.is_dir() or served_model is None:
            raise ValueError(
                "Attach requires an explicit local --tokenizer and a served model name"
            )
        from minisgl.utils import load_tokenizer

        report = exercise_http(
            args.url,
            served_model,
            args.prompt,
            tokenizer=load_tokenizer(str(path.resolve())),
            concurrency=args.concurrency if args.concurrency is not None else 48,
            timeout=args.request_timeout if args.request_timeout is not None else 300,
            min_prefill_tokens=args.min_prefill_tokens
            if args.min_prefill_tokens is not None
            else 16384,
        )
        report["tokenizer_path"] = str(path.resolve())
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        return
    model = (
        args.model.resolve() if args.model is not None else fixture(args.output / "fixture", args)
    )
    if not model.is_dir():
        raise ValueError("The serving gate requires an existing local checkpoint")
    from minisgl.utils import load_tokenizer

    args.tokenizer_instance = load_tokenizer(str(args.tokenizer or model))
    args.prompt_tokens = len(args.tokenizer_instance.encode(args.prompt))
    if args.prompt_tokens < 2 * args.page_size:
        raise ValueError("Prompt must span at least two pages to exercise prefix reuse")
    report = [run(model, args.output, unified, args) for unified in (False, True)]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
