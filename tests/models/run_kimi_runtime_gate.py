"""Offline tiny K3 integration gate. Synthetic correctness, never model accuracy.

Run from the repo root with PYTHONPATH=python. No network or remote checkpoints
are accepted. CUDA and the real serving Engine are explicit opt-ins; see
../../docs/kimi-k3-runtime-gate.md for the evidence boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from minisgl.kvcache.hybrid_pool import HybridCache
from minisgl.models import create_model
from minisgl.models.config import ModelConfig
from minisgl.models.kimi_config import KimiK3Config, KimiTextConfig, cache_config
from minisgl.models.kimi_weight import load_kimi_weight
from minisgl.utils import cached_load_hf_config, torch_dtype
from safetensors.torch import save_file

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
PROMPT = [5, 9, 12, 4, 17, 8, 11]
MAX_LENGTH = 64
SLOTS = 3
SEED = 20261010
SCHEMA = 1


def tiny_config() -> KimiTextConfig:
    """Three hybrid layers, grouped latent MoE, full MXFP4 column blocks."""
    return KimiTextConfig(
        hidden_size=32,
        vocab_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=MAX_LENGTH,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        linear_attn_config={
            "num_heads": 4,
            "head_dim": 8,
            "short_conv_kernel_size": 3,
            "kda_layers": [1, 3],
            "full_attn_layers": [2],
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        attn_res_block_size=2,
        num_experts=4,
        num_experts_per_token=2,
        num_expert_group=2,
        topk_group=1,
        moe_intermediate_size=64,
        routed_expert_hidden_size=32,
        latent_moe_use_norm=True,
        num_shared_experts=1,
        first_k_dense_replace=1,
        dtype="float32",
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_fixtures(root: Path) -> dict:
    manifest = json.loads((root / "fixture-manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("kind") != "synthetic-k3-runtime":
        raise ValueError("Not a supported synthetic K3 fixture directory")
    for relative, digest in manifest["sha256"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Fixture manifest path escapes the fixture directory")
        if _digest(root / path) != digest:
            raise ValueError(f"Synthetic fixture checksum mismatch: {relative}")
    return manifest


def generate_fixtures(root: Path, seed: int = SEED) -> dict:
    """Persist dense/packed twins without calling the production MXFP4 decoder.

    Directly generated signed E2M1 values are the dense reference. The packed
    twin serializes their nibble codes and per-row E8M0 scales into DIFFERENT
    shards. This is exact representability, not a claim about quantization loss.
    Existing fixtures are verified/reused; arbitrary files are never overwritten.
    """
    root = Path(root)
    if (root / "fixture-manifest.json").exists():
        manifest = verify_fixtures(root)
        if manifest["seed"] != seed:
            raise ValueError("Existing fixture seed differs; choose an empty directory")
        return manifest
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Fixture directory must be empty or contain a valid fixture manifest")
    config = tiny_config()
    with torch.device("meta"), torch_dtype(torch.float32):
        model = create_model(ModelConfig.from_hf(config))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dense, packed, scales = {}, {}, {}
    signed_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    packed_names = []
    for name, parameter in sorted(model.state_dict().items()):
        value = torch.randn(parameter.shape, generator=generator) * 0.08
        if "norm.weight" in name:
            value.add_(1.0)
        raw = "language_model." + name.replace(".mlp.", ".block_sparse_moe.")
        # Preserve embeddings/head/router/norms dense, as in mixed checkpoints.
        eligible = (
            parameter.ndim == 2
            and name.endswith(".weight")
            and parameter.shape[1] % 32 == 0
            and (".self_attn." in name or ".mlp." in name)
            and ".gate.weight" not in name
            and "routed_expert_down_proj" not in name
            and "routed_expert_up_proj" not in name
        )
        if eligible:
            rows, columns = parameter.shape
            codes = torch.randint(0, 16, (rows, columns), generator=generator, dtype=torch.uint8)
            exponents = torch.randint(121, 125, (rows, columns // 32), generator=generator)
            value = signed_values[codes.long()] * (2.0 ** (exponents - 127)).repeat_interleave(
                32, -1
            )
            prefix = raw.removesuffix(".weight")
            packed[prefix + ".weight_packed"] = codes[:, ::2] | (codes[:, 1::2] << 4)
            scales[prefix + ".weight_scale"] = exponents.to(torch.uint8)
            packed_names.append(name)
        else:
            packed[raw] = value
        dense[raw] = value
    for kind, tensors in (("dense", dense), ("packed", packed)):
        directory = root / kind
        directory.mkdir()
        text = config.to_dict()
        if kind == "packed":
            text["quantization_config"] = {
                "format": "mxfp4-pack-quantized",
                "quant_method": "compressed-tensors",
                "config_groups": {
                    "fixture": {
                        "targets": [name.removesuffix(".weight") for name in packed_names],
                        "weights": {
                            "num_bits": 4,
                            "type": "float",
                            "strategy": "group",
                            "group_size": 32,
                        },
                    }
                },
            }
        wrapper = KimiK3Config(text_config=text, dtype="float32")
        (directory / "config.json").write_text(json.dumps(wrapper.to_dict(), indent=2) + "\n")
        filename = "model-00001-of-00002.safetensors" if kind == "packed" else "model.safetensors"
        save_file(tensors, str(directory / filename))
        if kind == "packed":
            scale_file = "model-00002-of-00002.safetensors"
            save_file(scales, str(directory / scale_file))
            weight_map = dict.fromkeys(tensors, filename)
            weight_map.update(dict.fromkeys(scales, scale_file))
            (directory / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": weight_map}, indent=2) + "\n"
            )
    manifest = {
        "schema": SCHEMA,
        "kind": "synthetic-k3-runtime",
        "seed": seed,
        "tensor_count": len(dense),
        "parameter_count": sum(t.numel() for t in dense.values()),
        "packed_tensor_count": len(packed_names),
        "packed_names": packed_names,
        "sha256": {str(p.relative_to(root)): _digest(p) for p in sorted(root.glob("*/*"))},
    }
    (root / "fixture-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_fixture(path: Path, dtype: torch.dtype, device: torch.device):
    if not path.is_dir():
        raise ValueError("Only local fixture directories are accepted")
    config = ModelConfig.from_hf(cached_load_hf_config(str(path)))
    with torch.device("meta"), torch_dtype(dtype):
        model = create_model(config)
    expected = model.state_dict()
    # Mirror Engine's declared-parameter dtype assignment, including FP32 state parameters.
    weights = (
        (name, value.to(expected[name].dtype) if name in expected else value)
        for name, value in load_kimi_weight(
            str(path),
            device,
            dtype=dtype,
            packed=bool(getattr(config.kimi, "quantization_config", None)),
        )
    )
    model.load_weights(weights)
    return model, config


def cache_for(config, dtype, device):
    return HybridCache(cache_config(config.kimi), MAX_LENGTH * SLOTS, 1, SLOTS, dtype, device)


def page_table(device):
    return torch.arange(MAX_LENGTH * SLOTS, device=device).reshape(SLOTS, MAX_LENGTH)


def batch_for(tokens, device, *, start=0, slot=0, uid=10, padding=0):
    n = len(tokens)
    return SimpleNamespace(
        reqs=[
            SimpleNamespace(
                uid=uid, table_idx=slot, cached_len=start, device_len=start + n, extend_len=n
            )
        ],
        input_ids=torch.tensor(tokens + [999] * padding, device=device),
        positions=torch.arange(start, start + n + padding, device=device),
        out_loc=torch.arange(
            slot * MAX_LENGTH + start, slot * MAX_LENGTH + start + n + padding, device=device
        ),
    )


def _close(actual, expected, dtype, metrics, name):
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{name}: non-finite logits/state")
    rtol, atol = (0.04, 0.03) if dtype == torch.bfloat16 else (3e-5, 3e-6)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=name)
    metrics[name] = {
        "max_abs_error": float((actual.float() - expected.float()).abs().max()),
        "rtol": rtol,
        "atol": atol,
    }


def greedy_generate(model, config, dtype, device, *, cached: bool, steps=5):
    history, generated = list(PROMPT), []
    cache, table = cache_for(config, dtype, device), page_table(device)
    for step in range(steps):
        start = len(history) - 1 if cached and step else 0
        if not cached:
            cache.reset()
        logits = model.forward_batch(batch_for(history[start:], device, start=start), cache, table)
        if not torch.isfinite(logits).all():
            raise AssertionError("Non-finite generation logits")
        token = int(logits.argmax(-1).item())
        generated.append(token)
        history.append(token)
    return generated


@torch.inference_mode()
def run_runtime_gate(root: Path, dtype=torch.float32, device=torch.device("cpu")) -> dict:
    from kimi_fault_gate import run_nonfinite_fault_gate
    from kimi_gate_provenance import provenance

    manifest = verify_fixtures(root)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable; no CPU fallback")
    reports, baselines = {}, {}
    for kind in ("dense", "packed"):
        model, config = load_fixture(root / kind, dtype, device)
        cache, table, metrics = cache_for(config, dtype, device), page_table(device), {}
        full = model.forward_batch(batch_for(PROMPT, device), cache, table, all_logits=True)
        states = (cache.recurrent._conv.clone(), cache.recurrent._recurrent.clone())
        cache.reset()
        chunks = []
        for start, end in ((0, 2), (2, 5), (5, 6), (6, 7)):
            chunks.append(
                model.forward_batch(
                    batch_for(PROMPT[start:end], device, start=start), cache, table, all_logits=True
                )
            )
        _close(torch.cat(chunks), full, dtype, metrics, "prefill_chunk_decode")
        _close(cache.recurrent._conv, states[0], dtype, metrics, "convolution_state")
        _close(cache.recurrent._recurrent, states[1], dtype, metrics, "recurrent_state")
        cache.reset()
        a, b = batch_for(PROMPT, device), batch_for([3, 15, 7], device, uid=11, slot=1)
        isolated_b = model.forward_batch(
            b, cache_for(config, dtype, device), table, all_logits=True
        )
        combined = SimpleNamespace(
            reqs=a.reqs + b.reqs,
            input_ids=torch.cat(
                (a.input_ids, b.input_ids, torch.tensor([999, 999], device=device))
            ),
            positions=torch.cat(
                (a.positions, b.positions, torch.tensor([999, 999], device=device))
            ),
            out_loc=torch.cat((a.out_loc, b.out_loc, torch.tensor([-1, -1], device=device))),
        )
        actual = model.forward_batch(combined, cache, table, all_logits=True)
        _close(actual, torch.cat((full, isolated_b)), dtype, metrics, "heterogeneous_batch_padding")
        # Abort a processed request, reuse its slot, then deliver a stale release.
        cache.release_request(0, 10)
        successor = batch_for(PROMPT[:3], device, uid=12)
        actual = model.forward_batch(successor, cache, table, all_logits=True)
        _close(actual, full[:3], dtype, metrics, "abort_slot_reuse")
        cache.release_request(0, 10)
        actual = model.forward_batch(
            batch_for(PROMPT[3:], device, uid=12, start=3), cache, table, all_logits=True
        )
        _close(actual, full[3:], dtype, metrics, "stale_abort_isolation")
        # Inject failure after real layers have mutated KDA state. Unlike an
        # invalid embedding index, this remains safe to test on CUDA.
        cache.reset()
        forward = model.lm_head.forward

        def fail(_):
            raise RuntimeError("runtime-gate injected failure after layer state updates")

        model.lm_head.forward = fail
        try:
            try:
                model.forward_batch(batch_for(PROMPT, device), cache, table)
            except RuntimeError as error:
                if "runtime-gate injected" not in str(error):
                    raise
            else:
                raise AssertionError("Failure injection was not reached")
        finally:
            model.lm_head.forward = forward
        actual = model.forward_batch(batch_for(PROMPT, device), cache, table, all_logits=True)
        _close(actual, full, dtype, metrics, "failed_forward_restart")
        cache.reset()
        actual = model.forward_batch(batch_for(PROMPT, device), cache, table, all_logits=True)
        _close(actual, full, dtype, metrics, "reset_repeat")
        generated = greedy_generate(model, config, dtype, device, cached=True)
        recomputed = greedy_generate(model, config, dtype, device, cached=False)
        repeated = greedy_generate(model, config, dtype, device, cached=True)
        if generated != recomputed or generated != repeated:
            raise AssertionError(
                f"Greedy generation differs: {generated}, {recomputed}, {repeated}"
            )
        if kind == "dense":
            baselines = {"logits": full, "generation": generated}
        else:
            _close(full, baselines["logits"], dtype, metrics, "packed_dense_logits")
            if generated != baselines["generation"]:
                raise AssertionError("Packed/dense greedy generation mismatch")
        reports[kind] = {
            "status": "passed",
            "checks": metrics,
            "nonfinite_faults": run_nonfinite_fault_gate(model, config, dtype, device),
            "greedy_token_ids": generated,
            "cache_bytes": cache.allocated_bytes,
            "weight_bytes": sum(t.numel() * t.element_size() for t in model.state_dict().values()),
        }
    return {
        "status": "passed",
        "scope": "synthetic_model_loader_cache_only",
        "provenance": provenance(device),
        "long_prefill": "not_run",
        "device": str(device),
        "dtype": str(dtype),
        "seed": manifest["seed"],
        "fixtures": reports,
        "engine": "not_run",
        "released_checkpoint_accuracy": "not_tested",
        "independent_full_hybrid_reference": "not_tested",
    }


@torch.inference_mode()
def run_engine_smoke(path: Path, dtype: torch.dtype) -> dict:
    """Real CUDA Engine + greedy Sampler; deliberately not a scheduler/HTTP test."""
    from minisgl.core import Batch, Req, SamplingParams
    from minisgl.distributed import DistributedInfo
    from minisgl.engine import Engine, EngineConfig
    from minisgl.engine.sample import BatchSamplingArgs

    if not path.is_dir():
        raise ValueError("Engine smoke requires a local fixture directory")
    if not torch.cuda.is_available():
        raise RuntimeError("Engine smoke requires CUDA; it is not a CPU test")
    engine = Engine(
        EngineConfig(
            str(path),
            DistributedInfo(0, 1),
            dtype,
            max_running_req=SLOTS,
            num_page_override=MAX_LENGTH * SLOTS,
            max_seq_len_override=MAX_LENGTH,
        )
    )
    try:
        table = page_table(engine.device)
        engine.page_table[:SLOTS, :MAX_LENGTH].copy_(table)
        expected = greedy_generate(
            engine.model,
            ModelConfig.from_hf(cached_load_hf_config(str(path))),
            dtype,
            engine.device,
            cached=False,
        )

        def generate(uid, steps=5):
            req = Req(
                torch.tensor(PROMPT, dtype=torch.int32),
                0,
                0,
                5,
                uid,
                SamplingParams(temperature=0, max_tokens=5),
                None,
            )
            generated = []
            for step in range(steps):
                start = req.cached_len
                batch = Batch([req], "decode" if step else "prefill")
                batch.padded_reqs = [req]
                batch.input_ids = req.input_ids[start:].to(engine.device)
                batch.positions = torch.arange(start, req.device_len, device=engine.device)
                batch.out_loc = table[0, start : req.device_len]
                engine.attn_backend.prepare_metadata(batch)
                result = engine.forward_batch(batch, BatchSamplingArgs(temperatures=None))
                result.copy_done_event.synchronize()
                generated.append(int(result.next_tokens_cpu.item()))
                req.append_host(result.next_tokens_cpu)
            return generated

        aborted = generate(99, steps=2)
        engine.kv_cache.release_request(0, 99)
        first = generate(100)
        engine.kv_cache.release_request(0, 100)
        second = generate(101)
        engine.kv_cache.reset()
        third = generate(102)
        if not first == second == third == expected:
            raise AssertionError(
                f"Engine/reference/reuse/reset generations differ: {first}, {second}, {third}, {expected}"
            )
        return {
            "status": "passed",
            "scope": "cuda_engine_greedy_sampler_only",
            "greedy_token_ids": first,
            "aborted_partial_token_ids": aborted,
            "scheduler_http_cancellation_races": "not_tested",
        }
    finally:
        engine.shutdown()


def run_official_checks(hf_source: Path, fla_source: Path) -> dict:
    """Reuse existing independently pinned tests; never fetch or alter sources."""
    for path in (hf_source, fla_source):
        if not path.is_file():
            raise FileNotFoundError(path)
    environment = dict(
        os.environ,
        KIMI_OFFICIAL_REFERENCE_PATH=str(hf_source.resolve()),
        KIMI_FLA_REFERENCE_PATH=str(fla_source.resolve()),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    test = Path(__file__).with_name("test_kimi_official_reference.py")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", "-rsx", str(test)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode:
        raise RuntimeError(f"Official-source checks failed:\n{result.stdout}\n{result.stderr}")
    return {
        "status": "passed_with_documented_expected_disagreements",
        "scope": "official_operator_and_mla_stack_only",
        "source_sha256": {"hf": _digest(hf_source), "fla": _digest(fla_source)},
        "output": result.stdout,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="Persist/reuse verified fixtures; default: temporary directory",
    )
    parser.add_argument("--dtype", choices=DTYPES, default="float32")
    parser.add_argument("--device", choices=["cpu", "cuda", "cuda:0"], default="cpu")
    parser.add_argument(
        "--long-prefill",
        action="store_true",
        help="Also compare two >2048-token sequences; synthetic eager kernels only",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--engine-smoke",
        action="store_true",
        help="Also run actual CUDA Engine in fresh subprocesses",
    )
    parser.add_argument("--engine-checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--official-source", type=Path)
    parser.add_argument("--fla-source", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if bool(args.official_source) != bool(args.fla_source):
        parser.error("Provide both --official-source and --fla-source")
    if args.generate_only and (args.engine_smoke or args.official_source or args.long_prefill):
        parser.error("--generate-only cannot be combined with execution checks")
    if args.engine_smoke and args.device == "cpu":
        parser.error("--engine-smoke requires --device cuda")
    if args.generate_only and args.fixture_dir is None:
        parser.error("--generate-only requires --fixture-dir")
    torch.set_num_threads(1)
    if args.engine_checkpoint is not None:
        report = run_engine_smoke(args.engine_checkpoint, DTYPES[args.dtype])
    else:
        temporary = (
            tempfile.TemporaryDirectory(prefix="kimi-runtime-gate-")
            if args.fixture_dir is None
            else None
        )
        root = args.fixture_dir or Path(temporary.name)
        try:
            manifest = generate_fixtures(root, args.seed)
            report = (
                {"status": "generated_only", "manifest": manifest}
                if args.generate_only
                else run_runtime_gate(root, DTYPES[args.dtype], torch.device(args.device))
            )
            if args.long_prefill:
                from kimi_long_prefill_gate import run_long_prefill_gate

                report["long_prefill"] = run_long_prefill_gate(
                    root, DTYPES[args.dtype], torch.device(args.device)
                )
            if args.engine_smoke:
                engine_reports = {}
                for kind in ("dense", "packed"):
                    output = root / f"engine-{kind}.json"
                    subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "--engine-checkpoint",
                            str((root / kind).resolve()),
                            "--dtype",
                            args.dtype,
                            "--report",
                            str(output.resolve()),
                        ],
                        check=True,
                        timeout=180,
                    )
                    engine_reports[kind] = json.loads(output.read_text())
                report["engine"] = engine_reports
            if args.official_source:
                report["official_reference"] = run_official_checks(
                    args.official_source, args.fla_source
                )
                from kimi_gate_provenance import provenance

                report["provenance"] = provenance(torch.device(args.device), references_run=True)
        finally:
            if temporary is not None:
                temporary.cleanup()
    if "provenance" not in report:
        from kimi_gate_provenance import provenance

        report["provenance"] = provenance(
            torch.device("cuda:0" if args.engine_checkpoint else args.device),
            executed=not args.generate_only,
        )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(rendered)
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
