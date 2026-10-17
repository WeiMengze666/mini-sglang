"""Offline long-context checks for tiny synthetic K3 dense/packed fixtures.

This exercises the existing sequential eager model/cache seam. It does not
measure fused GPU kernels, serving concurrency, or released-checkpoint accuracy.
Only the loaded runtime configuration is extended; fixture files stay unchanged.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import torch
from minisgl.kvcache.kimi_pool import KimiHybridCache
from run_kimi_runtime_gate import _close, load_fixture, verify_fixtures

DECODE_STEPS = 3


def _cache_and_table(config, dtype, device, capacity, slots):
    cache = KimiHybridCache(config.kimi, capacity * slots, 1, slots, dtype, device)
    table = torch.arange(capacity * slots, device=device).reshape(slots, capacity)
    return cache, table


def _batch(pieces, starts, slots, uids, table, device):
    """Concatenate real rows; heterogeneous requests retain independent positions."""
    return SimpleNamespace(
        reqs=[
            SimpleNamespace(
                uid=uid,
                table_idx=slot,
                cached_len=start,
                device_len=start + len(piece),
                extend_len=len(piece),
            )
            for piece, start, slot, uid in zip(pieces, starts, slots, uids, strict=True)
        ],
        input_ids=torch.tensor([token for piece in pieces for token in piece], device=device),
        positions=torch.cat(
            [
                torch.arange(start, start + len(piece), device=device)
                for piece, start in zip(pieces, starts, strict=True)
            ]
        ),
        out_loc=torch.cat(
            [
                table[slot, start : start + len(piece)]
                for piece, start, slot in zip(pieces, starts, slots, strict=True)
            ]
        ),
    )


def _live_state(cache, table, slot, length):
    # MLA allocation uses torch.empty, so never inspect unwritten spare capacity.
    return {
        "convolution": cache._conv[:, slot].clone(),
        "recurrent": cache._recurrent[:, slot].clone(),
        "mla": cache._mla.index_select(1, table[slot, :length]).clone(),
    }


def _compare_state(actual, expected, dtype, metrics, prefix):
    for name in actual:
        _close(actual[name], expected[name], dtype, metrics, f"{prefix}_{name}")


def _finite_summary(tensor, name):
    if not tensor.numel() or not torch.isfinite(tensor).all():
        raise AssertionError(f"{name}: empty or non-finite live logits/state")
    return {
        "all_finite": True,
        "elements": tensor.numel(),
        "max_abs_value": float(tensor.float().abs().max()),
    }


@torch.inference_mode()
def run_long_prefill_gate(
    root: Path,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    length: int = 2051,
) -> dict:
    """Compare batched/isolated/chunked prefill and three cached decode steps.

    Default request lengths are 2051 and 2054, both above 2048. A shorter
    ``length`` is supported for fast unit tests and explicitly labeled in the
    report. Callers control torch's CPU thread count, as in the runtime gate CLI.
    """
    if not isinstance(length, int) or isinstance(length, bool) or length < 2:
        raise ValueError("Long-prefill length must be an integer >= 2")
    root, device = Path(root), torch.device(device)
    if dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("Long-prefill gate supports float32 and bfloat16 only")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable; no CPU fallback")
    manifest = verify_fixtures(root)
    lengths = [length, length + 3]
    capacity = max(lengths) + DECODE_STEPS
    uids, reports, dense_reference = [7001, 7002], {}, None
    for kind in ("dense", "packed"):
        model, config = load_fixture(root / kind, dtype, device)
        checkpoint_limit = config.kimi.max_position_embeddings
        # load_fixture creates a fresh KimiTextConfig shared with the layers.
        # No weights or on-disk config change. NoPE has no positional table;
        # unscaled RoPE, if supplied, computes positions directly in the operator.
        config.kimi.max_position_embeddings = max(checkpoint_limit, capacity)
        config = replace(
            config,
            rotary_config=replace(
                config.rotary_config, max_position=config.kimi.max_position_embeddings
            ),
        )
        generator = torch.Generator(device="cpu").manual_seed(manifest["seed"] + 17)
        sequences = [
            torch.randint(config.vocab_size, (size,), generator=generator).tolist()
            for size in lengths
        ]
        if sequences[0] == sequences[1][:length]:
            raise AssertionError("Long-prefill requests must have distinct token histories")
        batched_cache, table = _cache_and_table(config, dtype, device, capacity, 2)
        chunked_cache, chunked_table = _cache_and_table(config, dtype, device, capacity, 2)
        isolated = [_cache_and_table(config, dtype, device, capacity, 1) for _ in lengths]
        metrics = {}
        batched_logits = model.forward_batch(
            _batch(sequences, [0, 0], [0, 1], uids, table, device),
            batched_cache,
            table,
            all_logits=True,
        )
        batched_parts = list(batched_logits.split(lengths))
        isolated_parts = []
        for i, (cache, single_table) in enumerate(isolated):
            isolated_parts.append(
                model.forward_batch(
                    _batch([sequences[i]], [0], [0], [uids[i]], single_table, device),
                    cache,
                    single_table,
                    all_logits=True,
                )
            )
        _close(
            batched_logits,
            torch.cat(isolated_parts),
            dtype,
            metrics,
            "prefill_batched_vs_isolated",
        )

        # Different chunk sizes exercise independently advancing cached lengths.
        # The final chunks cross 2048 at the default length; small tests still
        # execute more than one chunk for both requests.
        chunk_sizes = [min(257, max(1, length // 3)), min(509, max(1, length // 2))]
        starts, chunks, chunk_counts = [0, 0], [[], []], [0, 0]
        while any(start < size for start, size in zip(starts, lengths, strict=True)):
            active = [i for i in range(2) if starts[i] < lengths[i]]
            ends = [min(starts[i] + chunk_sizes[i], lengths[i]) for i in active]
            pieces = [sequences[i][starts[i] : end] for i, end in zip(active, ends, strict=True)]
            output = model.forward_batch(
                _batch(
                    pieces,
                    [starts[i] for i in active],
                    active,
                    [uids[i] for i in active],
                    chunked_table,
                    device,
                ),
                chunked_cache,
                chunked_table,
                all_logits=True,
            )
            for i, end, logits in zip(
                active, ends, output.split([len(p) for p in pieces]), strict=True
            ):
                chunks[i].append(logits)
                starts[i] = end
                chunk_counts[i] += 1
        chunked_parts = [torch.cat(parts) for parts in chunks]
        _close(
            torch.cat(chunked_parts), batched_logits, dtype, metrics, "prefill_chunked_vs_batched"
        )

        def compare_states(stage, current_lengths):
            for i, size in enumerate(current_lengths):
                actual = _live_state(batched_cache, table, i, size)
                single_cache, single_table = isolated[i]
                _compare_state(
                    actual,
                    _live_state(single_cache, single_table, 0, size),
                    dtype,
                    metrics,
                    f"{stage}_request_{i}_batched_vs_isolated",
                )
                _compare_state(
                    _live_state(chunked_cache, chunked_table, i, size),
                    actual,
                    dtype,
                    metrics,
                    f"{stage}_request_{i}_chunked_vs_batched",
                )

        compare_states("prefill", lengths)
        last_logits = torch.stack([part[-1] for part in batched_parts])
        decode_inputs, decode_outputs = [], []
        for step in range(DECODE_STEPS):
            # Shared teacher-forced inputs isolate cache/path agreement from
            # any near-tied argmax divergence. These are not accuracy metrics.
            tokens = last_logits.argmax(-1).tolist()
            decode_inputs.append(tokens)
            pieces = [[token] for token in tokens]
            starts = [size + step for size in lengths]
            last_logits = model.forward_batch(
                _batch(pieces, starts, [0, 1], uids, table, device), batched_cache, table
            )
            chunked_logits = model.forward_batch(
                _batch(pieces, starts, [0, 1], uids, chunked_table, device),
                chunked_cache,
                chunked_table,
            )
            single_logits = []
            for i, (cache, single_table) in enumerate(isolated):
                single_logits.append(
                    model.forward_batch(
                        _batch([pieces[i]], [starts[i]], [0], [uids[i]], single_table, device),
                        cache,
                        single_table,
                    )
                )
            _close(
                last_logits,
                torch.cat(single_logits),
                dtype,
                metrics,
                f"decode_{step}_batched_vs_isolated",
            )
            _close(chunked_logits, last_logits, dtype, metrics, f"decode_{step}_chunked_vs_batched")
            compare_states(f"decode_{step}", [start + 1 for start in starts])
            decode_outputs.append(last_logits.clone())

        final_lengths = [size + DECODE_STEPS for size in lengths]
        final_states = [
            _live_state(batched_cache, table, i, size) for i, size in enumerate(final_lengths)
        ]
        if dense_reference is None:
            dense_reference = (batched_logits.clone(), torch.stack(decode_outputs), final_states)
        else:
            _close(batched_logits, dense_reference[0], dtype, metrics, "packed_dense_prefill")
            _close(
                torch.stack(decode_outputs),
                dense_reference[1],
                dtype,
                metrics,
                "packed_dense_decode",
            )
            for i, state in enumerate(final_states):
                _compare_state(
                    state, dense_reference[2][i], dtype, metrics, f"packed_dense_request_{i}"
                )
        reports[kind] = {
            "status": "passed",
            "checks": metrics,
            "prefill_logits": _finite_summary(batched_logits, "prefill logits"),
            "decode_logits": _finite_summary(torch.stack(decode_outputs), "decode logits"),
            "live_final_state": [
                {name: _finite_summary(value, name) for name, value in state.items()}
                for state in final_states
            ],
            "chunk_sizes": chunk_sizes,
            "chunk_counts": chunk_counts,
            "decode_input_token_ids": decode_inputs,
            "batched_cache_bytes": batched_cache.allocated_bytes,
            "checkpoint_max_position_embeddings": checkpoint_limit,
            "runtime_max_position_embeddings": config.kimi.max_position_embeddings,
            "mla_position_encoding": "nope" if config.kimi.mla_use_nope else "unscaled_rope",
        }
    verify_fixtures(root)
    return {
        "status": "passed",
        "scope": "synthetic_sequential_eager_long_prefill_cache_only",
        "device": str(device),
        "dtype": str(dtype),
        "seed": manifest["seed"],
        "sequence_lengths": lengths,
        "both_sequences_above_2048": min(lengths) > 2048,
        "long_acceptance": min(lengths) > 2048,
        "decode_steps": DECODE_STEPS,
        "decode_policy": "shared_teacher_forced_batched_argmax",
        "fixtures": reports,
        "checkpoint_files_unchanged": True,
        "execution": "sequential_eager_request_loops",
        "engine": "not_run",
        "concurrent_serving": "not_tested",
        "optimized_gpu_kernels": "not_tested",
        "released_checkpoint_accuracy": "not_tested",
    }
