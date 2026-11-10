"""Native indexed FlashInfer adapter parity, not independent arithmetic acceptance.

Run only inside an authorized gpuq job. Both routes call the same mature native kernel. All-T1 uses the independent
official TritonKDAKernel and mature chunk entry because the pinned FI facade
reshapes bias to an unsupported grouped-decode ABI; other cases use the pinned
FlashInfer class from identical raw projections, original BF16/FP32 indexed SSM
pools, BF16 convolution state, and post-convolution inputs. This probes adapter/storage/lifecycle correctness;
it does not qualify model quality, performance, decode or prefill CUDA Graphs.
"""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

SDK_COMMIT = "baa9befa381511cd4d56f83980951e5a45fffc6d"
LAYER_IDS, TARGET_LAYER = (2, 7, 11), 7
STATE_DTYPES = (torch.bfloat16, torch.float32)


def describe(value):
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "pointer": value.data_ptr(),
        "storage_offset": value.storage_offset(),
        "device": str(value.device),
    }


def assert_exact(actual, expected, label):
    assert actual.shape == expected.shape, f"{label}: shape mismatch"
    assert actual.dtype == expected.dtype, f"{label}: dtype mismatch"
    assert torch.equal(actual, expected), f"{label}: tensor mismatch"


def capture_storage(pool):
    return [pool.arena.buffer] if pool.arena is not None else [pool._conv, pool._recurrent]


def checkpoint_values(pool, checkpoint):
    saved = pool._checkpoints[checkpoint.slot]
    if saved is None:
        saved = pool._views(("checkpoint", checkpoint.slot))
    return tuple(part.clone() for part in saved)


class NativeTrace:
    """Observe actual FI facade or mature chunk entry before state mutation."""

    backend = "flashinfer"

    def __init__(self, kernel_class, *, state_dtype, expected_stream=None):
        self.kernel_class = kernel_class
        self.state_dtype = state_dtype
        self.expected_stream = expected_stream
        self.calls = []
        self.mode = "direct_official"

    def record(self, q, k, v, g, beta, kwargs, invoke):
        import tvm_ffi

        state = kwargs["ssm_states"]
        torch_stream = int(torch.cuda.current_stream(q.device).cuda_stream)
        ffi_stream = int(tvm_ffi.get_raw_stream(tvm_ffi.device(str(q.device))))
        if self.expected_stream is not None:
            assert torch_stream == self.expected_stream, "Native call left explicit side stream"
            assert ffi_stream == self.expected_stream, "Native call lacks its own FFI context"
        row = {
            "route": self.mode,
            "native_backend": self.backend,
            "torch_stream": torch_stream,
            "ffi_stream": ffi_stream,
            "tensors": {
                name: describe(value)
                for name, value in zip(("q", "k", "v", "g", "beta"), (q, k, v, g, beta))
            },
            "state": describe(state),
            "cache_indices": describe(kwargs["cache_indices"]),
            "query_start_loc": describe(kwargs["query_start_loc"]),
            "A_log": describe(kwargs["A_log"]),
            "dt_bias": describe(kwargs["dt_bias"]),
            "indices": kwargs["cache_indices"].tolist(),
            "cu_seqlens": kwargs["query_start_loc"].tolist(),
            "lower_bound": kwargs["lower_bound"],
            "return_intermediate_states": kwargs.get("return_intermediate_states", False),
            "chunk_indices": kwargs["chunk_indices"].tolist()
            if "chunk_indices" in kwargs
            else None,
            "values": [
                value.clone()
                for value in (q, k, v, g, beta, state, kwargs["A_log"], kwargs["dt_bias"])
            ],
        }
        self.calls.append(row)
        with torch.profiler.record_function("native_kda_" + self.mode):
            return invoke()

    def __enter__(self):
        original = self.kernel_class.extend

        def observe(instance, q, k, v, g, beta, **kwargs):
            return self.record(
                q, k, v, g, beta, kwargs, lambda: original(instance, q, k, v, g, beta, **kwargs)
            )

        self.guard = patch.object(self.kernel_class, "extend", observe)
        self.guard.start()
        return self

    def __exit__(self, *args):
        self.guard.stop()

    def compare(self):
        assert len(self.calls) == 2, "Expected one observed native entry on each route"
        direct, mini = self.calls
        assert direct["route"] == "direct_official" and mini["route"] == "mini_adapter"
        for key in (
            "native_backend",
            "state",
            "indices",
            "cu_seqlens",
            "lower_bound",
            "return_intermediate_states",
            "torch_stream",
            "ffi_stream",
            "chunk_indices",
        ):
            assert direct[key] == mini[key], f"Native ABI differs: {key}"
        for row in self.calls:
            assert row["state"]["dtype"] == str(self.state_dtype)
            assert row["cache_indices"]["dtype"] == "torch.int32"
            assert row["query_start_loc"]["dtype"] == "torch.int32"
            assert row["state"]["pointer"] % 16 == 0
            assert row["state"]["stride"][-3:] == [128 * 128, 128, 1]
            if self.backend == "sgl_chunk":
                assert len(row["dt_bias"]["shape"]) == 1 and row["dt_bias"]["stride"] == [1]
        for name, observed, expected in zip(
            ("q", "k", "v", "raw_gate", "raw_beta", "original_recurrent_state", "A_log", "dt_bias"),
            mini["values"],
            direct["values"],
        ):
            assert_exact(observed, expected, name)
        return [{key: value for key, value in row.items() if key != "values"} for row in self.calls]


class ChunkTrace(NativeTrace):
    """Observe the real chunk callable used by the independent SDK class and Mini."""

    backend = "sgl_chunk"

    def __enter__(self):
        import importlib
        from contextlib import ExitStack

        from minisgl.attention.kda import sgl

        native = importlib.import_module("sglang.srt.layers.attention.linear.kernels.kda_triton")
        original = native.chunk_kda
        chunk, decode, stream = sgl._kernels()
        assert chunk is original, "Reference and adapter must use the same mature chunk entry"

        def observe(**kw):
            assert kw["beta_is_raw"] and kw["use_qk_l2norm_in_kernel"]
            canonical = {
                "ssm_states": kw["initial_state"],
                "cache_indices": kw["initial_state_indices"],
                "query_start_loc": kw["cu_seqlens"],
                "A_log": kw["A_log"],
                "dt_bias": kw["dt_bias"],
                "lower_bound": kw["lower_bound"],
                "return_intermediate_states": kw.get("output_intermediate_states", False),
                "chunk_indices": kw["chunk_indices"],
            }
            return self.record(
                kw["q"], kw["k"], kw["v"], kw["g"], kw["beta"], canonical, lambda: original(**kw)
            )

        self.guard = ExitStack()
        self.guard.enter_context(patch.object(native, "chunk_kda", observe))
        self.guard.enter_context(patch.object(sgl, "_kernels", lambda: (observe, decode, stream)))
        return self

    def __exit__(self, *args):
        self.guard.close()


def initialize(layer, pool, tables, warm, seed):
    torch.manual_seed(seed)
    for value in layer.state_dict().values():
        value.normal_(std=0.02)
    layer.o_norm.weight.fill_(1)
    # A separate immutable checkpoint occupies a physical row in unified storage.
    pool.begin_request(0, 900, 0)
    for layer_id in LAYER_IDS:
        state = pool.get_kda_state(layer_id, 0)
        state.conv.normal_(std=0.1)
        state.recurrent.normal_(std=0.02)
    pool.finish_request(0, 900, 13)
    checkpoint = pool.snapshot(0, 900, expected_len=13)
    pool.release_request(0, 900)
    for seq, table in enumerate(tables):
        pool.begin_request(table, 100 + table, 0)
        for layer_id in LAYER_IDS:
            if warm[seq] or layer_id != TARGET_LAYER:
                state = pool.get_kda_state(layer_id, table)
                state.conv.normal_(std=0.1)
                state.recurrent.normal_(std=0.02)
        if warm[seq]:
            pool.finish_request(table, 100 + table, 13)
            pool.begin_request(table, 100 + table, 13)
    return checkpoint


def metadata(lengths, warm, indices):
    from minisgl.attention.kda.backend import KDAMetadata

    offsets = torch.tensor([0, *lengths], dtype=torch.int32, device="cuda").cumsum(
        0, dtype=torch.int32
    )
    return KDAMetadata(
        offsets, state_indices=indices, lengths=tuple(lengths), has_initial_states=tuple(warm)
    )


def launch_metadata(path):
    events = json.loads(path.read_text()).get("traceEvents", [])
    return [
        {"name": event.get("name"), "duration_us": event.get("dur"), "args": event.get("args", {})}
        for event in events
        if event.get("cat") == "kernel"
    ]


def run_forward(layer, pool, tables, lengths, warm, output, *, state_dtype, profile=False):
    from minisgl.attention.kda import flashinfer, prefill

    all_t1 = all(length == 1 for length in lengths)
    if all_t1:
        from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel

        kernel_class, trace_class = TritonKDAKernel, ChunkTrace
    else:
        from sglang.srt.layers.attention.linear.kernels.kda_flashinfer_prefill import (
            FlashInferKDAPrefillKernel,
        )

        kernel_class, trace_class = FlashInferKDAPrefillKernel, NativeTrace

    assert sum(lengths) > 1, "Single total-token prefill belongs to phase-correct SGL fallback"
    indices = pool.batch_indices(tables).to(torch.int32)
    native_conv, native_state = pool.indexed_kda_state(TARGET_LAYER)
    assert native_state.dtype == state_dtype
    assert native_conv.dtype == torch.bfloat16
    assert native_state.stride()[-3:] == (128 * 128, 128, 1)
    active = set(indices.tolist())
    inactive = torch.tensor(
        [row for row in range(len(native_state)) if row not in active],
        device="cuda",
        dtype=torch.long,
    )
    untouched = tuple(value.index_select(0, inactive) for value in (native_conv, native_state))
    other_layers = {
        layer_id: tuple(value.clone() for value in pool.indexed_kda_state(layer_id))
        for layer_id in LAYER_IDS
        if layer_id != TARGET_LAYER
    }
    storage = capture_storage(pool)
    initial_storage = [value.clone() for value in storage]
    x = torch.randn(sum(lengths), layer._config.hidden_size, device="cuda", dtype=torch.bfloat16)
    x.mul_(0.1)
    direct_kernel = kernel_class()
    profiler = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        )
        if profile
        else nullcontext()
    )
    from tvm_ffi import use_torch_stream

    parent_stream = torch.cuda.current_stream(x.device)
    side_stream = torch.cuda.Stream(device=x.device)
    assert side_stream.cuda_stream != torch.cuda.default_stream(x.device).cuda_stream
    side_stream.wait_stream(parent_stream)
    tracer = trace_class(
        kernel_class,
        state_dtype=state_dtype,
        expected_stream=int(side_stream.cuda_stream),
    )
    try:
        # Only Torch is scoped globally. The direct official reference supplies
        # its own FFI wrapper; the Mini adapter must establish its own wrapper.
        with torch.inference_mode(), profiler as prof, torch.cuda.stream(side_stream), tracer:
            data = prefill.prepare(
                layer, x, pool, TARGET_LAYER, indices, metadata(lengths, warm, indices)
            )
            assert data is not None
            q, k, v, gate, beta, original_state, plan, physical_indices = data
            assert original_state.data_ptr() == native_state.data_ptr()
            # Independent SDK reference, never Mini's indexed adapter. The
            # pinned FI facade is not called for its unsupported all-T1 ABI.
            reference_options = {}
            if all_t1:
                reference_options = {
                    "beta_is_raw": True,
                    "chunk_indices": torch.tensor(
                        [(seq, 0) for seq in range(len(lengths))],
                        device=x.device,
                        dtype=torch.int32,
                    ),
                }
            with use_torch_stream():
                direct = direct_kernel.extend(
                    q,
                    k,
                    v,
                    gate,
                    beta,
                    ssm_states=original_state,
                    cache_indices=physical_indices,
                    query_start_loc=plan["cu_seqlens"],
                    A_log=layer.A_log,
                    dt_bias=layer.dt_bias.reshape(-1),
                    lower_bound=layer._config.gate_lower_bound,
                    return_intermediate_states=False,
                    **reference_options,
                )[0].clone()
            expected_storage = [value.clone() for value in storage]
            for value, initial in zip(storage, initial_storage):
                value.copy_(initial)
            tracer.mode = "mini_adapter"
            actual = flashinfer.indexed_prefill(
                layer, x, pool, TARGET_LAYER, indices, metadata(lengths, warm, indices)
            )
            assert actual is not None, "Supported native prefill unexpectedly declined"
            assert_exact(actual, direct, "native output")
            assert torch.isfinite(actual).all(), "Non-finite native output"
            for index, (value, expected) in enumerate(zip(storage, expected_storage)):
                assert_exact(value, expected, f"full pool storage {index}")
            assert torch.isfinite(native_state.index_select(0, indices.long())).all()
            for label, value, expected in zip(
                ("conv", "SSM"), (native_conv, native_state), untouched
            ):
                assert_exact(value.index_select(0, inactive), expected, "inactive " + label)
            for layer_id, saved in other_layers.items():
                for value, expected in zip(pool.indexed_kda_state(layer_id), saved):
                    assert_exact(value, expected, f"inactive layer {layer_id}")
            trace = tracer.compare()
        parent_stream.wait_stream(side_stream)
        torch.cuda.synchronize()
        launches = []
        if profile:
            path = output.with_suffix(".trace.json")
            prof.export_chrome_trace(str(path))
            launches = launch_metadata(path)
        return {
            "lengths": list(lengths),
            "native_backend": tracer.backend,
            "has_initial_states": list(warm),
            "indices": indices.tolist(),
            "native_pool": describe(native_state),
            "side_stream": int(side_stream.cuda_stream),
            "default_stream": int(torch.cuda.default_stream(x.device).cuda_stream),
            "ffi_stream_context_checked": True,
            "same_post_conv_inputs": True,
            "original_pool_abi": True,
            "output_and_full_storage_exact": True,
            "all_inactive_rows_unchanged": True,
            "other_layers_unchanged": True,
            "calls": trace,
            "launches": launches,
        }
    except BaseException as error:
        output.with_suffix(".error.json").write_text(
            json.dumps(
                {
                    "error": repr(error),
                    "lengths": list(lengths),
                    "warm": list(warm),
                    "calls": [
                        {key: value for key, value in row.items() if key != "values"}
                        for row in tracer.calls
                    ],
                },
                indent=2,
            )
            + "\n"
        )
        raise


def run_case(lengths, mode, unified, output, *, state_dtype, profile=False):
    from minisgl.attention.config import KDAConfig
    from minisgl.kvcache.recurrent_pool import RecurrentStatePool
    from minisgl.kvcache.unified import UnifiedMemory
    from minisgl.layers.kda import KDALayer

    heads, dim, hidden = 12, 128, 512
    batch, capacity = len(lengths), len(lengths) * 2 + 3
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            layer = KDALayer(KDAConfig(hidden, heads, dim, 4, gate_lower_bound=-5.0))
    finally:
        torch.set_default_dtype(old_dtype)
    # Force padding beyond the complete payload to exercise physical slot stride.
    payload = len(LAYER_IDS) * (
        heads * dim * dim * state_dtype.itemsize + 3 * heads * dim * 3 * torch.bfloat16.itemsize
    )
    arena = UnifiedMemory(capacity + 3, payload + 256, payload + 256, "cuda") if unified else None
    if arena is not None:
        arena.buffer.zero_()
    pool = RecurrentStatePool(
        LAYER_IDS,
        capacity,
        heads,
        dim,
        4,
        torch.bfloat16,
        "cuda",
        max_cached_states=2,
        state_dtype=state_dtype,
        arena=arena,
    )
    tables = list(reversed(range(1, 2 * batch, 2)))
    warm = tuple(mode == "continued" or (mode == "mixed" and seq % 2 == 0) for seq in range(batch))
    checkpoint = initialize(layer, pool, tables, warm, 20261110 + sum(lengths))
    checkpoint_before = checkpoint_values(pool, checkpoint)
    report = {
        "lengths": list(lengths),
        "mode": mode,
        "unified_padded": unified,
        "state_dtype": str(state_dtype),
        "checkpoint_length": checkpoint.length,
        "target_layer": TARGET_LAYER,
        "independent_arithmetic_qualified": False,
        "model_qualified": False,
        "performance_qualified": False,
        "prefill_graph_qualified": False,
    }
    try:
        report["initial"] = run_forward(
            layer, pool, tables, lengths, warm, output, state_dtype=state_dtype, profile=profile
        )
        for value, expected in zip(checkpoint_values(pool, checkpoint), checkpoint_before):
            assert_exact(value, expected, "immutable checkpoint after prefill")
        for seq, table in enumerate(tables):
            pool.finish_request(table, 100 + table, (13 if warm[seq] else 0) + lengths[seq])
        # A short continuation uses the newly committed state on every row.
        continued_lengths = tuple(1 if seq % 2 == 0 else 3 for seq in range(batch))
        if sum(continued_lengths) == 1:
            continued_lengths = (3,)
        for seq, table in enumerate(tables):
            pool.begin_request(table, 100 + table, (13 if warm[seq] else 0) + lengths[seq])
        report["continuation"] = run_forward(
            layer, pool, tables, continued_lengths, (True,) * batch, output, state_dtype=state_dtype
        )
        for seq, table in enumerate(tables):
            frontier = (13 if warm[seq] else 0) + lengths[seq] + continued_lengths[seq]
            pool.finish_request(table, 100 + table, frontier)
            pool.release_request(table, 100 + table)
        # Restore the independent checkpoint and verify its exact logical state.
        pool.restore(0, 1900, checkpoint)
        pool.begin_request(0, 1900, checkpoint.length)
        for layer_pos, layer_id in enumerate(LAYER_IDS):
            state = pool.get_kda_state(layer_id, 0)
            for value, expected in zip(state, (part[layer_pos] for part in checkpoint_before)):
                assert_exact(value, expected, "checkpoint restored")
        pool.finish_request(0, 1900, checkpoint.length + 1)
        pool.release_request(0, 1900)
        # Fresh UIDs must get zero working states, and stale releases must be inert.
        for table in tables:
            pool.begin_request(table, 1100 + table, 0)
            pool.release_request(table, 100 + table)
            state = pool.get_kda_state(TARGET_LAYER, table)
            assert not state.conv.count_nonzero() and not state.recurrent.count_nonzero()
        report["recycled"] = run_forward(
            layer, pool, tables, lengths, (False,) * batch, output, state_dtype=state_dtype
        )
        for table, length in zip(tables, lengths):
            pool.finish_request(table, 1100 + table, length)
            pool.release_request(table, 1100 + table)
        for value, expected in zip(checkpoint_values(pool, checkpoint), checkpoint_before):
            assert_exact(value, expected, "immutable checkpoint after UID recycle")
        if arena is not None:
            arena.check_integrity()
        report.update(
            checkpoint_immutable=True,
            checkpoint_restore_exact=True,
            stale_release_inert=True,
            slot_uid_reuse_checked=True,
        )
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        return report
    finally:
        pool.reset()
        if arena is not None:
            arena.check_integrity()
        assert pool.is_idle


def case_matrix(large=False):
    cases = [(3,), tuple([1] * 8), (1, 2, 13, 65, 127, 128, 129, 257), tuple([1] * 48)]
    if large:
        cases += [(1, 65, 2048, 32768), tuple([1, 2, 13, 64, 65, 127] * 8)]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--large", action="store_true", help="Include long and mixed 48-row cases")
    parser.add_argument(
        "--profile", action="store_true", help="Retain launch trace for one mixed case"
    )
    args = parser.parse_args()
    if os.environ.get("MINISGL_GPUQ") != "1":
        parser.error("Run through gpuq; set MINISGL_GPUQ=1 only inside the queued job")
    from minisgl.distributed import DistributedInfo

    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        parser.error("Native FlashInfer KDA requires SM100 or SM103")
    output = args.output / ("rank" + os.environ.get("RANK", "0"))
    output.mkdir(parents=True, exist_ok=False)
    cases = case_matrix(args.large)
    reports = []
    with (
        torch.inference_mode(),
        patch("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1)),
        patch("minisgl.distributed.layout._DCP_SIZE", 1),
    ):
        for state_dtype in STATE_DTYPES:
            for index, lengths in enumerate(cases):
                for mode in ("cold", "continued", "mixed"):
                    for unified in (False, True):
                        dtype_name = str(state_dtype).removeprefix("torch.")
                        tag = f"state-{dtype_name}-case{index}-{mode}-unified{unified}"
                        result = run_case(
                            lengths,
                            mode,
                            unified,
                            output / (tag + ".json"),
                            state_dtype=state_dtype,
                            profile=(
                                args.profile
                                and state_dtype == torch.bfloat16
                                and index == 2
                                and mode == "mixed"
                                and unified
                            ),
                        )
                        row = {
                            "case": tag,
                            "state_dtype": str(state_dtype),
                            "adapter_exact_parity": True,
                            "slot_uid_reuse_checked": result["slot_uid_reuse_checked"],
                        }
                        reports.append(row)
                        print(json.dumps(row), flush=True)
    summary = {
        "scope": __doc__,
        "sdk_commit": SDK_COMMIT,
        "cases": reports,
        "state_dtypes": [str(dtype) for dtype in STATE_DTYPES],
        "status": "native-adapter-component-parity-passed-with-all-T1-chunk-fallback",
        "independent_arithmetic_qualified": False,
        "full_model_qualified": False,
        "performance_qualified": False,
        "prefill_graph_qualified": False,
        "decode_qualified": False,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
