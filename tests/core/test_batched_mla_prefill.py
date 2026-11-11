"""Batched MLA dispatch and cyclic prefix layouts; no CUDA execution."""

import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.mla import MLAOutput
from minisgl.attention.mla.backend import MLAAttentionBackend
from minisgl.attention.mla.flashinfer import FlashInferMLAAdapter
from minisgl.attention.mla.prefill import MLAPrefillMetadata


def batch_of(prefixes, lengths):
    reqs = [
        SimpleNamespace(table_idx=i, cached_len=p, extend_len=n, device_len=p + n)
        for i, (p, n) in enumerate(zip(prefixes, lengths))
    ]
    return SimpleNamespace(
        reqs=reqs,
        is_decode=False,
        positions=torch.arange(sum(lengths)),
        out_loc=torch.arange(sum(lengths), dtype=torch.int32),
    )


@pytest.mark.parametrize("rank", range(8))
def test_prefix_chunks_preserve_arbitrary_cyclic_phase_and_empty_requests(rank):
    prefixes, lengths, capacity = (11, 2, 0), (1, 2, 1), 10
    table = torch.stack([torch.arange(32, dtype=torch.int32) + i * 1024 for i in range(3)])
    plan = MLAPrefillMetadata.from_batch(
        batch_of(prefixes, lengths), table, SimpleNamespace(rank=rank, size=8), capacity
    )
    assert plan.queries.lengths.tolist() == list(lengths)
    assert plan.queries.cu_lengths.dtype == torch.int32
    assert plan.queries.cu_lengths.tolist() == [0, 1, 3, 4]
    calls = []
    for chunk_index, chunk in enumerate(plan.prefix_chunks):
        start = chunk_index * (capacity // 3)
        joined, expected, active = [], [], []
        for row, prefix in enumerate(prefixes):
            count = min(max(prefix - start, 0), capacity // 3)
            padded = (start % 8 + count + 7) // 8 if count else 0
            positions = torch.arange(padded * 8) + start - start % 8
            values = positions.float() + row * 1024
            joined.append(torch.stack((values, values + 0.5), -1))
            active.append((positions >= start) & (positions < start + count))
            values = table[row, start : start + count].float()
            expected.append(torch.stack((values, values + 0.5), -1))
        joined = torch.cat(joined)
        active = torch.cat(active)

        def gather(padded, total):
            calls.append(total)
            assert total == len(joined) and len(padded) * 8 == total
            valid = active[rank::8]
            torch.testing.assert_close(padded[valid], joined[rank::8][valid], rtol=0, atol=0)
            return joined

        def cache_gather(layer_id, locations):
            assert layer_id == 5
            values = locations.float()
            return values[:, None], (values + 0.5)[:, None]

        actual = chunk.gather(
            SimpleNamespace(gather_mla=cache_gather),
            5,
            SimpleNamespace(size=8, gather_interleaved=gather),
            torch.float32,
        )
        torch.testing.assert_close(torch.cat(actual, -1), torch.cat(expected), rtol=0, atol=0)
        assert chunk.layout.lengths.sum() <= capacity
        assert chunk.has_empty
    assert len(calls) == len(plan.prefix_chunks) == 4


class CpuCudaTensor(torch.Tensor):
    @property
    def is_cuda(self):
        return True


@pytest.mark.parametrize(
    "prefixes,lengths,capacity",
    [
        ((0,) * 8, (2,) * 8, 128),
        ((0,) * 48, (1,) * 48, 128),
        ((0, 7), (2, 1), 6),
    ],
)
def test_native_ragged_calls_are_batched_with_fp8_boundary_and_bounded_prefixes(
    monkeypatch, prefixes, lengths, capacity
):
    batch = batch_of(prefixes, lengths)
    table = torch.stack([torch.arange(32, dtype=torch.int32) + i * 64 for i in range(len(lengths))])
    cp = SimpleNamespace(rank=0, size=1)
    plan = MLAPrefillMetadata.from_batch(batch, table, cp, capacity)
    calls, projections, merges, fixes = [], [], [], []

    def kernel(**kwargs):
        calls.append(kwargs)
        assert kwargs["batch_size"] == len(lengths)
        assert (
            kwargs["query"].dtype
            == kwargs["key"].dtype
            == kwargs["value"].dtype
            == torch.float8_e4m3fn
        )
        assert kwargs["out"].dtype == torch.bfloat16
        for name in ("seq_lens", "cum_seq_lens_q", "cum_seq_lens_kv"):
            assert kwargs[name].dtype == torch.int32
        kwargs["out"].fill_(len(calls))
        if kwargs["return_lse"]:
            return kwargs["out"], torch.zeros((sum(lengths), 2), dtype=torch.float32)
        return kwargs["out"]

    def project(x, w):
        projections.append(len(x))
        return torch.zeros((len(x), len(w)), dtype=x.dtype)

    def merge(a, a_lse, b, b_lse, out, out_lse):
        merges.append(True)
        out.copy_(a)
        out_lse.copy_(a_lse)

    monkeypatch.setitem(
        sys.modules, "flashinfer.prefill", SimpleNamespace(trtllm_ragged_attention_deepseek=kernel)
    )
    monkeypatch.setitem(
        sys.modules, "tvm_ffi", SimpleNamespace(use_torch_stream=lambda: nullcontext())
    )
    monkeypatch.setitem(
        sys.modules, "sglang.srt.layers.attention.merge_state", SimpleNamespace(merge_state=merge)
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.kernels.ops.attention.fixup_zero_kv",
        SimpleNamespace(fixup_zero_kv_rows=lambda *args: fixes.append(True)),
    )
    monkeypatch.setattr("minisgl.layers.linear.linear_bf16", project)
    q = torch.zeros(sum(lengths), 2, 192, dtype=torch.bfloat16).as_subclass(CpuCudaTensor)
    latent = torch.zeros(sum(lengths), 4, dtype=torch.bfloat16)
    key = torch.zeros(sum(lengths), 64, dtype=torch.bfloat16)
    weight = torch.zeros(2 * 256, 4, dtype=torch.bfloat16)
    cache = SimpleNamespace(
        gather_mla=lambda layer, locations: (
            torch.zeros(len(locations), 4, dtype=torch.bfloat16),
            torch.zeros(len(locations), 64, dtype=torch.bfloat16),
        )
    )
    adapter = FlashInferMLAAdapter(kv_dtype=torch.float8_e4m3fn)
    adapter._expanded_workspace = torch.empty(1, dtype=torch.uint8)
    output = adapter.prefill_batch(
        q,
        latent,
        key,
        weight,
        None,
        None,
        nope_dim=128,
        value_dim=128,
        metadata=plan,
        cache=cache,
        layer_id=0,
        cp=cp,
    )
    assert isinstance(output, MLAOutput) and output.values.shape == (sum(lengths), 2, 128)
    assert len(calls) == len(projections) == 1 + len(plan.prefix_chunks)
    assert calls[0]["is_causal"] and calls[0]["o_sf_scale"] == 1.0
    assert len(merges) == len(fixes) == len(plan.prefix_chunks)
    assert all(not call["is_causal"] and call["o_sf_scale"] == -1 for call in calls[1:])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("covered", [False, True])
def test_actual_backend_dispatches_one_batch_and_finishes_once(monkeypatch, covered):
    import minisgl.attention.mla.backend as module

    batch = batch_of((0, 0, 0), (1, 2, 3))
    table = torch.arange(96, dtype=torch.int32).reshape(3, 32)
    cp = SimpleNamespace(rank=0, size=1)
    calls = []

    class Adapter:
        def prepare_prefill(self, batch, page_table, cp, cu):
            return MLAPrefillMetadata.from_batch(batch, page_table, cp, 128, cu_lengths=cu)

        def prefill_batch(self, q, latent, key, weight, positions, history, **kwargs):
            calls.append(("attention", len(q), kwargs["metadata"].queries.lengths.tolist()))
            return MLAOutput(torch.zeros(len(q), 1, 128)) if covered else None

        def __call__(self, q, *args, **kwargs):
            assert not covered
            calls.append(("fallback", len(q)))
            return torch.zeros(len(q), 1, 128)

    class Layer:
        def project(self, x, positions, *, is_decode=False):
            calls.append(("project", len(x)))
            return torch.zeros(len(x), 1, 192), torch.zeros(len(x), 4), torch.zeros(len(x), 64)

        def attend_partial(
            self,
            q,
            latent,
            key,
            positions,
            history,
            *,
            context_sharded,
            attention,
            causal_layout=None,
        ):
            assert not context_sharded
            result = attention(
                q,
                latent,
                key,
                torch.zeros(256, 4),
                positions,
                history,
                nope_dim=128,
                value_dim=128,
            )
            return result if isinstance(result, MLAOutput) else MLAOutput(result)

        def finish(self, x, result):
            calls.append(("finish", len(x)))
            return result.values

    cache = SimpleNamespace(
        device=torch.device("cpu"),
        store_mla=lambda *args, **kwargs: calls.append(("store", len(args[1]))),
    )
    monkeypatch.setattr(module, "ContextParallel", lambda: cp)
    monkeypatch.setattr(module, "get_global_ctx", lambda: SimpleNamespace(page_table=table))
    backend = MLAAttentionBackend(SimpleNamespace(), attention=Adapter(), cache=cache)
    backend.prepare_metadata(batch)
    result = backend.forward_layer(Layer(), 0, torch.zeros(6, 8), batch, cache, table)
    assert result.shape == (6, 1, 128)
    expected = ["project", "store", "attention"]
    expected += [] if covered else ["fallback"] * 3
    assert [x[0] for x in calls] == [*expected, "finish"]
    assert calls[2] == ("attention", 6, [1, 2, 3])
    if not covered:
        assert [call[1] for call in calls if call[0] == "fallback"] == [1, 2, 3]
