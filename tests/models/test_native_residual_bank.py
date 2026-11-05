"""All native bank row counts with varied weights and repeated immutable reads."""

import os
from types import SimpleNamespace

import pytest
import torch
from minisgl.layers.attn_residual import ResidualBank, attention_residual
from minisgl.layers.norm import TorchRMSNorm

pytestmark = pytest.mark.skipif(
    os.environ.get("MINISGL_RUN_CUDA_TESTS") != "1", reason="explicit GPU gate"
)
CASES = [(t, 7168, v, 8) for t in (1, 8, 48, 32768) for v in range(1, 9)] + [
    (8, 7168, 9, 16),
    (257, 1024, 3, 8),
]


@pytest.mark.parametrize("tokens,hidden,valid,capacity", CASES)
@pytest.mark.parametrize("normalized", [False, True])
def test_bank_repeated_reads(tokens, hidden, valid, capacity, normalized, request):
    request.node.user_properties.append(("native_covered", valid <= 8))
    torch.manual_seed(tokens + valid)
    head = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.2
    history = torch.randn(tokens, valid, hidden, device="cuda", dtype=head.dtype) * 0.2
    norms = []
    projs = []
    targets = []
    for _ in range(3):
        norm = TorchRMSNorm(hidden, 1e-6)
        norm.weight = 1 + torch.randn(hidden, device="cuda", dtype=head.dtype) * 0.1
        target = TorchRMSNorm(hidden, 1e-6)
        target.weight = 1 + torch.randn_like(norm.weight) * 0.1
        norms.append(norm)
        targets.append(target if normalized else None)
        projs.append(
            SimpleNamespace(weight=torch.randn(1, hidden, device="cuda", dtype=head.dtype) * 0.02)
        )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):

        def persistent():
            bank = ResidualBank(head, capacity)
            bank._storage.fill_(float("nan"))
            for raw in history.unbind(1):
                bank.append(raw)
            outputs = tuple(
                attention_residual(head, bank, n, p, t) for n, p, t in zip(norms, projs, targets)
            )
            return outputs, bank._storage

        def legacy():
            return tuple(
                attention_residual(head, list(history.unbind(1)), n, p, t)
                for n, p, t in zip(norms, projs, targets)
            )

        outputs, storage = persistent()
        expected = legacy()
        for actual, want in zip(outputs, expected):
            torch.testing.assert_close(actual, want, rtol=0, atol=0)
        torch.testing.assert_close(storage[:, :valid], history, rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured, saved = persistent()
        pointers = [x.data_ptr() for x in captured]
        for _ in range(3):
            head.mul_(-0.5)
            history.add_(0.001)
            graph.replay()
            assert [x.data_ptr() for x in captured] == pointers
            for actual, want in zip(captured, legacy()):
                torch.testing.assert_close(actual, want, rtol=0, atol=0)
            torch.testing.assert_close(saved[:, :valid], history, rtol=0, atol=0)
        graph.reset()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
