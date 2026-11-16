"""CPU checks of the real Engine forward dependency and token owner contract."""

import gc
import weakref
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from minisgl.engine.engine import Engine


def make_engine(monkeypatch, *, graph=False, copy_error=False):
    trace = []
    state = SimpleNamespace(current=None)

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, producer):
            assert self is copy and producer is forward
            trace.append(("wait", self.name, producer.name))

    forward, copy = Stream("forward"), Stream("copy")
    state.current = forward

    @contextmanager
    def stream_context(stream):
        previous = state.current
        state.current = stream
        try:
            yield
        finally:
            state.current = previous

    class Event:
        def record(self, stream):
            assert stream is copy and state.current is copy
            trace.append(("event", stream.name))
            self.stream = stream

        def synchronize(self):
            pytest.fail("forward must return without waiting for result D2H")

    class TokenIDs:
        def to(self, destination, *, non_blocking=False):
            if destination == torch.int32:
                assert state.current is forward
                return self  # Native FI already returns owned int32 IDs.
            assert destination == "cpu" and non_blocking and state.current is copy
            trace.append(("D2H", state.current.name))
            if copy_error:
                raise RuntimeError("injected copy error")
            return SimpleNamespace(values=(11, 17))

        def record_stream(self, stream):
            assert stream is copy and state.current is copy
            trace.append(("record-owner", stream.name))

    tokens = TokenIDs()
    token_ref = weakref.ref(tokens)
    queue = [tokens]
    del tokens

    class Logits:
        def __getitem__(self, rows):
            return self

    logits = Logits()

    def compute():
        assert state.current is forward
        trace.append(("model", "graph" if graph else "eager"))
        return logits

    def sample(value, args):
        assert value is logits and state.current is forward
        trace.append(("sample", "forward"))
        return queue.pop()

    monkeypatch.setattr(torch.cuda, "current_stream", lambda: state.current)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    engine = object.__new__(Engine)
    engine.stream, engine.result_copy_stream = forward, copy
    engine.ctx = SimpleNamespace(
        forward_batch=lambda batch: nullcontext(),
        kv_cache=SimpleNamespace(model_forward=lambda batch: nullcontext()),
    )
    engine.model = SimpleNamespace(forward=compute)
    engine.graph_runner = SimpleNamespace(
        can_use_cuda_graph=lambda batch: graph, replay=lambda batch: compute()
    )
    engine.sampler = SimpleNamespace(sample=sample)
    req = SimpleNamespace(complete_one=lambda: trace.append(("complete", "request")))
    batch = SimpleNamespace(reqs=[req], size=1)
    return engine, batch, trace, state, token_ref


@pytest.mark.parametrize("graph", [False, True], ids=["eager", "graph"])
def test_forward_returns_owned_IDs_with_copy_stream_dependency(monkeypatch, graph):
    engine, batch, trace, state, token_ref = make_engine(monkeypatch, graph=graph)
    output = engine.forward_batch(batch, object())
    assert state.current is engine.stream
    assert trace.count(("model", "graph" if graph else "eager")) == 1
    assert trace.count(("record-owner", "copy")) == 1
    assert (
        trace.index(("sample", "forward"))
        < trace.index(("wait", "copy", "forward"))
        < trace.index(("D2H", "copy"))
        < trace.index(("event", "copy"))
    )
    assert output.next_tokens_gpu is token_ref()
    assert output.next_tokens_cpu.values == (11, 17)
    assert output.copy_done_event.stream is engine.result_copy_stream
    gc.collect()
    assert token_ref() is not None  # The returned owner keeps IDs alive.
    del output
    gc.collect()
    assert token_ref() is None
    assert not torch.cuda.is_initialized()


def test_copy_failure_propagates_once_without_retry_or_producer_wait(monkeypatch):
    engine, batch, trace, state, _ = make_engine(monkeypatch, copy_error=True)
    with pytest.raises(RuntimeError, match="injected copy error"):
        engine.forward_batch(batch, object())
    assert state.current is engine.stream
    assert trace.count(("D2H", "copy")) == 1
    assert ("wait", "forward", "copy") not in trace
    assert not any(row[0] == "event" for row in trace)
    assert not torch.cuda.is_initialized()
