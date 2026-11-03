"""Layer and phase composition must preserve each child's independent metadata."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.base import (
    BaseAttnBackend,
    HybridAttnBackend,
    HybridAttnMetadata,
    PrefillDecodeAttnBackend,
)


class Backend(BaseAttnBackend):
    def __init__(self, name, events, *, fail=False):
        self.name, self.events, self.fail = name, events, fail

    def prepare_metadata(self, batch):
        batch.attn_metadata = SimpleNamespace(
            name=self.name, get_last_indices=lambda bs: torch.arange(bs)
        )
        self.events.append((self.name, "prepare"))

    def forward_layer(self, layer, layer_id, x, batch, cache, page_table):
        assert batch.attn_metadata.name == self.name
        self.events.append((self.name, layer_id))
        if self.fail:
            raise RuntimeError("kernel failed")
        return x + layer_id

    def forward(self, q, k, v, layer_id, batch):
        return self.forward_layer(None, layer_id, q, batch, None, None)

    def init_capture_graph(self, max_seq_len, bs_list):
        self.events.append((self.name, "init"))

    def prepare_for_capture(self, batch):
        self.prepare_metadata(batch)

    def prepare_for_replay(self, batch):
        assert batch.attn_metadata.name == self.name
        self.events.append((self.name, "replay"))

    def finish_capture(self, batch):
        self.events.append((self.name, "finish"))


def make_backend(events, **kwargs):
    return HybridAttnBackend(
        Backend("kda", events, **kwargs), Backend("mla", events), lambda layer_id: layer_id % 2 == 0
    )


def test_layer_predicate_dispatches_both_apis_and_preserves_child_metadata():
    events = []
    backend = make_backend(events)
    batch = SimpleNamespace()
    backend.prepare_metadata(batch)
    metadata = batch.attn_metadata
    for layer_id in range(4):
        torch.testing.assert_close(
            backend.forward_layer(None, layer_id, torch.zeros(1), batch, None, None),
            torch.tensor([float(layer_id)]),
        )
        backend.forward(torch.zeros(1), None, None, layer_id, batch)
        assert batch.attn_metadata is metadata
    assert events[2:] == [
        (kind, i) for i in range(4) for kind in ["kda" if i % 2 == 0 else "mla"] * 2
    ]
    assert isinstance(metadata, HybridAttnMetadata)
    assert metadata.get_last_indices(2).tolist() == [0, 1]


def test_child_failure_restores_parent_metadata():
    backend = make_backend([], fail=True)
    batch = SimpleNamespace()
    backend.prepare_metadata(batch)
    metadata = batch.attn_metadata
    with pytest.raises(RuntimeError, match="kernel failed"):
        backend.forward_layer(None, 0, torch.zeros(1), batch, None, None)
    assert batch.attn_metadata is metadata
    assert metadata.kda.name == "kda" and metadata.mla.name == "mla"


def test_phase_selection_composes_with_layer_selection():
    events = []
    prefill, decode = make_backend(events), make_backend(events)
    backend = PrefillDecodeAttnBackend(prefill, decode)
    batch = SimpleNamespace(is_prefill=True)
    for is_prefill, expected in ((True, prefill), (False, decode)):
        batch.is_prefill = is_prefill
        backend.prepare_metadata(batch)
        backend.forward_layer(None, 1, torch.zeros(1), batch, None, None)
        assert events[-1] == ("mla", 1)
        assert batch.attn_metadata.mla.name == expected.mla_backend.name
    backend.init_capture_graph(32, [1])
    backend.prepare_for_replay(batch)
    backend.finish_capture(batch)
    assert events[-6:] == [
        ("kda", "init"),
        ("mla", "init"),
        ("kda", "replay"),
        ("mla", "replay"),
        ("kda", "finish"),
        ("mla", "finish"),
    ]


def test_hybrid_cleanup_releases_both_children_after_failure():
    events = []
    backend = make_backend(events)

    def fail(batch):
        events.append(("kda", "failed"))
        raise RuntimeError("release failed")

    backend.kda_backend.finish_capture = fail
    with pytest.raises(RuntimeError, match="release failed"):
        backend.finish_capture(SimpleNamespace())
    assert events == [("kda", "failed"), ("mla", "finish")]


@pytest.mark.parametrize("precision,kernel", [("strict", "sgl"), ("default", "flashinfer")])
def test_flashinfer_factory_owns_its_kda_kernel(monkeypatch, precision, kernel):
    import minisgl.attention.fi as fi
    import minisgl.core as core
    from minisgl.attention import create_attention_backend

    cache = SimpleNamespace(device=torch.device("cpu"))
    ctx = core.Context(page_size=1, bf16_precision=precision)
    ctx.kv_cache = cache
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    monkeypatch.setattr(fi, "FlashInferMLABackend", lambda config: Backend("mla", []))
    config = SimpleNamespace(cache_config=object(), is_kda_layer=lambda layer_id: layer_id == 0)
    backend = create_attention_backend("fi", config)
    assert isinstance(backend, HybridAttnBackend)
    assert backend.kda_backend.kernel.name == kernel
    assert backend.is_kda_layer(0) and not backend.is_kda_layer(1)


def test_attention_provider_registration_is_explicit_and_idempotent(monkeypatch):
    import sys
    import types

    from minisgl.attention import load_attention_provider

    calls = []
    module = types.ModuleType("test_complete_attention_provider")
    module.register_attention_backends = lambda: calls.append("registered")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    load_attention_provider(module.__name__)
    load_attention_provider(module.__name__)
    assert calls == ["registered"]


@pytest.mark.parametrize("name", ["recurrent", "recurrent-fi", "recurrent-fi-triton"])
def test_removed_backend_names_are_rejected(name):
    from argparse import ArgumentTypeError

    from minisgl.attention import validate_attn_backend

    with pytest.raises(ArgumentTypeError, match="Unsupported Attention Backend"):
        validate_attn_backend(name)
