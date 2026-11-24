"""Layer and phase composition must preserve each child's independent metadata."""

from types import SimpleNamespace

import pytest
import torch
from minisgl.attention.base import (
    HybridAttnBackend,
    HybridAttnMetadata,
    LayerAttnBackend,
    PrefillDecodeAttnBackend,
)


class Backend(LayerAttnBackend):
    def __init__(self, name, events, *, fail=False):
        self.name, self.events, self.fail = name, events, fail

    def build_metadata(self, batch, *, cu_seqlens_q=None):
        self.events.append((self.name, "prepare"))
        return SimpleNamespace(
            cu_seqlens_q=torch.arange(3) if cu_seqlens_q is None else cu_seqlens_q,
            name=self.name,
            get_last_indices=lambda bs: torch.arange(bs),
        )

    def forward_layer(
        self, layer, layer_id, x, batch, cache, page_table, *, residual=None, metadata=None
    ):
        assert metadata.name == self.name
        self.events.append((self.name, layer_id))
        if self.fail:
            raise RuntimeError("kernel failed")
        return x + layer_id if residual is None else x + layer_id + residual

    def forward(self, q, k, v, layer_id, batch):
        return self.forward_layer(None, layer_id, q, batch, None, None)

    def init_capture_graph(self, max_seq_len, bs_list):
        self.events.append((self.name, "init"))

    def capture_metadata(self, batch):
        return self.build_metadata(batch)

    def replay_metadata(self, batch, metadata):
        assert metadata.name == self.name
        self.events.append((self.name, "replay"))

    def finish_capture(self, batch):
        self.events.append((self.name, "finish"))


def make_backend(events, **kwargs):
    return HybridAttnBackend(
        Backend("kda", events, **kwargs), Backend("mla", events), lambda layer_id: layer_id % 2 == 0
    )


def test_layer_predicate_dispatches_and_preserves_child_metadata():
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
        assert batch.attn_metadata is metadata
    assert events[2:] == [(kind, i) for i in range(4) for kind in ["kda" if i % 2 == 0 else "mla"]]
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


@pytest.mark.parametrize("name,kernel", [("flashinfer", "flashinfer"), ("auto", "triton")])
def test_attention_factory_owns_its_kda_kernel(monkeypatch, name, kernel):
    import minisgl.attention.mla.flashinfer as flashinfer
    import minisgl.core as core
    from minisgl.attention import create_attention_backend
    from minisgl.distributed import DistributedInfo

    monkeypatch.setattr("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1))
    cache = SimpleNamespace(device=torch.device("cpu"))
    ctx = core.Context(page_size=1)
    ctx.kv_cache = cache
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    monkeypatch.setattr(flashinfer, "FlashInferMLABackend", lambda config: Backend("mla", []))
    config = SimpleNamespace(cache_config=SimpleNamespace(recurrent_layers=(0,), mla_layers=(1,)), has_recurrent_state=True, is_kda_layer=lambda layer_id: layer_id == 0)
    backend = create_attention_backend(name, config)
    assert isinstance(backend, HybridAttnBackend)
    assert backend.kda_backend.prefill_kernel.name == kernel
    assert backend.is_kda_layer(0) and not backend.is_kda_layer(1)


@pytest.mark.parametrize("recurrent,mla,expected", [((0,), (), "kda"), ((), (0,), "mla")])
def test_single_kind_layout_constructs_only_its_backend(recurrent, mla, expected):
    from minisgl.attention.factory import _compose_layers

    layout = SimpleNamespace(recurrent_layers=recurrent, mla_layers=mla)
    calls = []
    def make(name):
        calls.append(name)
        return name
    assert _compose_layers(
        SimpleNamespace(cache_config=layout), lambda: make("kda"), lambda: make("mla")
    ) == expected
    assert calls == [expected]


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


@pytest.mark.parametrize(
    "name", ["fa", "fi", "sgl", "recurrent", "recurrent-fi", "recurrent-fi-triton"]
)
def test_removed_backend_names_are_rejected(name):
    from argparse import ArgumentTypeError

    from minisgl.attention import validate_attn_backend

    with pytest.raises(ArgumentTypeError, match="Unsupported Attention Backend"):
        validate_attn_backend(name)


def test_phase_and_layer_wrappers_forward_explicit_residual():
    events = []
    backend = PrefillDecodeAttnBackend(make_backend(events), make_backend(events))
    residual = torch.full((1,), 7.0)
    for is_prefill, layer_id in ((True, 0), (True, 1), (False, 0), (False, 1)):
        batch = SimpleNamespace(is_prefill=is_prefill)
        backend.prepare_metadata(batch)
        metadata = batch.attn_metadata
        result = backend.forward_layer(
            None, layer_id, torch.zeros(1), batch, None, None, residual=residual
        )
        assert result.item() == 7 + layer_id and batch.attn_metadata is metadata
