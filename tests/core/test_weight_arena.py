import pytest
import torch
from minisgl.layers.base import BaseOP, OPList
from minisgl.layers.linear import Linear
from minisgl.layers.weight import allocate_weight_views


def test_weight_arena_alignment_storage_and_independence():
    expected = {
        "a": torch.empty(3, 7),
        "b": torch.empty(5),
        "packed": torch.empty(3, 16, dtype=torch.uint8),
    }
    views = allocate_weight_views(expected, torch.device("cpu"))
    assert views["a"].untyped_storage().data_ptr() == views["b"].untyped_storage().data_ptr()
    assert views["a"].untyped_storage().data_ptr() != views["packed"].untyped_storage().data_ptr()
    for name, tensor in views.items():
        assert tensor.shape == expected[name].shape and tensor.dtype == expected[name].dtype
        assert tensor.data_ptr() % 256 == 0
        assert tensor.is_contiguous()
    views["a"].fill_(2)
    views["b"].fill_(3)
    assert views["a"].eq(2).all()
    with pytest.raises(ValueError, match="alignment"):
        allocate_weight_views(expected, torch.device("cpu"), alignment=3)


def test_streamed_assignment_copies_before_a_reused_source_buffer_changes():
    model = BaseOP()
    with torch.device("meta"):
        model.layers = OPList([Linear(4, 4), Linear(4, 4)])
    scratch = torch.empty(4, 4)

    def stream():
        for i, name in enumerate(model.state_dict()):
            scratch.fill_(i + 1)
            yield name, scratch

    model.load_weights(stream(), device=torch.device("cpu"))
    first, second = model.layers.op_list
    assert first.weight.eq(1).all() and second.weight.eq(2).all()
    assert first.weight.untyped_storage().data_ptr() == second.weight.untyped_storage().data_ptr()


@pytest.mark.parametrize("streamed", [False, True])
def test_after_load_runs_bottom_up_once_and_rebuilds_on_reload(streamed):
    events = []

    class Derived(Linear):
        def after_load(self):
            self._derived = self.weight.sum().item()
            events.append(self._derived)

    class Root(OPList):
        def after_load(self):
            assert all(hasattr(child, "_derived") for child in self.op_list)
            events.append("root")

    model = Root([Derived(2, 2), Derived(2, 2)])
    for value in (1.0, 3.0):
        state = {
            name: torch.full_like(tensor, value) for name, tensor in model.state_dict().items()
        }
        if streamed:
            model.load_weights(state.items(), device=torch.device("cpu"))
        else:
            model.load_state_dict(state)
        assert events[-3:] == [4 * value, 4 * value, "root"]
    assert len(events) == 6
