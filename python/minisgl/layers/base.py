from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Generic, Iterable, List, TypeAlias, TypeVar

import torch

_STATE_DICT: TypeAlias = Dict[str, torch.Tensor]


def _concat_prefix(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


class BaseOP:
    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...

    def children(self):
        return (
            op
            for name, op in vars(self).items()
            if not name.startswith("_") and isinstance(op, BaseOP)
        )

    def apply(self, fn):
        for child in self.children():
            child.apply(fn)
        fn(self)

    def after_load(self) -> None:
        """Rebuild optional inference metadata after all checkpoint tensors exist."""

    def validate_cuda_graph(self) -> None:
        """Reject an execution path that needs host decisions during replay."""

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}

        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                result[_concat_prefix(prefix, name)] = param
            elif isinstance(param, BaseOP):
                param.state_dict(prefix=_concat_prefix(prefix, name), result=result)

        return result

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], *, device: torch.device | None = None
    ) -> None:
        """Assign a checkpoint stream without staging a second full state dictionary.

        Expected tensors may be on meta. Each item is validated before assignment;
        a failure leaves an incomplete model which must not be used for inference.
        Unlike load_state_dict, this accepts an iterator and does not consume or
        mutate a caller-owned dictionary.
        """
        remaining = self.state_dict()
        if device is not None:
            from .weight import allocate_weight_views

            storage = allocate_weight_views(remaining, device)
        else:
            storage = None
        for name, item in weights:
            expected = remaining.pop(name, None)
            if expected is None:
                raise ValueError(f"Unexpected or duplicate checkpoint tensor: {name}")
            if item.shape != expected.shape or item.dtype != expected.dtype:
                raise ValueError(
                    f"Checkpoint tensor {name}: {tuple(item.shape)}/{item.dtype} != "
                    f"{tuple(expected.shape)}/{expected.dtype}"
                )
            if storage is not None:
                target = storage.pop(name)
                target.copy_(item)
                item = target
            parent = self
            parts = name.split(".")
            for part in parts[:-1]:
                parent = (
                    parent.get_op(part) if isinstance(parent, OPList) else getattr(parent, part)
                )
            setattr(parent, parts[-1], item)
        if remaining:
            sample = list(remaining)[:5]
            raise ValueError(f"Missing {len(remaining)} checkpoint tensors, including {sample}")
        self.apply(lambda op: op.after_load())

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for name, param in self.__dict__.items():
            if name.startswith("_"):
                continue
            if isinstance(param, torch.Tensor):
                item = state_dict.pop(_concat_prefix(prefix, name))
                assert isinstance(item, torch.Tensor)
                assert param.shape == item.shape and param.dtype == item.dtype
                setattr(self, name, item)
            elif isinstance(param, BaseOP):
                param.load_state_dict(
                    state_dict, prefix=_concat_prefix(prefix, name), _internal=True
                )

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
        if not _internal:
            self.apply(lambda op: op.after_load())


class StateLessOP(BaseOP):
    def __init__(self):
        super().__init__()

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        return result if result is not None else {}


T = TypeVar("T", bound=BaseOP)


class OPList(BaseOP, Generic[T]):
    def __init__(self, ops: List[T], *, indices: Iterable[int] | None = None):
        super().__init__()
        self.op_list = ops
        self._indices = list(range(len(ops)) if indices is None else indices)
        if len(self._indices) != len(ops) or len(set(self._indices)) != len(ops):
            raise ValueError("Operation indices must be unique and cover the local list")
        self._lookup = {str(key): i for i, key in enumerate(self._indices)}

    def items(self):
        return zip(self._indices, self.op_list)

    def children(self):
        return iter(self.op_list)

    def get_op(self, key):
        return self.op_list[self._lookup[str(key)]]

    def __contains__(self, key):
        return str(key) in self._lookup

    def set_op(self, key, op):
        self.op_list[self._lookup[str(key)]] = op

    def state_dict(self, *, prefix: str = "", result: _STATE_DICT | None = None) -> _STATE_DICT:
        result = result if result is not None else {}
        for i, op in self.items():
            op.state_dict(prefix=_concat_prefix(prefix, str(i)), result=result)
        return result

    def load_state_dict(
        self,
        state_dict: _STATE_DICT,
        *,
        prefix: str = "",
        _internal: bool = False,
    ) -> None:
        for i, op in self.items():
            op.load_state_dict(state_dict, prefix=_concat_prefix(prefix, str(i)), _internal=True)

        if not _internal and state_dict:
            raise RuntimeError(f"Unexpected keys in state_dict: {list(state_dict.keys())}")
        if not _internal:
            self.apply(lambda op: op.after_load())
