"""Opt-in operator-boundary tensor evidence; no inference or comparison policy.

A step scope records framework/library tensors before and after Python calls.
Immutable parameters are identified by storage; their values remain in the pinned
checkpoint; workspace/lookup allocations keep metadata, and pooled states keep
only indexed rows. Each dynamic value gets its own compact CPU payload, since
native in-place kernels need not increment PyTorch's tensor version counter.
Nsight NVTX ranges join these events to actual CUDA launch geometry.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.multiprocessing.reductions import StorageWeakRef


class TensorTrace:
    def __init__(self, output, *, prefixes, max_bytes=16 << 30, nvtx=False):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "tensors").mkdir()
        self.events = (self.output / "events.jsonl").open("x")
        self.prefixes, self.max_bytes, self.nvtx = tuple(prefixes), max_bytes, nvtx
        self.bytes = self.sequence = self.payloads = 0
        self.parameters, self.aliases, self.frames, self.models = {}, {}, {}, set()
        self.owners, self._alias_count = {}, 0
        self.active = False
        self._profile = self._jit = None
        self._installed = False

    @staticmethod
    def storage(tensor):
        return str(tensor.device), tensor.untyped_storage()._cdata

    def register_model(self, model):
        """Identify static storage without copying checkpoint tensors to the host."""
        seen = set()

        def visit(value, name):
            if isinstance(value, torch.Tensor):
                storage = self.storage(value)
                previous = self.parameters.get(storage)
                if previous is None or previous[0].expired():
                    self.parameters[storage] = (StorageWeakRef(value.untyped_storage()), name)
                return
            if id(value) in seen:
                return
            seen.add(id(value))
            self.owners.setdefault(
                id(value), name.replace("._modules.", ".").replace(".op_list.", ".")
            )
            if isinstance(value, dict):
                children = value.items()
            elif isinstance(value, (tuple, list)):
                children = enumerate(value)
            elif dataclasses.is_dataclass(value):
                children = (
                    (field.name, getattr(value, field.name)) for field in dataclasses.fields(value)
                )
            elif isinstance(value, SimpleNamespace) or type(value).__module__.startswith(
                ("minisgl.", "torch.nn.", "sglang.srt.models.", "sglang.srt.layers.")
            ):
                children = vars(value).items() if hasattr(value, "__dict__") else ()
            else:
                return
            for key, item in children:
                if any(
                    part in str(key).lower()
                    for part in ("workspace", "kv_cache", "state", "config")
                ):
                    continue
                visit(item, f"{name}.{key}")

        visit(model, type(model).__name__)

    def alias(self, tensor):
        storage = self.storage(tensor)
        previous = self.aliases.get(storage)
        if previous is None or previous[0].expired():
            previous = (StorageWeakRef(tensor.untyped_storage()), self._alias_count)
            self._alias_count += 1
            self.aliases[storage] = previous
        return previous[1]

    def snapshot(self, value, memo=None, *, rows=None, role=None):
        memo = {} if memo is None else memo
        if isinstance(value, torch.Tensor):
            storage = self.storage(value)
            meta = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "device": str(value.device),
                "storage_offset": value.storage_offset(),
                "data_ptr": value.data_ptr(),
                "nbytes": value.numel() * value.element_size(),
                "alias": self.alias(value),
            }
            if role is not None:
                return dict(meta, role=role)
            if rows is not None:
                meta["selection"] = {"axis": 0, "stored_indices": rows}
            parameter = self.parameters.get(storage)
            if parameter is not None and not parameter[0].expired():
                return dict(meta, role="parameter", parameter=parameter[1])
            key = (id(value), None if rows is None else tuple(rows))
            if key in memo:
                return dict(meta, payload=memo[key])
            if rows is not None:
                if value.ndim == 0:
                    raise IndexError("Cannot select rows from a scalar tensor")
                if any(not -value.shape[0] <= index < value.shape[0] for index in rows):
                    raise IndexError("Tensor trace row index is out of bounds")
            elements = value.numel() if rows is None else len(rows) * math.prod(value.shape[1:])
            size = elements * value.element_size()
            if self.bytes + size > self.max_bytes:
                raise RuntimeError("Tensor trace byte budget exceeded; trace is incomplete")
            payload = f"tensors/{self.payloads:08d}.pt"
            # Select on the source device so transfer/temporary memory follows
            # the selected-byte budget rather than the size of an entire pool.
            saved = value.detach() if rows is None else value.detach()[rows]
            saved = saved.to(device="cpu", copy=True).contiguous()
            torch.save(saved, self.output / payload)
            memo[key] = payload
            self.bytes += size
            self.payloads += 1
            return dict(meta, payload=payload)
        if isinstance(value, (dict, tuple, list)):
            if id(value) in memo:
                return {"container_ref": memo[id(value)]}
            reference = f"container:{len(memo)}"
            memo[id(value)] = reference
            if isinstance(value, dict):
                indices = next(
                    (
                        value[k]
                        for k in (
                            "cache_indices",
                            "initial_state_indices",
                            "ssm_state_indices",
                            "state_indices",
                        )
                        if isinstance(value.get(k), torch.Tensor)
                        and value[k].dtype in (torch.int32, torch.int64)
                        and value[k].numel() <= 4096
                    ),
                    None,
                )
                indices = None if indices is None else indices.detach().cpu().flatten().tolist()
                items = {}
                pools = {"ssm_states", "initial_state", "pool", "conv_states", "recurrent_pool"}
                lookups = {"cos_sin_cache", "cos_cache", "sin_cache", "freqs_cis"}
                for key, item in value.items():
                    name = str(key) if isinstance(key, (str, int)) else type(key).__name__
                    selection, item_role = None, role
                    if isinstance(item, torch.Tensor):
                        if "workspace" in name or name in {"float_ws", "int_ws"}:
                            item_role = "workspace"
                        elif name in lookups:
                            item_role = "lookup_table"
                        elif name in pools and item.ndim >= 3 and indices is not None:
                            selection = sorted({i for i in indices if 0 <= i < item.shape[0]})
                    items[name] = self.snapshot(item, memo, rows=selection, role=item_role)
            elif hasattr(value, "_fields"):
                items = {
                    key: self.snapshot(item, memo, role=role)
                    for key, item in zip(value._fields, value)
                }
            else:
                items = [self.snapshot(v, memo, role=role) for v in value]
            return {"container": reference, "kind": type(value).__name__, "items": items}
        if isinstance(value, float) and not math.isfinite(value):
            return {"scalar_float": str(value)}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, (torch.dtype, torch.device)):
            return str(value)
        return {"object_type": type(value).__module__ + "." + type(value).__qualname__}

    def emit(self, kind, **data):
        self.sequence += 1
        row = dict(
            sequence=self.sequence,
            kind=kind,
            pid=os.getpid(),
            tid=threading.get_native_id(),
            monotonic_ns=time.monotonic_ns(),
            wall_ns=time.time_ns(),
            **data,
        )
        self.events.write(json.dumps(row, allow_nan=False) + "\n")
        return self.sequence

    @staticmethod
    def state_indices(values):
        return {
            key: value.detach().cpu().tolist()
            for key, value in values.items()
            if any(part in key for part in ("indice", "index", "_idx", "_ids"))
            and isinstance(value, torch.Tensor)
            and value.dtype in (torch.int32, torch.int64)
            and value.numel() <= 4096
        }

    def _event(self, frame, event, arg):
        if self._profile is not None:
            self._profile(frame, event, arg)
        if not self.active or event not in ("call", "return"):
            return
        module = frame.f_globals.get("__name__", "")
        if not module.startswith(self.prefixes):
            return
        function = module + "." + frame.f_code.co_qualname
        leaf = frame.f_code.co_name.lower()
        workspace_getter = (
            "workspace" in leaf
            and leaf.startswith(("get", "_get", "alloc", "_alloc", "create", "_create"))
        ) or (leaf == "_get_cache_buf" and "workspace" in str(frame.f_locals.get("name", "")))
        role = "workspace" if workspace_getter else None
        if event == "call":
            owner = frame.f_locals.get("self")
            if ".models." in module and owner is not None and id(owner) not in self.models:
                self.register_model(owner)
                self.models.add(id(owner))
            owner = frame.f_locals.get("self")
            parent = next(reversed(self.frames.values()), None)
            sequence = self.emit(
                "call",
                owner=self.owners.get(id(owner)),
                parent=parent,
                function=function,
                values=self.snapshot(frame.f_locals, role=role),
                state_indices=self.state_indices(frame.f_locals),
            )
            self.frames[id(frame)] = sequence
            if self.nvtx:
                torch.cuda.nvtx.range_push(f"tensor-trace:{sequence}:{function}")
        elif id(frame) in self.frames:
            sequence = self.frames.pop(id(frame))
            self.emit(
                "return",
                call=sequence,
                function=function,
                state_indices=self.state_indices(frame.f_locals),
                values=self.snapshot({"locals": frame.f_locals, "result": arg}, role=role),
            )
            if self.nvtx:
                torch.cuda.nvtx.range_pop()

    def install(self, *, kernels=False):
        if self._installed:
            raise RuntimeError("Tensor trace is already installed")
        self._profile = sys.getprofile()
        if kernels:
            from triton.runtime.jit import JITFunction

            original = JITFunction.run
            self._jit = original

            def launch(fn, *args, **kwargs):
                if not self.active:
                    return original(fn, *args, **kwargs)
                name = fn.fn.__module__ + "." + fn.fn.__qualname__
                sequence = self.emit(
                    "triton_launch",
                    function=name,
                    arguments=self.snapshot({"args": args, "kwargs": kwargs}),
                )
                result = original(fn, *args, **kwargs)
                meta = getattr(result, "metadata", None)
                configuration = {
                    key: getattr(meta, key)
                    for key in ("num_warps", "num_ctas", "shared", "cluster_dims", "maxnreg")
                    if hasattr(meta, key)
                }
                self.emit(
                    "triton_complete",
                    launch=sequence,
                    configuration=self.snapshot(configuration),
                    arguments=self.snapshot({"args": args, "kwargs": kwargs}),
                )
                return result

            self._launch = launch
            JITFunction.run = launch
        self._callback = self._event
        sys.setprofile(self._callback)
        self._installed = True
        return self

    @contextmanager
    def step(self, name, **metadata):
        if self.active:
            raise RuntimeError("Tensor trace steps cannot overlap")
        policy = {
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "inference_mode": torch.is_inference_mode_enabled(),
        }
        self.emit(
            "step_begin", name=name, metadata=self.snapshot(metadata), execution_policy=policy
        )
        self.active = True
        try:
            yield
        finally:
            self.active = False
            self.emit("step_end", name=name)
            self.events.flush()

    def close(self, *, complete=True):
        self.active = False
        if self._installed:
            sys.setprofile(self._profile)
            if self._jit is not None:
                from triton.runtime.jit import JITFunction

                JITFunction.run = self._jit
            self._installed = False
        self.emit(
            "trace_end",
            complete=complete and not self.frames,
            payload_bytes=self.bytes,
            payloads=self.payloads,
            parameter_storages=len(self.parameters),
            pending_frames=len(self.frames),
        )
        self.events.close()
