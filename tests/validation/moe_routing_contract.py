"""Inspect vendor routing ABIs on CPU; this does not validate GPU arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch

    # This standalone process imports GPU SDK host interfaces without querying
    # hardware. Never apply this override inside a serving process.
    torch.cuda.is_available = lambda: False
    torch.cuda.device_count = lambda: 0

    def no_cuda(*a, **kw):
        raise AssertionError("CPU ABI audit must not initialize CUDA")

    torch.cuda._lazy_init = no_cuda
    import sglang.srt.layers.quantization  # noqa: F401
    from flashinfer.fused_moe import core
    from sglang.srt.layers.moe.moe_runner import flashinfer_trtllm as native
    from sglang.srt.layers.moe.topk import PackedTopKOutput, StandardTopKOutput

    ids = torch.arange(8, dtype=torch.int32).reshape(2, 4) + 32
    weights = torch.linspace(0.101, 0.394, 16).reshape(2, 8)[:, ::2]
    rows = []
    for dtype in (torch.float32, torch.bfloat16):
        source = weights.to(dtype)
        routing = native._get_routing_for_flashinfer_routed(StandardTopKOutput(source, ids, None))
        routed_ids, routed_weights, mode = core._split_precomputed_routing(routing)
        assert routed_weights.dtype == dtype
        torch.testing.assert_close(routed_weights, source, rtol=0, atol=0)
        torch.testing.assert_close(routed_ids, ids, rtol=0, atol=0)
        rows.append(
            {
                "format": "unpacked",
                "weight_dtype": str(routed_weights.dtype),
                "weight_shape": list(routed_weights.shape),
                "mode": str(mode),
            }
        )
    rounded = weights.to(torch.bfloat16).contiguous()
    bits = rounded.view(torch.int16).to(torch.int32) & 0xFFFF
    packed = (ids << 16) | bits
    routing = native._get_routing_for_flashinfer_routed(PackedTopKOutput(packed, None))
    packed_ids, routed_weights, mode = core._split_precomputed_routing(routing)
    assert packed_ids is packed and routed_weights is None
    torch.testing.assert_close((packed_ids >> 16).to(torch.int32), ids, rtol=0, atol=0)
    decoded = (packed_ids & 0xFFFF).to(torch.int16).view(torch.bfloat16)
    torch.testing.assert_close(decoded, rounded, rtol=0, atol=0)
    assert not torch.equal(weights, decoded.float())
    rows.append(
        {
            "format": "packed",
            "weight_dtype": "torch.bfloat16 (low16 bits)",
            "shape": list(packed.shape),
            "mode": str(mode),
        }
    )
    native_dir = Path(native.__file__).resolve().parent
    commit = subprocess.run(
        ["git", "-C", str(native_dir), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    sources = [Path(native.__file__), Path(core.__file__)]
    report = {
        "scope": __doc__,
        "passed": True,
        "gpu_arithmetic_checked": False,
        "validation_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_commit": commit.stdout.strip() if commit.returncode == 0 else None,
        "torch": torch.__version__,
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "rows": rows,
        "example_max_weight_rounding": float((weights - decoded.float()).abs().max()),
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limitation": "Host forwarding and bit layout only. Actual native fused-path coverage, expert finalization and end-to-end quality require GPU validation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
