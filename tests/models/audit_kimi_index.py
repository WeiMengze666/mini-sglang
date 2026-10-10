"""Metadata-only key coverage audit; neither downloads nor opens weight shards."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from minisgl.models.config import ModelConfig
from minisgl.models.kimi_config import KimiK3Config
from minisgl.models.kimi_k3 import KimiK3LinearForCausalLM
from minisgl.models.kimi_weight import normalize_kimi_name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("index", type=Path)
    args = parser.parse_args()
    config = ModelConfig.from_hf(KimiK3Config(**json.loads(args.config.read_text())))
    # This audit compares logical dequantized names, independent of storage.
    config.kimi.quantization_config = None
    weights = json.loads(args.index.read_text())["weight_map"]
    with torch.device("meta"):
        expected = set(KimiK3LinearForCausalLM(config).state_dict())
    actual = set()
    for raw in weights:
        name = normalize_kimi_name(raw, config.num_layers)
        if name is None or name.endswith(".weight_scale"):
            continue
        if name.endswith(".weight_packed"):
            scale = raw.removesuffix(".weight_packed") + ".weight_scale"
            if scale not in weights:
                raise ValueError(f"Missing scale: {scale}")
            name = name.removesuffix(".weight_packed") + ".weight"
        if name in actual:
            raise ValueError(f"Duplicate normalized key: {name}")
        actual.add(name)
    report = {
        "raw_keys": len(weights),
        "runtime_keys": len(expected),
        "mapped_keys": len(actual),
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "index_sha256": hashlib.sha256(args.index.read_bytes()).hexdigest(),
    }
    print(json.dumps(report, indent=2))
    if report["missing"] or report["unexpected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
