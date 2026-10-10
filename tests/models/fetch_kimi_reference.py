"""Explicit, bounded source-only fetch for optional independent K3 CPU tests.

Usage: python tests/models/fetch_kimi_reference.py /tmp/kimi-official-reference

Downloads pinned HF/FLA Python source and their licenses. Never imports downloaded
code, installs packages, fetches weights, or accesses the user's HF credentials.
The HF code remains under its upstream Apache-2.0/Kimi K3 license terms; the FLA
code is MIT. These third-party sources are deliberately not redistributed here.
"""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

HF_REV = "f831ab66814297da540d832a5235f8e904f29d06"
FLA_REV = "14bef3d9ec98ee359c9a9c7860795d91675dd091"
HF_ROOT = f"https://huggingface.co/moonshotai/Kimi-K3/resolve/{HF_REV}"
FLA_ROOT = f"https://raw.githubusercontent.com/fla-org/flash-linear-attention/{FLA_REV}"
FILES = [
    (
        "modeling_kimi_linear.py",
        f"{HF_ROOT}/modeling_kimi_linear.py",
        "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a",
    ),
    (
        "KIMI_LICENSE",
        f"{HF_ROOT}/LICENSE",
        "20c797ce19af0c17de52c6afb144644768a591c521655f5ebf5712c9850f2887",
    ),
    (
        "fla_kda_naive.py",
        f"{FLA_ROOT}/fla/ops/kda/naive.py",
        "60a32285d4b67068ff633b48bbe8ab31028066d24f00d27e12199a88fc73f016",
    ),
    (
        "FLA_LICENSE",
        f"{FLA_ROOT}/LICENSE",
        "41a83c8187efc1e3ccc21909e806a9e52338e69448554d9754706c3d1cd610e7",
    ),
]
MAX_BYTES = 128 * 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=True)
    for name, url, digest in FILES:
        target = args.destination / name
        if target.exists():
            data = target.read_bytes()
        else:
            with urlopen(url, timeout=30) as response:
                data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError(f"Unexpected size/hash for {name}; refusing to use it")
        if not target.exists():
            target.write_bytes(data)
        print(f"Verified {target} ({len(data)} bytes)")
    print(f"KIMI_OFFICIAL_REFERENCE_PATH={args.destination / 'modeling_kimi_linear.py'}")
    print(f"KIMI_FLA_REFERENCE_PATH={args.destination / 'fla_kda_naive.py'}")


if __name__ == "__main__":
    main()
