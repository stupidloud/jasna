"""Print the GPU-stack pins from pyproject.toml as key=value lines.

The overlay build takes torch/TensorRT binaries from the official package but
still needs the matching pure-Python pieces from PyPI (torchgen, the CPU
torchvision sources, the TensorRT bindings), so the versions must agree with
what the release was built from: the ``nvidia`` extra.
"""
from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    pins: dict[str, str] = {}
    for req in data["project"]["optional-dependencies"]["nvidia"]:
        m = re.match(r"^\s*([A-Za-z0-9_.-]+)\s*==\s*([^\s;]+)", req)
        if m:
            pins[m.group(1).lower().replace("-", "_")] = m.group(2)

    torch_version, _, torch_local = pins["torch"].partition("+")
    cuda = re.match(r"cu(\d+)", torch_local)
    if cuda is None:
        raise SystemExit(f"torch pin {pins['torch']!r} carries no cuXXX tag")
    cuda_major = cuda.group(1)[:-1] or cuda.group(1)  # cu130 -> 13

    out = {
        "torch": torch_version,
        "torchvision": pins["torchvision"].partition("+")[0],
        "tensorrt": pins["tensorrt"],
        "tensorrt_bindings": f"tensorrt-cu{cuda_major}-bindings",
        "app_version": data["project"]["version"],
    }
    for key, value in out.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
