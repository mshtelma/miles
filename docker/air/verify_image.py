"""Check the pinned Python and CUDA contracts of the GLM AIR image.

The runtime check is CPU-safe so it can run in a Docker build. B300 kernel
execution, image registration and two-node DCP recovery are separate gates.

Args:
  --pins: A requirements file with exact versions to compare to the image.
  --runtime: Check CPU-safe imports, the Python/PyTorch ABI and the JIT toolchain.
  --gpu-imports: On a B300 node, also import CUDA extensions and Megatron.

Example:
  python /opt/miles-air/verify_image.py --pins /opt/miles-air/requirements-runtime.txt --runtime
"""

import argparse
import importlib
import json
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement


def _installed_pins(path: Path) -> dict[str, str]:
    versions = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, expected = line.partition("==")
        if not separator or not name or not expected:
            raise ValueError(f"invalid exact-version requirement: {line}")
        actual = metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"{name}: expected {expected}, installed {actual}")
        versions[name] = actual
    return versions


def _required_dependency_closure(projects: tuple[str, ...]) -> None:
    """Check packages used by the AIR control plane against their metadata."""
    for project in projects:
        for raw in metadata.requires(project) or []:
            requirement = Requirement(raw)
            if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
                continue
            try:
                installed = metadata.version(requirement.name)
            except metadata.PackageNotFoundError as error:
                raise RuntimeError(f"{project} requires missing {requirement.name}") from error
            if installed not in requirement.specifier:
                raise RuntimeError(f"{project} requires {requirement}, installed {installed}")


def _runtime_contract(*, gpu_imports: bool) -> dict[str, str]:
    if sys.version_info[:3] != (3, 12, 3) or Path(sys.prefix) != Path("/opt/venv"):
        raise RuntimeError(f"AIR Python 3.12.3 venv is required: {sys.version}; {sys.prefix}")
    import torch

    if torch.version.cuda != "13.0" or not torch.compiled_with_cxx11_abi():
        raise RuntimeError("PyTorch must use CUDA 13.0 and the C++11 ABI")
    for name in ("fla.ops.cp", "tilelang", "sglang", "ray", "databricks.sdk", "mlflow"):
        importlib.import_module(name)
    flash_attn_varlen_func = importlib.import_module("flash_attn").flash_attn_varlen_func
    if not callable(flash_attn_varlen_func):
        raise RuntimeError("the pinned FlashAttention 2 varlen function is missing")
    _required_dependency_closure(("torch", "sglang", "mlflow-skinny", "databricks-sdk", "grpcio-tools"))
    if gpu_imports:
        if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
            raise RuntimeError("GPU import gate requires a B300 with compute capability 10.3")
        for name in ("flash_mla", "sgl_kernel", "transformer_engine.pytorch", "megatron.core", "mbridge"):
            importlib.import_module(name)
    compiler = shutil.which("nvcc")
    cuda_headers = ("cuda_runtime.h", "cublasLt.h", "nvrtc.h")
    if compiler is None or any(not (Path("/usr/local/cuda/include") / name).is_file() for name in cuda_headers):
        raise RuntimeError("CUDA JIT compiler or required headers are missing")
    nvcc = subprocess.run([compiler, "--version"], capture_output=True, text=True, check=True)
    if re.search(r"release 13\.0,", nvcc.stdout) is None:
        raise RuntimeError("nvcc is not CUDA 13.0")
    if shutil.which("ninja") is None or shutil.which("c++") is None:
        raise RuntimeError("Ninja or the C++ compiler is missing")
    if not Path("/usr/include/python3.12/Python.h").is_file():
        raise RuntimeError("Python development headers are missing for TorchInductor JIT")
    return {"python": sys.version.split()[0], "cuda": torch.version.cuda, "nvcc": compiler}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--runtime", action="store_true")
    parser.add_argument("--gpu-imports", action="store_true")
    args = parser.parse_args(argv)
    if args.gpu_imports and not args.runtime:
        parser.error("--gpu-imports requires --runtime")
    report = {"versions": _installed_pins(args.pins)}
    if args.runtime:
        report.update(_runtime_contract(gpu_imports=args.gpu_imports))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
