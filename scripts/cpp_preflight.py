"""Compile a minimal Torch C++ extension to validate the toolchain."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

import torch
from torch.utils.cpp_extension import load

_MODULE_NAME = "viterbo_cpp_preflight"
_CPP_SOURCE = dedent(
    """
    #include <torch/extension.h>

    torch::Tensor preflight_double(torch::Tensor x) {
      return x * 2;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
      m.def("double_values", &preflight_double, "Double tensor values");
    }
    """
)


def _ensure_compiler_env() -> None:
    """Force Ninja usage and enable verbose logs for extension builds."""
    os.environ.setdefault("USE_NINJA", "1")
    os.environ.setdefault("VITERBO_CPP_VERBOSE", "1")
    if "CXX" not in os.environ and "CC" in os.environ:
        # Mirror CC overrides for C++ builds when Torch expects CXX.
        os.environ["CXX"] = os.environ["CC"]


def _build_extension() -> None:
    """Compile the preflight extension and exercise the exported function."""
    _ensure_compiler_env()
    verbose = os.getenv("VITERBO_CPP_VERBOSE", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        source_path = Path(tmpdir) / "preflight.cpp"
        source_path.write_text(_CPP_SOURCE)
        sys.stderr.write(f"[cpp-preflight] wrote source to {source_path}\n")
        extension = load(
            name=_MODULE_NAME,
            sources=[str(source_path)],
            extra_cflags=["-O0"],
            verbose=verbose,
        )

    sample = torch.arange(3, dtype=torch.float64)
    result = extension.double_values(sample)
    if not torch.allclose(result, sample * 2):
        raise SystemExit(
            "[cpp-preflight] extension returned unexpected result; "
            "toolchain output may be corrupted.",
        )
    sys.stderr.write("[cpp-preflight] build successful; extension output validated.\n")


def main() -> None:
    """Entry point used by CI."""
    try:
        _build_extension()
    except Exception as exc:  # pragma: no cover - fail fast helper
        raise SystemExit(f"[cpp-preflight] failed: {exc}") from exc


if __name__ == "__main__":
    main()
