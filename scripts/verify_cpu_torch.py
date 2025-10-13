"""Validate torch CPU-only build for CI."""

from __future__ import annotations

import sys

import torch


def main() -> None:
    """Exit with an error if torch includes GPU runtimes."""
    cuda_version = getattr(torch.version, "cuda", None)
    hip_version = getattr(torch.version, "hip", None)

    if cuda_version:
        sys.exit(
            f"CUDA-enabled torch build detected (torch.version.cuda={cuda_version!r}).",
        )

    if hip_version:
        sys.exit(
            f"HIP-enabled torch build detected (torch.version.hip={hip_version!r}).",
        )

    if torch.backends.cuda.is_built():
        sys.exit("Torch reports CUDA backend compiled in; expected CPU-only build.")


if __name__ == "__main__":
    main()
