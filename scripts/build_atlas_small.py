"""Command-line harness for building atlas datasets and benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

from viterbo.atlas_build import build_atlas_dataset, default_plan


def parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments for the atlas dataset build."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=["tiny", "small"],
        default="tiny",
        help="Dataset preset to build (default: tiny).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artefacts/datasets"),
        help="Base directory for dataset artefacts.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("artefacts/benchmarks"),
        help="Directory for benchmark timing logs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing manifest/dataset if available.",
    )
    parser.add_argument(
        "--limit-generator",
        action="append",
        dest="generators",
        help="Optional generator name to process; repeat to select multiple.",
    )
    parser.add_argument(
        "--seed-offset",
        type=int,
        default=0,
        help="Offset added to generator base seeds for reproducibility sweeps.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for building atlas datasets from the command line."""
    args = parse_args()
    plan = default_plan()
    build_atlas_dataset(
        args.preset,
        plan=plan,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        resume=args.resume,
        limit_generators=args.generators,
        seed_offset=args.seed_offset,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
