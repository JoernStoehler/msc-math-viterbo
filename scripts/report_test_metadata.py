"""Report pytest test metadata for quick inspection or scripting."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

try:
    from scripts._test_metadata_helpers import (
        GOAL_MARKERS,
        collect_tests_in_file,
        iter_test_files,
    )
except ModuleNotFoundError:  # pragma: no cover - import fallback for direct execution
    import pathlib
    import sys as _sys

    _sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from scripts._test_metadata_helpers import (  # type: ignore[redefined-builtin]
        GOAL_MARKERS,
        collect_tests_in_file,
        iter_test_files,
    )


def _docstring_summary(docstring: str | None) -> str:
    """Extract a single-line summary from a docstring."""
    if docstring is None:
        return "(missing docstring)"

    stripped = docstring.strip()
    if not stripped:
        return "(missing docstring)"

    first_line = stripped.splitlines()[0].strip()
    return first_line if first_line else "(missing docstring)"


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the metadata reporter CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Emit a stable summary of pytest tests, including their goal markers and docstrings."
        )
    )
    parser.add_argument(
        "--marker",
        choices=sorted(GOAL_MARKERS),
        action="append",
        dest="markers",
        help="Filter to tests carrying the specified goal marker (repeatable).",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["tests"],
        help="File or directory roots to scan (default: tests).",
    )
    args = parser.parse_args(argv)

    requested_markers = set(args.markers or [])

    entries: list[str] = []
    for path in iter_test_files(args.paths):
        for testcase in collect_tests_in_file(path):
            goal_markers = sorted(testcase.markers & GOAL_MARKERS)
            if requested_markers and not (requested_markers & set(goal_markers)):
                continue

            marker_display = "missing"
            if goal_markers:
                marker_display = ",".join(goal_markers)

            summary = _docstring_summary(testcase.docstring)
            entries.append(f"{testcase.path}::{testcase.name} [{marker_display}] - {summary}")

    import sys

    entries.sort()
    sys.stdout.write(f"LINES:{len(entries)}\n")
    if entries:
        sys.stdout.write("\n".join(entries) + "\n")

    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    raise SystemExit(main())
