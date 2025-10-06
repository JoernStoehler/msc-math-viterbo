"""Validate that each pytest test declares goal markers and docstrings."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

# Support execution both as a module (python -m scripts.check_test_metadata)
# and as a script (python scripts/check_test_metadata.py).
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


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the metadata check CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Ensure each pytest test has a goal marker (goal_math, goal_code, or "
            "goal_performance) and a descriptive docstring."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["tests"],
        help="File or directory roots to scan (default: tests).",
    )
    args = parser.parse_args(argv)

    issues: list[str] = []
    total_tests = 0
    files_scanned = 0

    for path in iter_test_files(args.paths):
        files_scanned += 1
        for testcase in collect_tests_in_file(path):
            total_tests += 1
            goal_markers = testcase.markers & GOAL_MARKERS
            if not goal_markers:
                issues.append(
                    f"{testcase.path}:{testcase.lineno}: "
                    "missing goal marker (use one of goal_math, goal_code, goal_performance)."
                )
            elif len(goal_markers) > 1:
                joined = ", ".join(sorted(goal_markers))
                issues.append(
                    f"{testcase.path}:{testcase.lineno}: "
                    f"multiple goal markers detected ({joined}); choose exactly one."
                )

            if testcase.docstring is None or not testcase.docstring.strip():
                issues.append(
                    f"{testcase.path}:{testcase.lineno}: "
                    "missing or empty docstring describing the test's intent."
                )

    if issues:
        sys.stderr.write("Found test metadata issues:\n")
        for issue in issues:
            sys.stderr.write(f"{issue}\n")
        sys.stderr.write(
            "Hint: annotate each test with a goal marker (e.g., @pytest.mark.goal_math) "
            "and add a high-level docstring describing the invariant or behaviour.\n"
        )
        return 1

    sys.stdout.write(
        f"Validated {total_tests} tests across {files_scanned} files; all have goal markers and docstrings.\n"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
