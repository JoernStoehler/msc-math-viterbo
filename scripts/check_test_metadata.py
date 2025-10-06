"""Validate that each pytest test declares goal markers and docstrings."""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

GOAL_MARKERS = {"goal_math", "goal_code", "goal_performance"}


@dataclass
class TestCase:
    """Representation of a collected test function."""

    path: Path
    name: str
    lineno: int
    markers: set[str]
    docstring: str | None


def _marker_from_decorator(decorator: ast.AST) -> str | None:
    """Extract the pytest marker name if the decorator is a goal marker."""
    target = decorator
    if isinstance(target, ast.Call):
        target = target.func

    if isinstance(target, ast.Attribute):
        if isinstance(target.value, ast.Attribute):
            if (
                isinstance(target.value.value, ast.Name)
                and target.value.value.id == "pytest"
                and target.value.attr == "mark"
            ):
                if target.attr in GOAL_MARKERS:
                    return target.attr
    return None


def _markers_from_decorators(decorators: Sequence[ast.AST]) -> set[str]:
    markers: set[str] = set()
    for decorator in decorators:
        marker = _marker_from_decorator(decorator)
        if marker is not None:
            markers.add(marker)
    return markers


def _collect_tests(
    node: ast.AST,
    *,
    path: Path,
    parent_markers: Iterable[str] | None = None,
) -> Iterable[TestCase]:
    inherited = set(parent_markers or ())

    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name.startswith("test_"):
            markers = inherited | _markers_from_decorators(node.decorator_list)
            docstring = ast.get_docstring(node, clean=False)
            yield TestCase(
                path=path,
                name=node.name,
                lineno=node.lineno,
                markers=markers,
                docstring=docstring,
            )
        return

    if isinstance(node, ast.ClassDef):
        markers = inherited | _markers_from_decorators(node.decorator_list)
        for child in node.body:
            yield from _collect_tests(child, path=path, parent_markers=markers)
        return

    for child in getattr(node, "body", []):
        yield from _collect_tests(child, path=path, parent_markers=inherited)


def _iter_test_files(paths: Sequence[str]) -> Iterable[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(path.rglob("test_*.py"))
        elif path.suffix == ".py":
            yield path


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

    for path in _iter_test_files(args.paths):
        files_scanned += 1
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for testcase in _collect_tests(tree, path=path):
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
