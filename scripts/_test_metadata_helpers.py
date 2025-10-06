"""Shared utilities for inspecting pytest test metadata."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

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
) -> Iterator[TestCase]:
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


def iter_test_files(paths: Sequence[str]) -> Iterator[Path]:
    """Yield python test files under the provided paths."""
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(path.rglob("test_*.py"))
        elif path.suffix == ".py":
            yield path


def collect_tests_in_file(path: Path) -> Iterator[TestCase]:
    """Yield test cases discovered in the given python file."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    yield from _collect_tests(tree, path=path)


def iter_test_cases(paths: Sequence[str]) -> Iterator[TestCase]:
    """Iterate over collected test cases discovered within the provided paths."""
    for path in iter_test_files(paths):
        yield from collect_tests_in_file(path)
