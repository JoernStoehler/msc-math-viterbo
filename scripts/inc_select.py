#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Incremental test selector — module import graph + file hashes (KISS).

Why this exists
  - Keep local feedback loops fast by running only the tests that plausibly
    depend on files changed since the last run.
  - Do not rely on Git history or coverage contexts; operate directly on the
    working tree with a simple, portable heuristic.

What this script does (in plain terms)
  1) Walk the repo and find all Python files (``**/*.py``), excluding cache/vendor dirs.
  2) Compute a static import graph via AST: edges point from a file to the files it imports
     within the repo (third‑party imports are ignored).
  3) Compute content hashes for all files and determine a set of "dirty" files:
       - added, removed, or content‑changed files are considered dirty
       - dirtiness is then propagated to importers (reverse edges) until fixpoint
       - test files that import any dirty module become dirty, even if the test itself
         did not change
  4) Compose the selection to run as:
       - all dirty test files (emit the file path so pytest runs the file)
       - plus previously failing nodeids from ``.cache/last-junit.xml`` (if available),
         so fixes surface immediately without re‑running whole files
  5) Write the selection to stdout, one entry per line, for consumption by pytest's
     "@argsfile" syntax. If there is nothing to run because there are no Python changes and
     there were no prior failures, print a clear skip message to stderr and return a dedicated
     exit code so the runner can skip invoking pytest entirely.

Guardrails (correctness first)
  - "Plumbing" changes advise a full run: any ``conftest.py``, ``pytest.ini``, pytest settings
    in ``pyproject.toml``, ``uv.lock``, or changes to this selector file.
  - Large or controversial dependency patterns (e.g., heavy dynamic imports, plugin loaders)
    should be handled conservatively by the caller (fall back to full run if in doubt).

Input/Output contract
  - Input: current working tree only. Optional prior failures read from ``.cache/last-junit.xml``.
  - Output: list of test file paths and nodeids on stdout; informational messages on stderr.
  - Exit status: 0 = selection emitted; 2 = skip (no changes/no failures); 3 = advise full run
    (large impact). Any other non‑zero is treated as fallback to full by the caller.

Notes on precision and limits
  - This is intentionally coarse (module‑level). It will over‑select if a module is imported but
    unused along the exercised code path; that bias is safe and inexpensive.
  - Dynamic imports and monkey‑patching are not modeled; if such files change, prefer a full run.
  - The import resolver is a best‑effort heuristic for internal modules; third‑party imports are
    ignored (they are outside the repo and assumed stable in local loops).

Usage
  - Called by the Justfile target ``just test`` (and ``just test-incremental``) roughly as:
        uv run --script scripts/inc_select.py > .cache/impacted_nodeids.txt || sel_status=$?
        if [ -s .cache/impacted_nodeids.txt ] && [ "$sel_status" = "0" ]; then
            pytest @.cache/impacted_nodeids.txt
        elif [ "$sel_status" = "2" ]; then
            echo "Selector: no changes and no prior failures — skipping pytest run."
        else
            pytest  # fallback full run
        fi
  - Any pytest run (impacted or fallback) should use ``--junitxml .cache/last-junit.xml`` so that
    previously failing nodeids are tracked between runs without requiring a coverage run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.util import resolve_name
from pathlib import Path
from typing import NamedTuple
from xml.etree import ElementTree as ET

ROOT = Path.cwd()
CACHE_DIR = ROOT / ".cache"
GRAPH_JSON = CACHE_DIR / "inc_graph.json"
LAST_JUNIT = CACHE_DIR / "last-junit.xml"

SELECTION_THRESHOLD = 0.4

EXCLUDE_DIRS = {".git", ".venv", "node_modules", ".cache", "build", "dist", "site"}

class ExitCodes(NamedTuple):
    """Exit status constants used by the selector."""

    success: int = 0
    skip: int = 2
    advise_full: int = 3


EXIT = ExitCodes()


def _log(msg: str) -> None:
    _stderr(f"[inc] {msg}")


def to_repo_path(path: Path) -> str:
    """Return the repository-relative POSIX path for ``path``."""

    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def normalize_path_str(path_str: str) -> str:
    """Normalise stored path strings to repository-relative POSIX form."""

    path_obj = Path(path_str)
    if path_obj.is_absolute():
        try:
            return path_obj.relative_to(ROOT).as_posix()
        except ValueError:
            return path_obj.as_posix()
    return path_obj.as_posix()


def _stderr(msg: str) -> None:
    sys.stderr.write(msg + "\n")


def is_test_file(p: Path) -> bool:
    """Return True if ``p`` looks like a pytest test module under ``tests/``.

    This intentionally avoids guessing outside the canonical ``tests/**/test_*.py``
    and ``tests/**/*_test.py`` patterns.
    """
    s = p.as_posix()
    if "/tests/" in s or s.startswith("tests/"):
        name = p.name
        return name.startswith("test_") or name.endswith("_test.py")
    return False


def walk_py_files() -> list[Path]:
    """Return all Python files under the repo, excluding cache/vendor dirs."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fname in filenames:
            if fname.endswith(".py"):
                files.append(Path(dirpath) / fname)
    files.sort()
    return files


def sha256_of_file(p: Path) -> str:
    """Compute the SHA‑256 hex digest of a file without loading it all at once."""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(262_144), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class ModuleInfo:
    """Metadata about a Python module in the repository."""

    module: str
    package: str
    is_package: bool


def module_info_for_path(path: Path) -> ModuleInfo:
    """Return dotted module metadata for ``path`` relative to the repo root."""

    rel = path.relative_to(ROOT)
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if not parts:
        return ModuleInfo(module="", package="", is_package=False)

    if parts[-1] == "__init__.py":
        module_parts = parts[:-1]
        is_package = True
    else:
        module_parts = parts[:-1] + [parts[-1][:-3]]
        is_package = False

    module_parts = [p for p in module_parts if p]
    module = ".".join(module_parts)
    if is_package:
        package_parts = module_parts
    else:
        package_parts = module_parts[:-1]
    package = ".".join(package_parts)
    return ModuleInfo(module=module, package=package, is_package=is_package)


def build_module_index(files: list[Path], infos: dict[Path, ModuleInfo]) -> dict[str, Path]:
    """Build a best-effort map from dotted module names to repo paths."""

    idx: dict[str, Path] = {}
    for path in files:
        info = infos[path]
        if info.module:
            idx[info.module] = path
            parts = info.module.split(".")
            if len(parts) > 1 and parts[0] != "tests":
                idx[".".join(parts[1:])] = path
    return idx


def resolve_import(
    info: ModuleInfo, node: ast.AST, idx: dict[str, Path]
) -> list[Path]:
    """Resolve an import node to internal module paths."""

    if isinstance(node, ast.Import):
        return dedupe_paths(idx[alias.name] for alias in node.names if alias.name in idx)

    if not isinstance(node, ast.ImportFrom):
        return []

    level = node.level or 0
    module = node.module or ""

    base_name = module
    if level:
        package = info.module if info.is_package else info.package
        if package:
            try:
                relative = ("." * level) + module
                base_name = resolve_name(relative or ".", package)
            except ValueError:
                base_name = module

    candidates: list[str] = []
    if base_name:
        candidates.append(base_name)

    for alias in node.names:
        if alias.name == "*":
            continue
        dotted = ".".join(part for part in (base_name, alias.name) if part)
        if dotted:
            candidates.append(dotted)

    if not candidates:
        return []

    seen: set[Path] = set()
    resolved: list[Path] = []
    for cand in candidates:
        path = idx.get(cand)
        if path and path not in seen:
            resolved.append(path)
            seen.add(path)
    return resolved


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    """Return ``paths`` without duplicates while preserving order."""

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen:
            unique.append(path)
            seen.add(path)
    return unique


def parse_imports(
    files: list[Path], idx: dict[str, Path], infos: dict[Path, ModuleInfo]
) -> dict[Path, list[Path]]:
    """Parse imports for each file and return a mapping ``file -> deps``.

    Any parse or decode error yields an empty dependency list — failing closed is
    acceptable because changes to such a file will mark it dirty by content hash.
    """
    edges: dict[Path, list[Path]] = {}
    for p in files:
        try:
            src = p.read_text(encoding="utf-8")
            tree = ast.parse(src)
        except (OSError, SyntaxError, ValueError):
            edges[p] = []
            continue
        deps: list[Path] = []
        info = infos[p]
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                deps.extend(resolve_import(info, n, idx))
        edges[p] = dedupe_paths(deps)
    return edges


def load_graph() -> dict:
    """Load the previous graph JSON, or an empty structure if missing/invalid."""
    try:
        data = json.loads(GRAPH_JSON.read_text())
        if not isinstance(data, dict):
            return {"nodes": {}, "edges": {}}
        # ensure shape
        data.setdefault("nodes", {})
        data.setdefault("edges", {})
        return data
    except OSError:
        return {"nodes": {}, "edges": {}}
    except json.JSONDecodeError:
        return {"nodes": {}, "edges": {}}


def save_graph(graph: dict) -> None:
    """Persist the current graph JSON under ``.cache/`` (baseline upgrade)."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        GRAPH_JSON.write_text(json.dumps(graph, indent=2))
    except OSError:
        # Non-fatal: selector remains usable even if persisting fails
        pass


def parse_junit_failures(path: Path) -> set[str]:
    """Return nodeids that last failed from a JUnit XML file (best‑effort)."""
    if not path.exists():
        return set()
    try:
        tree = ET.parse(str(path))
    except (ET.ParseError, OSError, ValueError):
        return set()
    root = tree.getroot()
    failed: set[str] = set()
    for case in root.iter("testcase"):
        name = case.attrib.get("name")
        file_attr = case.attrib.get("file")
        if not name or not file_attr:
            continue
        nodeid = f"{Path(file_attr).as_posix()}::{name}"
        if any(child.tag in {"failure", "error"} for child in case):
            failed.add(nodeid)
    return failed


def persist_snapshot(
    files: list[Path], file_hashes: dict[Path, str], edges: dict[Path, list[Path]]
) -> None:
    """Persist the graph snapshot derived from ``files`` and ``edges``."""

    nodes = {
        to_repo_path(path): {"hash": file_hashes[path], "is_test": is_test_file(path)}
        for path in files
    }
    edges_out = {
        to_repo_path(path): sorted(to_repo_path(dep) for dep in deps)
        for path, deps in edges.items()
    }
    save_graph({"nodes": nodes, "edges": edges_out})


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the selector."""

    parser = argparse.ArgumentParser(
        description="Incremental pytest selector using the module import graph."
    )
    parser.add_argument(
        "--plumbing",
        action="append",
        default=None,
        help=(
            "Additional file suffixes to mark as plumbing (e.g. --plumbing config.yaml). "
            "May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional diagnostics about the resolved selection.",
    )
    return parser.parse_args()


def main() -> int:
    """Compute dirty tests, emit selection, and persist the updated graph.

    Behavior
      - Advises a full run (by emitting no selection) if plumbing changed.
      - Emits file paths of dirty test modules and previously failing nodeids.
      - On "no changes and no prior failures" prints a skip message and writes a
        sentinel so callers can skip invoking pytest.
    """
    prev = load_graph()
    prev_nodes: dict[str, dict] = {}
    for raw_path, meta in prev.get("nodes", {}).items():
        if isinstance(meta, dict):
            prev_nodes[normalize_path_str(raw_path)] = meta
    prev_edges: dict[str, list[str]] = {}
    for raw_src, raw_targets in prev.get("edges", {}).items():
        norm_src = normalize_path_str(raw_src)
        prev_edges.setdefault(norm_src, [])
        if not isinstance(raw_targets, list):
            continue
        for target in raw_targets:
            norm_target = normalize_path_str(target)
            if norm_target not in prev_edges[norm_src]:
                prev_edges[norm_src].append(norm_target)

    args = parse_args()

    files = walk_py_files()
    # compute hashes and classify
    current_hash: dict[Path, str] = {p: sha256_of_file(p) for p in files}
    current_set = {to_repo_path(p) for p in files}

    # Determine changed set by comparing against previous graph hashes.
    added = current_set - set(prev_nodes.keys())
    deleted = set(prev_nodes.keys()) - current_set
    modified = {
        to_repo_path(p)
        for p, h in current_hash.items()
        if prev_nodes.get(to_repo_path(p), {}).get("hash") not in {None, h}
    }
    changed = set(added | deleted | modified)

    # Invalidation: plumbing
    plumbing_patterns = [
        "conftest.py",
        "pytest.ini",
        "pyproject.toml",
        "uv.lock",
        "scripts/inc_select.py",
    ]
    if args.plumbing:
        plumbing_patterns.extend(args.plumbing)
    # Robust plumbing detection against absolute paths
    overlap = {p for p in changed for pat in plumbing_patterns if p.endswith(pat)}
    if len(overlap) > 0:
        _log("plumbing: " + ", ".join(sorted(overlap)))
        infos = {path: module_info_for_path(path) for path in files}
        idx = build_module_index(files, infos)
        edges_cur = parse_imports(files, idx, infos)
        persist_snapshot(files, current_hash, edges_cur)
        _log("plumbing changed; fall back to full run")
        return EXIT.success

    # Build import graph from current files
    infos = {path: module_info_for_path(path) for path in files}
    idx = build_module_index(files, infos)
    edges_cur = parse_imports(files, idx, infos)
    # Build reverse edges; use the union of previous and current edges for safety.
    # If imports moved since the last run, unioning avoids missing dependency paths
    # and under‑selecting. This keeps the bias conservative and the code simple.
    all_edges: dict[str, list[str]] = {}
    # Seed with previous edges
    for src, targets in prev_edges.items():
        all_edges.setdefault(src, [])
        for target in targets:
            if target not in all_edges[src]:
                all_edges[src].append(target)
    # Add current edges
    for path, deps in edges_cur.items():
        src = to_repo_path(path)
        all_edges.setdefault(src, [])
        for dep in deps:
            tgt = to_repo_path(dep)
            if tgt not in all_edges[src]:
                all_edges[src].append(tgt)

    # Build reverse edges
    rev: dict[str, list[str]] = {}
    for k, vs in all_edges.items():
        for t in vs:
            rev.setdefault(t, []).append(k)

    # Initial dirty set: changed files
    dirty: set[str] = set(changed)

    # If nothing changed, but there are prior failures, we still want to run them
    try:
        prior_failures = parse_junit_failures(LAST_JUNIT)
    except OSError:
        prior_failures = set()

    if not dirty and prior_failures:
        _log("no Python changes; running previously failing tests only")
        for nid in sorted(prior_failures):
            sys.stdout.write(nid + "\n")
        persist_snapshot(files, current_hash, edges_cur)
        return EXIT.success

    # propagate to importers (tests or code), and compute distance levels
    queue = list(dirty)
    seen = set(queue)
    dist: dict[str, int] = {n: 0 for n in queue}
    while queue:
        n = queue.pop()
        for up in rev.get(n, []):
            if up not in seen:
                seen.add(up)
                dist[up] = dist.get(n, 0) + 1
                queue.append(up)
    all_dirty = seen

    # derive dirty test files
    dirty_tests: set[str] = set()
    for p_str in all_dirty:
        pp = Path(p_str)
        if is_test_file(pp):
            dirty_tests.add(p_str)

    # tests that changed themselves are definitely dirty
    for p_str in changed:
        if is_test_file(Path(p_str)):
            dirty_tests.add(p_str)

    # previous failures
    prev_fail = prior_failures

    # Gather totals for threshold decisions
    total_tests = sum(1 for p in files if is_test_file(p)) or 1

    # No changes fast-path
    if not dirty_tests and not prev_fail:
        _stderr("[inc] no Python changes and no prior failures; skipping test run")
        # Persist baseline to ensure future diffs work even if none existed before
        persist_snapshot(files, current_hash, edges_cur)
        return EXIT.skip

    # Large-change fallback: if selection would be too large, advise full run
    # Compute selection size in terms of test files only.
    selected_test_files = set(dirty_tests)
    fraction = (len(selected_test_files) / total_tests) if total_tests else 1.0
    if selected_test_files and fraction > SELECTION_THRESHOLD:
        _log(
            "large impact: selected_test_files="
            f"{len(selected_test_files)}/{total_tests} (p={fraction:.2f}) — advise full run"
        )
        persist_snapshot(files, current_hash, edges_cur)
        return EXIT.advise_full

    # emit selection: order test files by relevance and add failing nodeids whose files aren't selected
    selected: list[str] = []
    # Order buckets: changed test files (distance 0), direct importers (distance 1), indirect (>=2)
    changed_tests = [p for p in selected_test_files if p in (added | modified)]
    direct_tests = [
        p for p in selected_test_files if p not in changed_tests and dist.get(p, 99) == 1
    ]
    indirect_tests = [
        p for p in selected_test_files if p not in changed_tests and p not in direct_tests
    ]
    for bucket in (sorted(changed_tests), sorted(direct_tests), sorted(indirect_tests)):
        selected.extend(bucket)

    # Include failing nodeids whose files are not already selected as files
    failing_extra = []
    for nid in sorted(prev_fail):
        file_part = nid.split("::", 1)[0]
        if file_part not in selected_test_files:
            failing_extra.append(nid)

    if selected or failing_extra:
        summary = (
            "module-graph selection: dirty_test_files="
            f"{len(selected_test_files)} prior_failures={len(prev_fail)} "
            f"selected={len(selected) + len(failing_extra)}"
        )
        if selected:
            summary += f" first={selected[0]}"
        _log(summary)
        if args.debug:
            _log(
                "debug: changed="
                + ", ".join(sorted(changed))
                + " | selected="
                + ", ".join(selected)
            )
    # Print selection: test files (ordered) then failing nodeids for non-selected files
    for it in selected:
        sys.stdout.write(it + "\n")
    for nid in failing_extra:
        sys.stdout.write(nid + "\n")

    persist_snapshot(files, current_hash, edges_cur)

    return EXIT.success


if __name__ == "__main__":
    raise SystemExit(main())
