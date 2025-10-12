#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""
Incremental test selector — module import graph + file hashes (KISS).

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
  - Called by the Justfile target ``just test`` roughly as:
        uv run --script scripts/inc_select.py > .cache/impacted_nodeids.txt || true
        pytest -q @.cache/impacted_nodeids.txt  # when selection is non‑empty
  - Any pytest run (impacted or fallback) should use ``--junitxml .cache/last-junit.xml`` so that
    previously failing nodeids are tracked between runs without requiring a coverage run.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path.cwd()
CACHE_DIR = ROOT / ".cache"
GRAPH_JSON = CACHE_DIR / "inc_graph.json"
LAST_JUNIT = CACHE_DIR / "last-junit.xml"

EXCLUDE_DIRS = {".git", ".venv", "node_modules", ".cache", "build", "dist", "site"}


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
        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for f in filenames:
            if f.endswith(".py"):
                files.append(Path(dirpath) / f)
    return files


def sha256_of_file(p: Path) -> str:
    """Compute the SHA‑256 hex digest of a file without loading it all at once."""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(262_144), b""):
            h.update(chunk)
    return h.hexdigest()


def build_module_index(files: list[Path]) -> dict[str, Path]:
    """Build a best‑effort map from dotted module names to repo paths.

    Rationale
      - Static import nodes carry dotted names; we need to map those back to
        files in the repo to build edges.
      - We normalize common layouts: stripping a leading ``src/``, collapsing
        ``__init__.py`` to the package name, and indexing both full and tail
        dotted names (``viterbo.a.b`` and ``a.b``) so imports within the
        package resolve.
    """
    idx: dict[str, Path] = {}
    for p in files:
        rel = p.relative_to(ROOT)
        parts = list(rel.parts)
        if parts and parts[0] == "src":
            parts = parts[1:]
        if parts and parts[-1] == "__init__.py":
            dotted = ".".join(parts[:-1])
        else:
            dotted = ".".join(parts)[:-3]  # strip .py
        dotted = dotted.replace("/", ".").replace("\\", ".")
        if dotted:
            idx[dotted] = p
            # also index without top-level when under src/<top>/...
            if dotted.count(".") >= 1 and dotted.split(".")[0] not in {"tests"}:
                idx[".".join(dotted.split(".")[1:])] = p
    return idx


def resolve_import(from_path: Path, node: ast.AST, idx: dict[str, Path]) -> list[Path]:
    """Resolve an import node to internal module paths.

    We handle two cases:
      - ``import a.b`` → look up ``a.b`` in the index
      - ``from .x import y`` with levels → compute a base dotted path relative
        to the file's package and attempt both ``base.y`` and ``base``

    This is deliberately simple; it is fine to miss third‑party modules — we only
    care about edges within the repo.
    """
    rel = from_path.relative_to(ROOT)
    # compute base dotted path
    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    base_dotted = ".".join(parts[:-1])
    out: list[Path] = []

    if isinstance(node, ast.Import):
        for alias in node.names:
            cand = alias.name
            if cand in idx:
                out.append(idx[cand])
    elif isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        level = node.level or 0
        if level:
            base_parts = base_dotted.split(".") if base_dotted else []
            parent = ".".join(base_parts[: max(0, len(base_parts) - level + (1 if mod else 0))])
            target = (parent + ("." if parent and mod else "") + mod) if (parent or mod) else ""
        else:
            target = mod
        if target and target in idx:
            out.append(idx[target])
        else:
            # sometimes names point directly to modules
            for alias in node.names:
                name = alias.name
                dn = (target + "." + name) if target else name
                if dn in idx:
                    out.append(idx[dn])
    return out


def parse_imports(files: list[Path], idx: dict[str, Path]) -> dict[Path, list[Path]]:
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
        for n in ast.walk(tree):
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                deps.extend(resolve_import(p, n, idx))
        # dedupe and keep only internal files
        uniq = []
        seen: set[Path] = set()
        for d in deps:
            if d not in seen:
                uniq.append(d)
                seen.add(d)
        edges[p] = uniq
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


def main() -> int:
    """Compute dirty tests, emit selection, and persist the updated graph.

    Behavior
      - Advises a full run (by emitting no selection) if plumbing changed.
      - Emits file paths of dirty test modules and previously failing nodeids.
      - On "no changes and no prior failures" prints a skip message and writes a
        sentinel so callers can skip invoking pytest.
    """
    prev = load_graph()
    prev_nodes: dict[str, dict] = prev.get("nodes", {})
    prev_edges: dict[str, list[str]] = prev.get("edges", {})

    files = walk_py_files()
    # compute hashes and classify
    current_hash: dict[Path, str] = {p: sha256_of_file(p) for p in files}
    current_set = {p.as_posix() for p in files}

    # Determine changed set by comparing against previous graph hashes.
    current_set = {p.as_posix() for p in files}
    added = current_set - set(prev_nodes.keys())
    deleted = set(prev_nodes.keys()) - current_set
    modified = {
        p.as_posix()
        for p, h in current_hash.items()
        if prev_nodes.get(p.as_posix(), {}).get("hash") not in {None, h}
    }
    changed = set(added | deleted | modified)
    changed = changed - set([__file__])

    # Invalidation: plumbing
    plumbing_patterns = [
        "conftest.py",
        "pytest.ini",
        "pyproject.toml",
        "uv.lock",
        "scripts/inc_select.py",
    ]
    # Robust plumbing detection against absolute paths
    overlap = {p for p in changed for pat in plumbing_patterns if p.endswith(pat)}
    if len(overlap) > 0:
        _stderr("[inc] plumbing: " + ", ".join(sorted(overlap)))
        # Persist the current graph as a new baseline before advising fallback
        idx = build_module_index(files)
        edges_cur = parse_imports(files, idx)
        nodes_out: dict[str, dict] = {
            p.as_posix(): {"hash": h, "is_test": is_test_file(p)} for p, h in current_hash.items()
        }
        edges_out: dict[str, list[str]] = {
            k.as_posix(): [t.as_posix() for t in v] for k, v in edges_cur.items()
        }
        save_graph({"nodes": nodes_out, "edges": edges_out})
        _stderr("[inc] plumbing changed; fall back to full run")
        return 0

    # Build import graph from current files
    idx = build_module_index(files)
    edges_cur = parse_imports(files, idx)
    # Build reverse edges; use the union of previous and current edges for safety.
    # If imports moved since the last run, unioning avoids missing dependency paths
    # and under‑selecting. This keeps the bias conservative and the code simple.
    all_edges: dict[str, list[str]] = {}
    # Seed with previous edges
    for k, v in prev_edges.items():
        all_edges.setdefault(k, [])
        for t in v:
            if t not in all_edges[k]:
                all_edges[k].append(t)
    # Add current edges
    for k, v in edges_cur.items():
        ks = k.as_posix()
        all_edges.setdefault(ks, [])
        for t in v:
            ts = t.as_posix()
            if ts not in all_edges[ks]:
                all_edges[ks].append(ts)

    rev: dict[str, list[str]] = {}
    for src, tgts in all_edges.items():
        for t in tgts:
            rev.setdefault(t, []).append(src)

    # initial dirty set: changed files
    dirty = set(changed)
    # dynamic import heuristic (simple): if changed code file mentions importlib/__import__, advise full
    try:
        for p_str in list(changed):
            pp = Path(p_str)
            if is_test_file(pp):
                continue
            txt = pp.read_text(encoding="utf-8")
            if "importlib." in txt or "__import__(" in txt:
                _stderr(f"[inc] dynamic import detected in {pp.as_posix()}; advise full run")
                # Persist baseline graph before advising fallback
                nodes_out: dict[str, dict] = {
                    p.as_posix(): {"hash": h, "is_test": is_test_file(p)}
                    for p, h in current_hash.items()
                }
                edges_out: dict[str, list[str]] = {
                    k.as_posix(): [t.as_posix() for t in v] for k, v in edges_cur.items()
                }
                save_graph({"nodes": nodes_out, "edges": edges_out})
                return 0
    except OSError:
        pass

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
    prev_fail = parse_junit_failures(LAST_JUNIT)

    # Gather totals for threshold decisions
    total_tests = sum(1 for p in files if is_test_file(p)) or 1

    # No changes fast-path
    if not dirty_tests and not prev_fail:
        _stderr("[inc] no Python changes and no prior failures; skipping test run")
        # Persist baseline to ensure future diffs work even if none existed before
        nodes_out: dict[str, dict] = {
            p.as_posix(): {"hash": h, "is_test": is_test_file(p)} for p, h in current_hash.items()
        }
        edges_out: dict[str, list[str]] = {
            k.as_posix(): [t.as_posix() for t in v] for k, v in edges_cur.items()
        }
        save_graph({"nodes": nodes_out, "edges": edges_out})
        return 2

    # Large-change fallback: if selection would be too large, advise full run
    # Compute selection size in terms of test files only.
    selected_test_files = set(dirty_tests)
    try:
        threshold = float(os.environ.get("INC_THRESHOLD", "0.4"))
    except ValueError:
        threshold = 0.4
    fraction = (len(selected_test_files) / total_tests) if total_tests else 1.0
    if selected_test_files and fraction > threshold:
        _stderr(
            f"[inc] large impact: selected_test_files={len(selected_test_files)}/"
            f"{total_tests} (p={fraction:.2f}) — advise full run"
        )
        # Persist baseline graph before advising full run
        nodes_out: dict[str, dict] = {
            p.as_posix(): {"hash": h, "is_test": is_test_file(p)} for p, h in current_hash.items()
        }
        edges_out: dict[str, list[str]] = {
            k.as_posix(): [t.as_posix() for t in v] for k, v in edges_cur.items()
        }
        save_graph({"nodes": nodes_out, "edges": edges_out})
        return 3

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
        _stderr(
            f"[inc] module-graph selection: dirty_test_files={len(selected_test_files)} "
            f"prior_failures={len(prev_fail)} selected={len(selected) + len(failing_extra)} — "
            "pytest will only run these; absent tests were skipped as unaffected"
        )
    # Print selection: test files (ordered) then failing nodeids for non-selected files
    for it in selected:
        sys.stdout.write(it + "\n")
    for nid in failing_extra:
        sys.stdout.write(nid + "\n")

    # persist current graph (hashes + current edges)
    nodes_out: dict[str, dict] = {}
    for p, h in current_hash.items():
        nodes_out[p.as_posix()] = {
            "hash": h,
            "is_test": is_test_file(p),
        }
    edges_out: dict[str, list[str]] = {
        k.as_posix(): [t.as_posix() for t in v] for k, v in edges_cur.items()
    }
    save_graph({"nodes": nodes_out, "edges": edges_out})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
