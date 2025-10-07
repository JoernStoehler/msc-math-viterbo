#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""
Incremental test selector (module-graph + hashes, KISS).

Outputs on stdout one entry per line for pytest to consume:
- test file paths to run (changed/dirty tests)
- plus nodeids of previously failing tests (from last JUnit), if any

Exit code: always 0. Prints a clear message to stderr and creates
`.cache/impacted_none` when there's nothing to run (no Python changes
and no prior failures).

Guardrails:
- Full run suggested (selector returns empty and no sentinel) if
  plumbing changed: conftest.py, pytest.ini, pyproject.toml (pytest blocks),
  uv.lock, or this script itself.
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
    """Return True if `p` looks like a pytest test module under tests/."""
    s = p.as_posix()
    if "/tests/" in s or s.startswith("tests/"):
        name = p.name
        return name.startswith("test_") or name.endswith("_test.py")
    return False


def walk_py_files() -> list[Path]:
    """Return all Python files under the repo excluding cache/vendor dirs."""
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        # prune excluded dirs
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for f in filenames:
            if f.endswith(".py"):
                files.append(Path(dirpath) / f)
    return files


def sha256_of_file(p: Path) -> str:
    """Compute sha256 hex digest of a file."""
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(262_144), b""):
            h.update(chunk)
    return h.hexdigest()


def build_module_index(files: list[Path]) -> dict[str, Path]:
    """Map plausible dotted names to paths (heuristic, KISS).

    - For path like src/viterbo/a/b.py → dotted names include viterbo.a.b
    - For package __init__.py → viterbo.a
    - Also index tests as dotted-ish (tests.pkg.mod)
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
    """Resolve an import node to repo-internal module paths, best-effort."""
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
    """Parse imports for each file and return edges mapping file->deps."""
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
    """Load graph JSON if present, otherwise an empty structure."""
    if GRAPH_JSON.exists():
        try:
            return json.loads(GRAPH_JSON.read_text())
        except (OSError, json.JSONDecodeError, ValueError):
            return {"nodes": {}, "edges": {}}
    return {"nodes": {}, "edges": {}}


def save_graph(graph: dict) -> None:
    """Persist graph JSON under .cache/."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    GRAPH_JSON.write_text(json.dumps(graph, indent=2))


def parse_junit_failures(path: Path) -> set[str]:
    """Return nodeids that last failed from a JUnit XML file (best-effort)."""
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
    """Entry point: compute dirty tests and emit selection."""
    prev = load_graph()
    prev_nodes: dict[str, dict] = prev.get("nodes", {})
    prev_edges: dict[str, list[str]] = prev.get("edges", {})

    files = walk_py_files()
    # compute hashes and classify
    current_hash: dict[Path, str] = {p: sha256_of_file(p) for p in files}
    current_set = {p.as_posix() for p in files}

    added = current_set - set(prev_nodes.keys())
    deleted = set(prev_nodes.keys()) - current_set
    modified = {p.as_posix() for p, h in current_hash.items() if prev_nodes.get(p.as_posix(), {}).get("hash") not in {None, h}}

    # Invalidation: plumbing
    plumbing_patterns = ["pytest.ini", "pyproject.toml", "uv.lock", "scripts/inc_select.py"]
    if any(Path(p).name == "conftest.py" for p in list(added | deleted | modified)) or any(
        any(pp in p for pp in plumbing_patterns) for p in list(added | deleted | modified)
    ):
        # signal fallback by not writing selection and returning 0; caller will fallback to full
        _stderr("[inc] plumbing changed; advise full run")
        return 0

    # Build import graph from current files
    idx = build_module_index(files)
    edges_cur = parse_imports(files, idx)
    # reverse edges use union of prev and current for safety
    all_edges: dict[str, list[str]] = {}
    for k, v in prev_edges.items():
        all_edges.setdefault(k, [])
        for t in v:
            if t not in all_edges[k]:
                all_edges[k].append(t)
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

    # initial dirty set: added, deleted, modified
    dirty = set(added | deleted | modified)
    # propagate to importers (tests or code)
    queue = list(dirty)
    seen = set(queue)
    while queue:
        n = queue.pop()
        for up in rev.get(n, []):
            if up not in seen:
                seen.add(up)
                queue.append(up)
    all_dirty = seen

    # derive dirty test files
    dirty_tests: set[str] = set()
    for p_str in all_dirty:
        pp = Path(p_str)
        if is_test_file(pp):
            dirty_tests.add(p_str)

    # tests that changed themselves are definitely dirty
    for p_str in (added | modified):
        if is_test_file(Path(p_str)):
            dirty_tests.add(p_str)

    # previous failures
    prev_fail = parse_junit_failures(LAST_JUNIT)

    # No changes fast-path
    if not dirty_tests and not prev_fail:
        _stderr("[inc] no Python changes and no prior failures; skipping test run")
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            (CACHE_DIR / "impacted_none").write_text("skip\n")
        except OSError:
            pass
        # still persist current graph below

    # emit selection: test files + failing nodeids
    selected: set[str] = set(dirty_tests) | set(prev_fail)
    for it in sorted(selected):
        sys.stdout.write(it + "\n")

    # persist current graph (hashes + current edges)
    nodes_out: dict[str, dict] = {}
    for p, h in current_hash.items():
        nodes_out[p.as_posix()] = {
            "hash": h,
            "is_test": is_test_file(p),
        }
    edges_out: dict[str, list[str]] = {k.as_posix(): [t.as_posix() for t in v] for k, v in edges_cur.items()}
    save_graph({"nodes": nodes_out, "edges": edges_out})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
