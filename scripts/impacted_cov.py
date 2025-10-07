#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# # No extra dependencies required; uses stdlib only.
# ///
"""Select impacted pytest nodeids from a coverage contexts map.

Usage (prints nodeids to stdout; metrics to stderr when verbose):
  uv run --script scripts/impacted_cov.py --map .cache/coverage.json \
      > .cache/impacted_nodeids.txt || true

Exit codes:
  0 = impacted nodeids were printed to stdout
  2 = fallback requested (no map, invalidation, parse error, or empty/large selection)

Notes:
  - Build the coverage map on main/nightly with per-test contexts enabled, e.g.:
      uv run pytest -q --cov=src/viterbo --cov-context=test
      uv run coverage json -o .cache/coverage.json --show-contexts
  - During PR runs, this script reads the JSON and selects only the impacted tests.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


@dataclass
class SelectorMetrics:
    """Lightweight container for selector metrics."""

    impacted_count: int
    total_in_map: int
    fraction: float
    millis: int
    reason: str | None = None


@dataclass
class LastStatus:
    """Prior test statuses parsed from a JUnit XML report."""

    passed: set[str]
    failed: set[str]
    skipped: set[str]


def _stderr(msg: str) -> None:
    """Write a line to stderr (without using print to satisfy lint)."""
    sys.stderr.write(msg + "\n")


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a shell command and capture text output without raising on error."""
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def _posix(p: str | Path) -> str:
    """Normalize a path to POSIX-style separators."""
    return Path(p).as_posix()


def _write_last_selected(path: Path, nodeids: set[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(sorted(nodeids)) + "\n")
    except OSError:
        pass


def _path_suffix_match(a: str, b: str) -> bool:
    """Return True if path `a` ends with `b` on component boundaries."""
    ap = Path(a).parts
    bp = Path(b).parts
    if len(bp) > len(ap):
        return False
    return ap[-len(bp) :] == bp


def _iter_changed_files(base: str) -> set[str]:
    """Return changed Python file paths versus `base`. Includes A/C/M/R."""
    cp = _run(["git", "diff", "--name-only", "--diff-filter=ACMR", base, "--", "*.py"])
    if cp.returncode != 0:
        return set()
    return {line.strip() for line in cp.stdout.splitlines() if line.strip()}


def _parse_changed_hunks(base: str) -> dict[str, set[int]]:
    """Parse changed line numbers per file from unified diff (-U0)."""
    cp = _run(["git", "diff", "-U0", base, "--", "*.py"])
    if cp.returncode != 0:
        return {}

    hunks: dict[str, set[int]] = {}
    cur_file: str | None = None
    pat_file = re.compile(r"^\+\+\+ b/(.+)$")
    pat_hunk = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    for line in cp.stdout.splitlines():
        m = pat_file.match(line)
        if m:
            cur_file = m.group(1)
            continue
        m = pat_hunk.match(line)
        if m and cur_file:
            start = int(m.group(1))
            count = int(m.group(2) or "1")
            s = hunks.setdefault(cur_file, set())
            s.update(range(start, start + count))
    # Normalize keys to posix paths
    return {_posix(k): v for k, v in hunks.items()}


def _load_coverage_json(path: Path) -> dict:
    """Load coverage JSON; return empty dict on any error or shape mismatch."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}
    # Basic shape check
    if not isinstance(data, dict) or "files" not in data:
        return {}
    return data


def _collect_total_contexts(files_meta: dict) -> set[str]:
    """Collect the set of all contexts (test nodeids) present in the map."""
    total: set[str] = set()
    for meta in files_meta.values():
        contexts = meta.get("contexts")
        if not contexts:
            continue
        if isinstance(contexts, dict):
            # Heuristics: detect mapping shape
            # Case 1: line(str) -> list[str] contexts
            keys = list(contexts.keys())
            if keys and isinstance(keys[0], str) and keys[0].isdigit():
                for ctxs in contexts.values():
                    if isinstance(ctxs, list):
                        for c in ctxs:
                            # Normalize contexts like 'nodeid|run' → 'nodeid'
                            total.add(str(c).split("|", 1)[0])
            else:
                # Case 2: context(str) -> list[int] lines
                for ctx, _lines in contexts.items():
                    total.add(str(ctx).split("|", 1)[0])
        # Else: unknown shape → ignore
    return total


def _select_nodeids(
    files_meta: dict,
    changed_hunks: dict[str, set[int]],
    strict_paths: bool,
) -> set[str]:
    """Select nodeids for changed lines using suffix-aware path matching."""
    nodeids: set[str] = set()
    changed_keys = {_posix(k) for k in changed_hunks.keys()}

    for cov_path, meta in files_meta.items():
        cov_posix = _posix(cov_path)
        # Find matching changed file path
        changed_match: str | None = None
        for ch in changed_keys:
            if (cov_posix == ch) or (not strict_paths and (_path_suffix_match(cov_posix, ch) or _path_suffix_match(ch, cov_posix))):
                changed_match = ch
                break
        if not changed_match:
            continue

        contexts = meta.get("contexts")
        if not contexts or not isinstance(contexts, dict):
            continue

        # Two possible shapes: line->contexts OR context->lines
        sample_keys = list(contexts.keys())
        if sample_keys and isinstance(sample_keys[0], str) and sample_keys[0].isdigit():
            # line(str) -> list[str]
            for ln_str, ctxs in contexts.items():
                try:
                    ln = int(ln_str)
                except (TypeError, ValueError):
                    continue
                if ln in changed_hunks.get(changed_match, set()):
                    if isinstance(ctxs, list):
                        for c in ctxs:
                            nodeids.add(str(c).split("|", 1)[0])
        else:
            # context(str) -> list[int]
            for ctx, lines in contexts.items():
                if not isinstance(lines, (list, tuple)):
                    continue
                for ln in lines:
                    try:
                        lni = int(ln)
                    except (TypeError, ValueError):
                        continue
                    if lni in changed_hunks.get(changed_match, set()):
                        nodeids.add(str(ctx).split("|", 1)[0])
                        break

    return nodeids


def _parse_junit(path: Path) -> LastStatus | None:
    """Parse pytest JUnit XML into nodeid→status sets.

    Construct nodeids as "file::name" when the `file` attribute is present. This
    aligns with pytest nodeids and coverage contexts. Falls back to classname.
    """
    if not path.exists():
        return None
    try:
        tree = ET.parse(str(path))
    except (ET.ParseError, OSError, ValueError):
        return None
    root = tree.getroot()
    passed: set[str] = set()
    failed: set[str] = set()
    skipped: set[str] = set()

    for case in root.iter("testcase"):
        name = case.attrib.get("name")
        file_attr = case.attrib.get("file")
        if not name:
            continue
        if file_attr:
            nodeid = f"{_posix(file_attr)}::{name}"
        else:
            classname = case.attrib.get("classname") or ""
            path_guess = _posix(Path(classname.replace(".", "/") + ".py")) if classname else ""
            if not path_guess:
                continue
            nodeid = f"{path_guess}::{name}"
        nodeid = nodeid.split("|", 1)[0]

        has_fail = any(child.tag in {"failure", "error"} for child in case)
        has_skip = any(child.tag == "skipped" for child in case)
        if has_skip and not has_fail:
            skipped.add(nodeid)
        elif has_fail:
            failed.add(nodeid)
        else:
            passed.add(nodeid)

    return LastStatus(passed=passed, failed=failed, skipped=skipped)


def _should_invalidate(changed_files: Iterable[str]) -> tuple[bool, str | None]:
    """Return (True, reason) if changes require falling back to full run."""
    risky_patterns = [
        re.compile(r"(^|/)conftest\.py$"),
        re.compile(r"^tests/"),
        re.compile(r"^pyproject\.toml$"),
        re.compile(r"^pytest\.ini$"),
        re.compile(r"^uv\.lock$"),
        re.compile(r"^src/viterbo/__init__\.py$"),
        re.compile(r"^scripts/impacted_cov\.py$"),
    ]
    for f in changed_files:
        posix = _posix(f)
        for pat in risky_patterns:
            if pat.search(posix):
                return True, f"invalidated_by:{posix}"
    return False, None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: emit impacted nodeids or signal fallback (exit 2)."""
    parser = argparse.ArgumentParser(description="Select impacted pytest nodeids from coverage JSON contexts map.")
    parser.add_argument("--map", dest="map_path", default=os.environ.get("IMPACTED_MAP", ".cache/coverage.json"), help="Path to coverage JSON with contexts")
    parser.add_argument("--threshold", type=float, default=float(os.environ.get("IMPACTED_THRESHOLD", 0.4)), help="Max impacted fraction before falling back to full (default: 0.4)")
    parser.add_argument("--strict-paths", action="store_true", help="Require exact path equality (no suffix matching)")
    parser.add_argument("--verbose", action="store_true", default=os.environ.get("IMPACTED_VERBOSE", "0") not in {"", "0", "false", "no"}, help="Emit selector metrics to stderr")
    parser.add_argument("--junit", default=os.environ.get("IMPACTED_LAST_JUNIT", ".cache/last-junit.xml"), help="Path to last JUnit XML for status-aware selection")
    parser.add_argument("--skip-prev-fail-unaffected", action="store_true", default=os.environ.get("IMPACTED_SKIP_PREV_FAIL_UNAFFECTED", "0") not in {"", "0", "false", "no"}, help="Skip previously failing tests when unaffected by the diff")

    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    # Determine diff base strictly from last coverage run; fallback to origin/main if missing.
    coverage_base_file = Path(".cache/coverage_base.txt")
    if coverage_base_file.exists():
        try:
            base_ref = coverage_base_file.read_text().strip().splitlines()[0].strip() or "origin/main"
        except OSError:
            base_ref = "origin/main"
    else:
        base_ref = "origin/main"

    changed_files = _iter_changed_files(base_ref)
    changed_hunks = _parse_changed_hunks(base_ref)

    # Parse last JUnit early for no-diff fallback behaviour
    last = _parse_junit(Path(args.junit))
    prev_pass = last.passed if last else set()
    prev_fail = last.failed if last else set()
    prev_skip = last.skipped if last else set()

    # If no Python changes, re-run only previously failing tests (if any); else fallback.
    if not changed_files or not changed_hunks:
        if prev_fail:
            for nid in sorted(prev_fail):
                sys.stdout.write(nid + "\n")
            _write_last_selected(Path(".cache/last-selected.txt"), set(prev_fail))
            return 0
        if args.verbose:
            _stderr("[impacted] no Python changes and no prior failures; fallback to full")
        return 2

    inv, reason = _should_invalidate(changed_files)
    if inv:
        if args.verbose:
            _stderr(f"[impacted] invalidation triggered: {reason}")
        return 2

    cov_json = Path(args.map_path)
    if not cov_json.exists():
        if args.verbose:
            _stderr(f"[impacted] coverage map missing: {cov_json}")
        return 2

    data = _load_coverage_json(cov_json)
    files_meta = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files_meta, dict):
        if args.verbose:
            _stderr("[impacted] invalid coverage JSON format; fallback")
        return 2

    total_contexts = _collect_total_contexts(files_meta)
    impacted = _select_nodeids(files_meta, changed_hunks, strict_paths=args.strict_paths)

    # Post-filter obvious non-nodeid contexts (best-effort): require pattern '::'
    impacted = {nid for nid in impacted if "::" in nid}

    total_in_map = max(1, len(total_contexts))
    impacted_fraction = (len(impacted) / total_in_map) if total_in_map else 1.0

    # Augment selection with last failing tests

    selected = set(impacted)
    # Include previously failing tests by default (unless skipping unaffected failures)
    if args.skip_prev_fail_unaffected:
        selected.update(prev_fail & impacted)
    else:
        selected.update(prev_fail)

    unaffected = total_contexts - impacted
    rerun_impacted_pass = impacted & prev_pass
    rerun_impacted_fail = impacted & prev_fail
    rerun_prev_fail_unaffected = (prev_fail & unaffected) if not args.skip_prev_fail_unaffected else set()
    skip_unaffected_pass = unaffected & prev_pass
    unknown_status_impacted = impacted - (prev_pass | prev_fail | prev_skip)

    t1 = time.perf_counter()
    metrics = SelectorMetrics(
        impacted_count=len(impacted),
        total_in_map=total_in_map,
        fraction=impacted_fraction,
        millis=int((t1 - t0) * 1000),
        reason=None if len(impacted) > 0 else "empty_selection",
    )

    if args.verbose:
        _stderr(
            f"[impacted] impacted={metrics.impacted_count} total={metrics.total_in_map} "
            f"p={metrics.fraction:.3f} time_ms={metrics.millis}"
            + (f" reason={metrics.reason}" if metrics.reason else "")
        )
        _stderr(
            "[impacted] selection: "
            f"rerun_impacted_pass={len(rerun_impacted_pass)} "
            f"rerun_impacted_fail={len(rerun_impacted_fail)} "
            f"rerun_prev_fail_unaffected={len(rerun_prev_fail_unaffected)} "
            f"skip_unaffected_pass={len(skip_unaffected_pass)} "
            f"unknown_impacted={len(unknown_status_impacted)}"
        )

    if len(impacted) == 0 or impacted_fraction > args.threshold:
        # Signal fallback to full run
        return 2

    # Emit nodeids to stdout, sorted for determinism
    for nid in sorted(selected):
        sys.stdout.write(nid + "\n")

    # Persist last selection for optional diffing by users/tools
    _write_last_selected(Path(".cache/last-selected.txt"), selected)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
