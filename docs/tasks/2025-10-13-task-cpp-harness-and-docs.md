---
title: "C++ extension harness: multi-file + docs"
created: 2025-10-13
status: completed
owner: Codex agent
branch: main
priority: medium
labels: [task]
deps:
  - src/viterbo/_cpp/__init__.py
  - src/viterbo/_cpp/add_one.cpp
  - tests/test_cpp_extension.py
---

## Summary

Extend the C++ extension scaffold to support multi-file builds and add brief docs for building, profiling, and fallback behaviour.

## Deliverables

- Add an example multi-file extension under `src/viterbo/_cpp/` (headers + .cpp) and load it lazily.
- Document build flags and common errors in `docs/architecture/overview.md`.
- Add a benchmark that exercises the C++ path with a modest workload.

## Acceptance Criteria

- CI smoke remains green (fallback when build unavailable).
- Local build instructions tested in devcontainer.

## Status Log

- 2025-10-15 — Added multi-file affine example with lazy loader and benchmark, documented extension workflow in `docs/architecture/overview.md`; merged to `main`.
