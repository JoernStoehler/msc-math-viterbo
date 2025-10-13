---
title: "C++ extension harness: multi-file + docs"
created: 2025-10-13
status: draft
owner: TBD
branch: task/cpp-harness-docs
priority: medium
labels: [task]
---

## Summary

Extend the C++ extension scaffold to support multi-file builds and add brief docs for building, profiling, and fallback behavior.

## Deliverables

- Add an example multi-file extension under `src/viterbo/_cpp/` (headers + .cpp)
- Document build flags and common errors in `docs/README.md`
- Add a benchmark that exercises the C++ path with a modest workload

## Acceptance Criteria

- CI smoke remains green (fallback when build unavailable)
- Local build instructions tested in devcontainer

