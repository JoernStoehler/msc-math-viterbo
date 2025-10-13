---
title: "CI: enforce CPU-only Torch wheels via uv"
created: 2025-10-13
status: in-review
owner: Codex agent
branch: task/ci-cpu-torch-index
priority: medium
labels: [task]
deps:
  - Justfile
  - .github/workflows/ci.yml
---

## Summary

Harden the CI installation so Torch CPU wheels are always used (no accidental CUDA pulls), while preserving GPU wheels locally.

## Deliverables

- Add a per-environment index override (e.g., `PIP_INDEX_URL`) scoped to CI jobs.
- Consider uv project metadata or scripts to prevent accidental Torch upgrades that pull CUDA.
- Document the policy in `docs/architecture/overview.md`.

## Acceptance Criteria

- CI installation does not fetch CUDA meta-packages.
- Local GPU installs remain unaffected.

## Status Log

- 2025-10-14 — Rebased CI on CPU-only Torch index via env overrides, added guard script, documented policy, and validated `just ci-cpu` under the constrained indexes.
