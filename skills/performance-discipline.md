---
name: performance-discipline
description: Manage performance investigations, benchmarking, and profiling without premature optimization.
last-updated: 2025-10-17
---

# Performance Discipline

## Philosophy

- Start with pure Python/Torch implementations. Introduce C++ or other accelerations only when profiling identifies sustained hotspots.
- Keep math layer APIs pure and device-agnostic; let callers decide device placement.
- Treat performance regressions beyond existing tolerances as escalation triggers.

## Benchmarking Workflow

1. Run `just bench` for smoke-tier benchmarks. Review new results in `.benchmarks/`.
2. Compare against prior runs. Document notable regressions or improvements in the task notes.
3. Store large benchmark artefacts outside the repo (e.g., under `artefacts/`) when necessary; commit only summaries.
4. When gathering reproducible evidence, pin RNG with `torch.Generator` instances to ensure deterministic comparisons.

## Profiling Guidance

- Profile on demand using local tools inside the devcontainer. Notebook explorations are acceptable for diagnostics; ensure raw outputs stay in `artefacts/` or task notes rather than version control.
- Annotate profiling methodology (input sizes, environments) in the task log to help reproduce findings.
- Escalate before adopting experimental toolchains or hardware changes that impact CI or other developers.

## Transitioning to C++ Extensions

- Only create or modify `_cpp/` modules after confirming a Python implementation bottleneck.
- Ensure CPU-only implementations exist before introducing optional CUDA pathways.
- Document build requirements in the associated skill or README if deviations from the standard toolchain arise.

## Related Skills

- `testing-workflow` — validates functionality before benchmarking.
- `coding-standards` — safeguards architectural boundaries when refactoring for performance.
- `collaboration-reporting` — use when summarizing performance findings for maintainers.
