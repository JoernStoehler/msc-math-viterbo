---
name: performance-discipline
description: Use when measuring, profiling, and addressing bottlenecks in the main algorithm; escalate to C++ only with evidence.
last-updated: 2025-10-18
---

# Performance Discipline

## Instructions
- Start with pure Python/Torch; measure first with smoke benches (`just bench`) and representative inputs.
- Pin RNG with `torch.Generator` for determinism; record environment details and inputs alongside results.
- Profile only after confirming regressions; propose C++/pybind11 kernels when a sustained hotspot is proven.
- Escalate before adding new extensions or CUDA paths; include evidence and acceptance thresholds in the task.

## Philosophy

- Measure first with representative inputs; optimize second. Introduce C++ only when profiling identifies sustained hotspots.
- Treat performance regressions beyond existing tolerances as escalation triggers.
- Preemptive coding conventions live in `good-code-loop` and module skills (e.g., `math-layer`).

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

- `testing-and-ci` — validates functionality before benchmarking.
- `good-code-loop` — safeguards architectural boundaries when refactoring for performance.
- `collaboration-reporting` — use when summarizing performance findings for maintainers.
