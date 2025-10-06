# Codebase Structure Overview

The Python package has been reorganised into topical subpackages to reflect the main domains of the
project:

- `viterbo.symplectic` provides symplectic form helpers, EHZ capacity implementations, and systolic
  invariants. Capacity algorithms are organised by family under `viterbo.symplectic.capacity`. The
  facet‑normal algorithms live in `viterbo.symplectic.capacity.facet_normals` with `reference.py`,
  `fast.py`, and `subset_utils.py` modules.
- `viterbo.geometry` collects convex‑geometry primitives including half‑space utilities, canonical
  polytope generators, and volume routines. Each domain exposes a readable reference variant and a
  jit‑able fast variant. JAX is the default array runtime. Any SciPy/NumPy interop is centralized in
  small wrappers under `viterbo._wrapped/`.
- `viterbo.optimization` wraps search heuristics and reusable linear-program infrastructure.

The pytest suite mirrors this layout with matching directories under `tests/` so that regression,
performance, and benchmarking coverage tracks the public API surface of each subpackage.
