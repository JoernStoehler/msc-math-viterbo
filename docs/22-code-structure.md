# Codebase Structure Overview

The Python package has been reorganised into topical subpackages to reflect the
main domains of the project:

- `viterbo.symplectic` provides symplectic form helpers, EHZ capacity
  implementations, and systolic invariants. The reference and optimised facet
  normal algorithms now live under
  `viterbo.symplectic.capacity_algorithms`.
- `viterbo.geometry` collects convex-geometry primitives including half-space
  utilities, canonical polytope generators, and volume routines.
- `viterbo.optimization` wraps search heuristics and reusable linear-program
  infrastructure.
- `viterbo.examples` exposes small smoke-test entry points that are referenced
  from the documentation and tutorials.

The pytest suite mirrors this layout with matching directories under `tests/`
so that regression, performance, and benchmarking coverage tracks the public
API surface of each subpackage.
