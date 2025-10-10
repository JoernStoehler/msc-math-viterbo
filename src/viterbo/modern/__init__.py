"""Modern, JAX-first surface for symplectic polytope workflows.

The :mod:`viterbo.modern` package provides a clean-room reimagining of the
library that emphasises three themes:

* **Pure mathematics first.** Core geometry and symplectic quantities are
  modelled as JAX-friendly, side-effect-free functions whose signatures lean on
  jaxtyping. Dataclasses and pytrees describe the structured inputs we pass
  between layers.
* **Data pipelining at the edges.** Dataset construction and consumption live in
  thin adapters that translate between tabular data (Polars dataframes) and the
  pure JAX layer. These adapters will remain imperative, but everything they
  call stays functional.
* **Composable building blocks.** Rather than a monolithic API, we prefer small
  modules for distinct quantities—volumes, capacities, spectra, generators—and
  expose explicit padding strategies when batching becomes necessary.

At this stage every public function is a stub that raises
:class:`NotImplementedError`. The module layout and docstrings are intended to
set expectations for future implementations while keeping imports lightweight
and free from historical coupling to the legacy :mod:`viterbo` codebase.
"""

from __future__ import annotations

__all__: tuple[str, ...] = ()
