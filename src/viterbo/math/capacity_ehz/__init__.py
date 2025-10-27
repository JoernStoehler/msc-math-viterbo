"""EHZ capacity algorithms and minimal-action cycles.

Modules
- ``algorithms`` — capacity solvers: ``capacity_ehz_algorithm1``,
  ``capacity_ehz_algorithm2``, ``capacity_ehz_primal_dual``.
- ``cycle`` — minimal-action front-end.
- ``lagrangian_product`` — vertex-contact discrete search (≤3 bounces) on
  Lagrangian products ``K × T`` with planar factors.
- ``ratios`` — derived quantities (e.g., systolic ratio).
- ``stubs`` — planned/experimental solvers including the Chaidez–Hutchings
  oriented-edge spectrum in ``R^4`` and the Haim–Kislev programme.

Conventions
- Symplectic ambient dimension is even (``d = 2n``). APIs are Torch-first and
  preserve dtype/device. 4D paths support certain product structures explicitly.

Import concrete functionality from submodules; no re-export indirection.
"""

from __future__ import annotations
