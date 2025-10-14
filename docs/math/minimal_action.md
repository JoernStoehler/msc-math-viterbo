---
title: minimal_action — EHZ capacities and cycles
---

# `viterbo.math.minimal_action`

Capacity solvers and minimal-action diagnostics focused on Lagrangian products
and small planar test cases. Production code currently targets 2D inputs while
stubs document the intended 4D+ generalisations.

## Implemented helpers

- `minimal_action_cycle_lagrangian_product(vertices_q, normals_p, offsets_p, *, max_bounces=3)`
  - Vertex-contact Minkowski billiard for planar Lagrangian products `K × T`.
    Enumerates all two- and three-bounce candidates as guaranteed by Rudolf
    (2022), enforcing the strong reflection rule at each bounce and returning
    both the minimal action and the stitched orbit.
- `minimal_action_cycle(vertices, normals, offsets)`
  - Simpler planar orbit builder used by early smoke tests. Keeps the polygon
    order and reports the polygon itself as the cycle; useful for deterministic
    dataset completion such as AtlasTiny.
- `capacity_ehz_algorithm1(normals, offsets)` / `capacity_ehz_algorithm2(vertices)` /
  `capacity_ehz_primal_dual(vertices, normals, offsets)`
  - Placeholder EHZ capacity routines for planar bodies. They provide
    cross-checks between vertex and facet inputs and are used by smoke tests and
    datasets.
- `systolic_ratio(volume, capacity_ehz, symplectic_dimension)`
  - Computes the Viterbo systolic ratio when the volume and capacity are known.

## Stubs and planned work

- `minimal_action_cycle_lagrangian_product_generic(normals_q, offsets_q, normals_p, offsets_p)`
  - Future facet-interior solver allowing bounces away from vertices. Will
    replace the vertex-only restriction once the constrained optimiser lands.
- `capacity_ehz_haim_kislev(...)`, `capacity_ehz_via_qp(...)`,
  `capacity_ehz_via_lp(...)`, `oriented_edge_spectrum_4d(...)`
  - Higher-dimensional backends (4D and above) that remain on the roadmap.

## Notes for consumers

- Inputs must satisfy the even-dimension policy (`d = 2n`).
- Functions preserve dtype/device and rely on helpers from
  `viterbo.math.polytope` and `viterbo.math.constructions`.
- `datasets/atlas_tiny.py` uses the planar helpers to populate derived fields.
  When 4D solvers ship, migrate the dataset to the generic APIs.
